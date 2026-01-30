using System.CommandLine;
using Kwikread;

// Data directory for corrections
var dataDir = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
    "kwikread");

// Build the CLI
var rootCommand = new RootCommand("kwikread - Self-learning handwriting OCR");

// Process command
var processCommand = new Command("process", "Process an image file for OCR");
var imageArgument = new Argument<FileInfo>("image", "Path to the image file");
var outputOption = new Option<FileInfo?>(
    new[] { "-o", "--output" },
    "Output CSV file path");
var reviewOption = new Option<bool>(
    new[] { "-r", "--review" },
    () => true,
    "Enable interactive review mode");

processCommand.AddArgument(imageArgument);
processCommand.AddOption(outputOption);
processCommand.AddOption(reviewOption);

processCommand.SetHandler(async (image, output, review) =>
{
    await ProcessImageAsync(image, output, review, dataDir);
}, imageArgument, outputOption, reviewOption);

// Stats command
var statsCommand = new Command("stats", "Show OCR learning statistics");
statsCommand.SetHandler(() =>
{
    var tracker = new CorrectionTracker(dataDir);
    var stats = tracker.GetStats();

    Console.WriteLine("Learning Statistics");
    Console.WriteLine("===================");
    Console.WriteLine($"Total events:        {stats.TotalEvents}");
    Console.WriteLine($"Approvals:           {stats.Approvals}");
    Console.WriteLine($"Corrections:         {stats.Rejections}");
    Console.WriteLine($"Accuracy rate:       {stats.AccuracyRate:F1}%");
    if (stats.TotalEvents >= 10)
    {
        Console.WriteLine($"Recent accuracy:     {stats.RecentAccuracyRate:F1}% (last 10)");
    }
    Console.WriteLine();
    Console.WriteLine($"Training data ready: {stats.TrainingDataCount} corrections");
    if (stats.ReadyForTraining)
    {
        Console.WriteLine("Ready for fine-tuning! Run 'kwikread train' to start.");
    }
    else
    {
        Console.WriteLine($"Need {500 - stats.TrainingDataCount} more corrections before training.");
    }
});

// Train command
var trainCommand = new Command("train", "Trigger model fine-tuning on accumulated corrections");
trainCommand.SetHandler(() =>
{
    var tracker = new CorrectionTracker(dataDir);
    var stats = tracker.GetStats();

    if (!stats.ReadyForTraining)
    {
        Console.WriteLine($"Not enough training data yet.");
        Console.WriteLine($"Current: {stats.TrainingDataCount} corrections");
        Console.WriteLine($"Required: 500 corrections");
        return;
    }

    var trainingData = tracker.ExportTrainingData();
    Console.WriteLine($"Exporting {trainingData.Count} training pairs...");

    // Export to CSV for training
    var exportPath = Path.Combine(dataDir, "training_data.csv");
    using var writer = new StreamWriter(exportPath);
    writer.WriteLine("image_path,correct_text,original_ocr");
    foreach (var pair in trainingData)
    {
        writer.WriteLine($"\"{pair.ImagePath}\",\"{EscapeCsv(pair.CorrectText)}\",\"{EscapeCsv(pair.OriginalOcrText)}\"");
    }

    Console.WriteLine($"Training data exported to: {exportPath}");
    Console.WriteLine();
    Console.WriteLine("Fine-tuning would use Pavlov.Learning's NeuralNetworkRetrainingService pattern.");
    Console.WriteLine("Integration pending - training data is ready for manual fine-tuning.");
});

// Export command
var exportCommand = new Command("export", "Export training data to CSV");
var exportOutputOption = new Option<FileInfo>(
    new[] { "-o", "--output" },
    () => new FileInfo(Path.Combine(dataDir, "training_data.csv")),
    "Output CSV file path");
exportCommand.AddOption(exportOutputOption);
exportCommand.SetHandler((output) =>
{
    var tracker = new CorrectionTracker(dataDir);
    var trainingData = tracker.ExportTrainingData();

    if (trainingData.Count == 0)
    {
        Console.WriteLine("No training data available yet.");
        Console.WriteLine("Process images and make corrections to build training data.");
        return;
    }

    using var writer = new StreamWriter(output.FullName);
    writer.WriteLine("image_path,correct_text,original_ocr");
    foreach (var pair in trainingData)
    {
        writer.WriteLine($"\"{pair.ImagePath}\",\"{EscapeCsv(pair.CorrectText)}\",\"{EscapeCsv(pair.OriginalOcrText)}\"");
    }

    Console.WriteLine($"Exported {trainingData.Count} training pairs to: {output.FullName}");
}, exportOutputOption);

rootCommand.AddCommand(processCommand);
rootCommand.AddCommand(statsCommand);
rootCommand.AddCommand(trainCommand);
rootCommand.AddCommand(exportCommand);

return await rootCommand.InvokeAsync(args);

// Process image handler
static async Task ProcessImageAsync(FileInfo imageFile, FileInfo? outputFile, bool reviewMode, string dataDir)
{
    if (!imageFile.Exists)
    {
        Console.WriteLine($"Error: Image file not found: {imageFile.FullName}");
        return;
    }

    Console.WriteLine($"Processing: {imageFile.Name}");
    Console.WriteLine();

    // Find models directory
    var modelsDir = Path.Combine(AppContext.BaseDirectory, "Models");
    if (!Directory.Exists(modelsDir))
    {
        modelsDir = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location)!, "Models");
    }
    if (!Directory.Exists(modelsDir))
    {
        modelsDir = Path.Combine(Directory.GetCurrentDirectory(), "Models");
    }
    if (!Directory.Exists(modelsDir))
    {
        Console.WriteLine($"Error: Models directory not found. Expected at: {modelsDir}");
        Console.WriteLine("Please ensure ONNX models are in the Models/ directory.");
        return;
    }

    try
    {
        using var engine = new OcrEngine(modelsDir);
        var tracker = new CorrectionTracker(dataDir);

        Console.WriteLine("Running OCR...");
        var recognizedText = engine.RecognizeFromFile(imageFile.FullName);

        Console.WriteLine();
        Console.WriteLine("Recognized text:");
        Console.WriteLine("----------------");
        Console.WriteLine(recognizedText);
        Console.WriteLine("----------------");
        Console.WriteLine();

        var finalText = recognizedText;

        if (reviewMode)
        {
            Console.WriteLine("Review mode enabled. Press Enter to accept, or type corrections:");
            Console.Write("> ");
            var correction = Console.ReadLine();

            if (!string.IsNullOrWhiteSpace(correction))
            {
                finalText = correction;
                tracker.RecordCorrection(imageFile.FullName, recognizedText, correction);
                Console.WriteLine($"Correction recorded for training.");
            }
            else
            {
                tracker.RecordApproval(imageFile.FullName, recognizedText);
                Console.WriteLine("Text accepted and recorded.");
            }
        }

        // Output to CSV if requested
        if (outputFile != null)
        {
            var csvLine = $"\"{imageFile.Name}\",\"{EscapeCsv(finalText)}\"";

            var writeHeader = !outputFile.Exists;
            using var writer = new StreamWriter(outputFile.FullName, append: true);

            if (writeHeader)
            {
                await writer.WriteLineAsync("filename,text");
            }
            await writer.WriteLineAsync(csvLine);

            Console.WriteLine($"Saved to: {outputFile.FullName}");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error during OCR: {ex.Message}");
        if (ex.InnerException != null)
        {
            Console.WriteLine($"Inner error: {ex.InnerException.Message}");
        }
    }
}

static string EscapeCsv(string text)
{
    return text.Replace("\"", "\"\"").Replace("\n", " ").Replace("\r", "");
}
