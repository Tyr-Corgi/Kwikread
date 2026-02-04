using System.Text.Json;

namespace Kwikread;

/// <summary>
/// Tracks OCR corrections for learning.
/// Follows Pavlov.Learning patterns and can be integrated later.
/// Thread-safe implementation with file locking for concurrent access.
/// </summary>
public class CorrectionTracker
{
    private readonly string _dataPath;
    private CorrectionData _data;
    private readonly object _lock = new();
    private bool _loadFailed = false;

    public CorrectionTracker(string dataDirectory)
    {
        _dataPath = Path.Combine(dataDirectory, "corrections.json");
        _data = LoadData();
    }

    /// <summary>
    /// Records a correction (human rejection of OCR output)
    /// Thread-safe with file locking.
    /// </summary>
    public void RecordCorrection(string imagePath, string originalText, string correctedText)
    {
        var correction = new CorrectionRecord
        {
            Id = Guid.NewGuid().ToString(),
            ImagePath = imagePath,
            OriginalText = originalText,
            CorrectedText = correctedText,
            Timestamp = DateTime.UtcNow,
            EventType = "human_rejection"
        };

        lock (_lock)
        {
            _data.Corrections.Add(correction);
            _data.TotalCorrections++;
            SaveData();
        }
    }

    /// <summary>
    /// Records an approval (human accepted OCR output as-is)
    /// Thread-safe with file locking.
    /// </summary>
    public void RecordApproval(string imagePath, string recognizedText)
    {
        var approval = new CorrectionRecord
        {
            Id = Guid.NewGuid().ToString(),
            ImagePath = imagePath,
            OriginalText = recognizedText,
            CorrectedText = recognizedText,
            Timestamp = DateTime.UtcNow,
            EventType = "human_approval"
        };

        lock (_lock)
        {
            _data.Corrections.Add(approval);
            _data.TotalApprovals++;
            SaveData();
        }
    }

    /// <summary>
    /// Gets statistics about learning progress
    /// </summary>
    public LearningStats GetStats()
    {
        var total = _data.Corrections.Count;
        var approvals = _data.Corrections.Count(c => c.EventType == "human_approval");
        var rejections = _data.Corrections.Count(c => c.EventType == "human_rejection");

        // Calculate accuracy trend
        double recentAccuracy = 0;
        if (total >= 10)
        {
            var recent = _data.Corrections.TakeLast(10);
            recentAccuracy = recent.Count(c => c.EventType == "human_approval") * 10.0;
        }

        return new LearningStats
        {
            TotalEvents = total,
            Approvals = approvals,
            Rejections = rejections,
            AccuracyRate = total > 0 ? (approvals * 100.0 / total) : 0,
            RecentAccuracyRate = recentAccuracy,
            ReadyForTraining = rejections >= 500,
            TrainingDataCount = rejections
        };
    }

    /// <summary>
    /// Exports corrections as training data (image path, corrected text pairs)
    /// </summary>
    public List<TrainingPair> ExportTrainingData()
    {
        return _data.Corrections
            .Where(c => c.EventType == "human_rejection")
            .Select(c => new TrainingPair
            {
                ImagePath = c.ImagePath,
                CorrectText = c.CorrectedText,
                OriginalOcrText = c.OriginalText
            })
            .ToList();
    }

    private CorrectionData LoadData()
    {
        if (!File.Exists(_dataPath))
            return new CorrectionData();

        try
        {
            // Use FileShare.Read to allow concurrent readers while we read
            using var stream = new FileStream(_dataPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new StreamReader(stream);
            var json = reader.ReadToEnd();
            return JsonSerializer.Deserialize<CorrectionData>(json) ?? new CorrectionData();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to load correction data: {ex.Message}");
            _loadFailed = true;
            return new CorrectionData();
        }
    }

    private void SaveData()
    {
        var directory = Path.GetDirectoryName(_dataPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        // Create backup if original load failed to prevent data loss
        if (_loadFailed && File.Exists(_dataPath))
        {
            var backupPath = _dataPath + ".backup." + DateTime.UtcNow.ToString("yyyyMMddHHmmss");
            File.Copy(_dataPath, backupPath);
            Console.WriteLine($"Warning: Created backup at {backupPath} before overwriting potentially corrupted file");
        }

        var json = JsonSerializer.Serialize(_data, new JsonSerializerOptions { WriteIndented = true });

        // Use exclusive file lock to prevent concurrent writes from other processes
        using var stream = new FileStream(
            _dataPath,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None);  // Exclusive lock - no other process can read or write
        using var writer = new StreamWriter(stream);
        writer.Write(json);
    }
}

public class CorrectionData
{
    public List<CorrectionRecord> Corrections { get; set; } = new();
    public int TotalCorrections { get; set; }
    public int TotalApprovals { get; set; }
}

public class CorrectionRecord
{
    public string Id { get; set; } = "";
    public string ImagePath { get; set; } = "";
    public string OriginalText { get; set; } = "";
    public string CorrectedText { get; set; } = "";
    public DateTime Timestamp { get; set; }
    public string EventType { get; set; } = ""; // human_approval, human_rejection
}

public class LearningStats
{
    public int TotalEvents { get; set; }
    public int Approvals { get; set; }
    public int Rejections { get; set; }
    public double AccuracyRate { get; set; }
    public double RecentAccuracyRate { get; set; }
    public bool ReadyForTraining { get; set; }
    public int TrainingDataCount { get; set; }
}

public class TrainingPair
{
    public string ImagePath { get; set; } = "";
    public string CorrectText { get; set; } = "";
    public string OriginalOcrText { get; set; } = "";
}
