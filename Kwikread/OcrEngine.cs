using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Text.Json;

namespace Kwikread;

/// <summary>
/// TrOCR-based OCR engine using ONNX Runtime.
/// Supports both merged decoder (Xenova) and separate decoder (Optimum) formats.
/// </summary>
public class OcrEngine : IDisposable
{
    private readonly InferenceSession _encoderSession;
    private readonly InferenceSession _decoderSession;
    private readonly InferenceSession? _decoderWithPastSession; // For separate decoder format
    private readonly string[] _vocabulary;
    private readonly bool _useSeparateDecoders;

    // TrOCR image preprocessing constants
    private const int ImageSize = 384;
    private static readonly float[] ImageMean = { 0.5f, 0.5f, 0.5f };
    private static readonly float[] ImageStd = { 0.5f, 0.5f, 0.5f };

    // Model dimensions (defaults for base model, updated from config if available)
    private int _numDecoderLayers = 12;
    private int _numDecoderHeads = 16;
    private int _decoderHeadDim = 64;
    private int _encoderHiddenSize = 768;
    private int _decoderStartTokenId = 2;
    private int _eosTokenId = 2;
    private int _padTokenId = 1;
    private const int MaxLength = 128;

    public OcrEngine(string modelDirectory)
    {
        var encoderPath = Path.Combine(modelDirectory, "encoder_model.onnx");
        var decoderPath = Path.Combine(modelDirectory, "decoder_model.onnx");
        var decoderWithPastPath = Path.Combine(modelDirectory, "decoder_with_past_model.onnx");
        var vocabPath = Path.Combine(modelDirectory, "vocab.json");
        var configPath = Path.Combine(modelDirectory, "config.json");
        var genConfigPath = Path.Combine(modelDirectory, "generation_config.json");

        if (!File.Exists(encoderPath))
            throw new FileNotFoundException($"Encoder model not found: {encoderPath}");
        if (!File.Exists(decoderPath))
            throw new FileNotFoundException($"Decoder model not found: {decoderPath}");

        // Check if we have separate decoders (Optimum format) or merged (Xenova format)
        _useSeparateDecoders = File.Exists(decoderWithPastPath);

        // Load model config if available
        LoadModelConfig(configPath, genConfigPath);

        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            InterOpNumThreads = Environment.ProcessorCount,
            IntraOpNumThreads = Environment.ProcessorCount
        };

        Console.WriteLine("Loading encoder model...");
        _encoderSession = new InferenceSession(encoderPath, sessionOptions);

        Console.WriteLine("Loading decoder model...");
        _decoderSession = new InferenceSession(decoderPath, sessionOptions);

        if (_useSeparateDecoders)
        {
            Console.WriteLine("Loading decoder with past model (separate decoder format)...");
            _decoderWithPastSession = new InferenceSession(decoderWithPastPath, sessionOptions);
        }

        // Load vocabulary
        _vocabulary = LoadVocabulary(vocabPath);

        Console.WriteLine($"OCR Engine loaded. Vocabulary size: {_vocabulary.Length}");
        Console.WriteLine($"Model config: {_numDecoderLayers} layers, {_numDecoderHeads} heads, encoder hidden={_encoderHiddenSize}");
        Console.WriteLine($"Decoder format: {(_useSeparateDecoders ? "separate (Optimum)" : "merged (Xenova)")}");
    }

    private void LoadModelConfig(string configPath, string genConfigPath)
    {
        // Load generation config for special tokens
        if (File.Exists(genConfigPath))
        {
            try
            {
                var genConfig = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(genConfigPath));
                if (genConfig.TryGetProperty("decoder_start_token_id", out var dstId))
                    _decoderStartTokenId = dstId.GetInt32();
                if (genConfig.TryGetProperty("eos_token_id", out var eosId))
                    _eosTokenId = eosId.GetInt32();
                if (genConfig.TryGetProperty("pad_token_id", out var padId))
                    _padTokenId = padId.GetInt32();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to parse generation config '{genConfigPath}': {ex.Message}. Using defaults.");
            }
        }

        // Load model architecture config
        if (File.Exists(configPath))
        {
            try
            {
                var config = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(configPath));

                // Get encoder hidden size
                if (config.TryGetProperty("encoder", out var encoder))
                {
                    if (encoder.TryGetProperty("hidden_size", out var hs))
                        _encoderHiddenSize = hs.GetInt32();
                }

                // Get decoder dimensions
                if (config.TryGetProperty("decoder", out var decoder))
                {
                    if (decoder.TryGetProperty("decoder_layers", out var layers))
                        _numDecoderLayers = layers.GetInt32();
                    if (decoder.TryGetProperty("decoder_attention_heads", out var heads))
                        _numDecoderHeads = heads.GetInt32();
                    if (decoder.TryGetProperty("d_model", out var dModel))
                        _decoderHeadDim = dModel.GetInt32() / _numDecoderHeads;
                }

                // Use root-level decoder_start_token_id (0 for this model)
                // Note: this differs from decoder section (2) but matches what Xenova expects
                if (config.TryGetProperty("decoder_start_token_id", out var dst))
                    _decoderStartTokenId = dst.GetInt32();
                if (config.TryGetProperty("eos_token_id", out var eos))
                    _eosTokenId = eos.GetInt32();
                if (config.TryGetProperty("pad_token_id", out var pad))
                    _padTokenId = pad.GetInt32();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to parse model config '{configPath}': {ex.Message}. Using defaults.");
            }
        }
    }

    /// <summary>
    /// Recognize text from an image file.
    /// Automatically detects orientation and corrects rotated images.
    /// Automatically detects if image contains multiple lines and segments them.
    /// </summary>
    public string RecognizeFromFile(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);

        // Detect and correct image orientation
        Console.WriteLine("Detecting image orientation...");
        var orientationResult = OrientationDetector.DetectOrientation(image);

        if (orientationResult.NeedsRotation)
        {
            Console.WriteLine($"Image orientation: {orientationResult.Orientation}");
            using var correctedImage = OrientationDetector.CorrectOrientation(image, orientationResult);
            return RecognizeMultiLine(correctedImage);
        }

        return RecognizeMultiLine(image);
    }

    /// <summary>
    /// Recognize text from a multi-line image.
    /// Segments the image into lines and processes each separately.
    /// </summary>
    public string RecognizeMultiLine(Image<Rgb24> image)
    {
        // Check if image is likely a single line or multi-line
        const int SingleLineHeightThreshold = 400;
        
        if (image.Height <= SingleLineHeightThreshold)
        {
            // Likely a single line, process directly
            Console.WriteLine("Processing as single-line image...");
            return Recognize(image);
        }
        
        // Multi-line image - segment into lines
        Console.WriteLine("Processing as multi-line image...");
        Console.WriteLine("Segmenting lines...");
        
        using var preprocessed = LineSegmenter.PreprocessImage(image);
        var lines = LineSegmenter.SegmentLines(preprocessed);
        
        if (lines.Count == 0)
        {
            Console.WriteLine("Warning: No lines detected. Processing entire image as single line.");
            return Recognize(image);
        }
        
        // Process each line
        var results = new List<string>();
        for (int i = 0; i < lines.Count; i++)
        {
            Console.WriteLine($"\nProcessing line {i + 1}/{lines.Count}...");

            try
            {
                // Skip blank/empty regions to avoid hallucinations
                if (!HasSufficientContent(lines[i]))
                {
                    Console.WriteLine($"  Skipping (insufficient text content)");
                    continue;
                }

                var lineText = Recognize(lines[i]);

                // Clean up the text
                lineText = lineText.Replace("<s>", "").Trim();

                if (!string.IsNullOrWhiteSpace(lineText))
                {
                    results.Add(lineText);
                    Console.WriteLine($"  → {lineText}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Error processing line {i + 1}: {ex.Message}");
            }
            finally
            {
                lines[i].Dispose();
            }
        }

        return string.Join("\n", results);
    }

    /// <summary>
    /// Check if an image has sufficient text content (not blank/empty).
    /// Uses adaptive thresholding to handle both dark and light text.
    /// </summary>
    private bool HasSufficientContent(Image<Rgb24> image)
    {
        const double MinVarianceThreshold = 500.0;  // Minimum variance for "has content"

        double sum = 0;
        double sumSq = 0;
        int totalPixels = 0;
        var brightnesses = new List<double>();

        // Sample every 4th pixel for speed
        for (int y = 0; y < image.Height; y += 4)
        {
            for (int x = 0; x < image.Width; x += 4)
            {
                var pixel = image[x, y];
                var brightness = (pixel.R + pixel.G + pixel.B) / 3.0;
                sum += brightness;
                sumSq += brightness * brightness;
                totalPixels++;
                brightnesses.Add(brightness);
            }
        }

        if (totalPixels == 0) return false;

        double mean = sum / totalPixels;
        double variance = (sumSq / totalPixels) - (mean * mean);

        // Adaptive threshold: count pixels significantly darker than the mean
        // This catches both dark text on white and light text on white
        double adaptiveThreshold = mean - 50; // Pixels at least 50 units darker than mean
        int contrastPixels = brightnesses.Count(b => b < adaptiveThreshold);
        double contrastRatio = (double)contrastPixels / totalPixels;

        // For high-mean regions (light background), require more contrast evidence
        // Low-mean regions (darker overall) likely have real content
        double requiredContrast = mean > 200 ? 0.02 : 0.01;

        // Need: variance (texture) AND contrast pixels (text darker than background)
        bool hasContent = variance > MinVarianceThreshold && contrastRatio > requiredContrast;
        return hasContent;
    }

    /// <summary>
    /// Recognize text from a single-line image.
    /// </summary>
    public string Recognize(Image<Rgb24> image)
    {
        // Preprocess image
        var pixelValues = PreprocessImage(image);

        // Run encoder
        var encoderOutput = RunEncoder(pixelValues);

        // Run decoder with autoregressive generation
        var tokenIds = GenerateTokens(encoderOutput);

        // Decode tokens to text
        return DecodeTokens(tokenIds);
    }

    private float[] PreprocessImage(Image<Rgb24> image)
    {
        // Enhance contrast to make text more visible
        using var enhanced = image.Clone(x => x
            .Contrast(1.3f)      // Increase contrast
            .Brightness(1.05f)); // Slight brightness boost

        // Squish resize to 384x384 (matches HuggingFace ViTImageProcessor behavior)
        // TrOCR was trained with squished images, so we must match that
        using var resizedImage = enhanced.Clone(x => x.Resize(ImageSize, ImageSize));

        // Convert to float tensor with normalization
        var pixels = new float[3 * ImageSize * ImageSize];

        for (int y = 0; y < ImageSize; y++)
        {
            for (int x = 0; x < ImageSize; x++)
            {
                var pixel = resizedImage[x, y];
                var idx = y * ImageSize + x;

                // Normalize: (pixel / 255 - mean) / std
                pixels[0 * ImageSize * ImageSize + idx] = ((pixel.R / 255f) - ImageMean[0]) / ImageStd[0];
                pixels[1 * ImageSize * ImageSize + idx] = ((pixel.G / 255f) - ImageMean[1]) / ImageStd[1];
                pixels[2 * ImageSize * ImageSize + idx] = ((pixel.B / 255f) - ImageMean[2]) / ImageStd[2];
            }
        }

        return pixels;
    }

    private int _encoderSeqLength = 577; // Will be updated from actual encoder output

    private float[] RunEncoder(float[] pixelValues)
    {
        var inputTensor = new DenseTensor<float>(pixelValues, new[] { 1, 3, ImageSize, ImageSize });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", inputTensor)
        };

        using var results = _encoderSession.Run(inputs);
        var output = results.First().AsTensor<float>();
        
        // Get actual dimensions from encoder output
        _encoderSeqLength = output.Dimensions[1];
        
        return output.ToArray();
    }

    private List<int> GenerateTokens(float[] encoderOutput)
    {
        if (_useSeparateDecoders)
            return GenerateTokensSeparateDecoders(encoderOutput);
        else
            return GenerateTokensMergedDecoder(encoderOutput);
    }

    /// <summary>
    /// Generate tokens using separate decoder models (Optimum export format).
    /// Uses decoder_model.onnx for first token, decoder_with_past_model.onnx for subsequent tokens.
    /// </summary>
    private List<int> GenerateTokensSeparateDecoders(float[] encoderOutput)
    {
        var generatedIds = new List<int> { _decoderStartTokenId };
        var encoderDims = new[] { 1, _encoderSeqLength, _encoderHiddenSize };

        const float RepetitionPenalty = 1.2f;
        const int RepetitionWindowSize = 10;
        const int MaxGeneratedLength = 50;

        // KV cache storage - stores both decoder and encoder caches
        var decoderCache = new Dictionary<string, float[]>();
        var encoderCache = new Dictionary<string, float[]>();
        int decoderCachedSeqLen = 0;

        for (int i = 0; i < MaxGeneratedLength; i++)
        {
            float[] lastTokenLogits;

            if (i == 0)
            {
                // First token: use decoder_model.onnx (no past key values)
                var inputIdsTensor = new DenseTensor<long>(new[] { (long)_decoderStartTokenId }, new[] { 1, 1 });
                var encoderTensor = new DenseTensor<float>(encoderOutput, encoderDims);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                    NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderTensor)
                };

                try
                {
                    using var results = _decoderSession.Run(inputs);
                    var resultsList = results.ToList();
                    var logits = resultsList.First(r => r.Name == "logits").AsTensor<float>();

                    // Cache present key values for next iteration
                    foreach (var result in resultsList)
                    {
                        if (result.Name.StartsWith("present"))
                        {
                            var tensor = result.AsTensor<float>();
                            var data = tensor.ToArray();

                            if (result.Name.Contains(".decoder."))
                                decoderCache[result.Name] = data;
                            else if (result.Name.Contains(".encoder."))
                                encoderCache[result.Name] = data;
                        }
                    }
                    decoderCachedSeqLen = 1;

                    // Get logits
                    var vocabSize = logits.Dimensions[2];
                    lastTokenLogits = new float[vocabSize];
                    for (int j = 0; j < vocabSize; j++)
                        lastTokenLogits[j] = logits[0, 0, j];
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"First token error: {ex.Message}");
                    break;
                }
            }
            else
            {
                // Subsequent tokens: use decoder_with_past_model.onnx
                var inputIdsTensor = new DenseTensor<long>(new[] { (long)generatedIds[^1] }, new[] { 1, 1 });

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor)
                };

                // Add past key values from cache - both decoder and encoder caches
                for (int layer = 0; layer < _numDecoderLayers; layer++)
                {
                    // Decoder self-attention cache
                    var decoderKeyName = $"past_key_values.{layer}.decoder.key";
                    var decoderValueName = $"past_key_values.{layer}.decoder.value";
                    var presentDecoderKeyName = $"present.{layer}.decoder.key";
                    var presentDecoderValueName = $"present.{layer}.decoder.value";

                    if (decoderCache.TryGetValue(presentDecoderKeyName, out var dkData))
                    {
                        var keyClone = (float[])dkData.Clone();
                        inputs.Add(NamedOnnxValue.CreateFromTensor(decoderKeyName,
                            new DenseTensor<float>(keyClone, new[] { 1, _numDecoderHeads, decoderCachedSeqLen, _decoderHeadDim })));
                    }
                    if (decoderCache.TryGetValue(presentDecoderValueName, out var dvData))
                    {
                        var valueClone = (float[])dvData.Clone();
                        inputs.Add(NamedOnnxValue.CreateFromTensor(decoderValueName,
                            new DenseTensor<float>(valueClone, new[] { 1, _numDecoderHeads, decoderCachedSeqLen, _decoderHeadDim })));
                    }

                    // Encoder cross-attention cache (static, computed on first step)
                    var encoderKeyName = $"past_key_values.{layer}.encoder.key";
                    var encoderValueName = $"past_key_values.{layer}.encoder.value";
                    var presentEncoderKeyName = $"present.{layer}.encoder.key";
                    var presentEncoderValueName = $"present.{layer}.encoder.value";

                    if (encoderCache.TryGetValue(presentEncoderKeyName, out var ekData))
                    {
                        var keyClone = (float[])ekData.Clone();
                        inputs.Add(NamedOnnxValue.CreateFromTensor(encoderKeyName,
                            new DenseTensor<float>(keyClone, new[] { 1, _numDecoderHeads, _encoderSeqLength, _decoderHeadDim })));
                    }
                    if (encoderCache.TryGetValue(presentEncoderValueName, out var evData))
                    {
                        var valueClone = (float[])evData.Clone();
                        inputs.Add(NamedOnnxValue.CreateFromTensor(encoderValueName,
                            new DenseTensor<float>(valueClone, new[] { 1, _numDecoderHeads, _encoderSeqLength, _decoderHeadDim })));
                    }
                }

                try
                {
                    using var results = _decoderWithPastSession!.Run(inputs);
                    var resultsList = results.ToList();
                    var logits = resultsList.First(r => r.Name == "logits").AsTensor<float>();

                    // Update decoder cache (encoder cache stays the same)
                    foreach (var result in resultsList)
                    {
                        if (result.Name.StartsWith("present") && result.Name.Contains(".decoder."))
                        {
                            var tensor = result.AsTensor<float>();
                            decoderCache[result.Name] = tensor.ToArray();
                        }
                    }
                    decoderCachedSeqLen++;

                    // Get logits
                    var vocabSize = logits.Dimensions[2];
                    lastTokenLogits = new float[vocabSize];
                    for (int j = 0; j < vocabSize; j++)
                        lastTokenLogits[j] = logits[0, 0, j];
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Token {i} error: {ex.Message}");
                    break;
                }
            }

            // Apply repetition penalty
            var recentWindow = generatedIds.Skip(Math.Max(0, generatedIds.Count - RepetitionWindowSize));
            foreach (var tokenId in recentWindow)
            {
                if (tokenId >= 0 && tokenId < lastTokenLogits.Length)
                    lastTokenLogits[tokenId] /= RepetitionPenalty;
            }

            var nextToken = Array.IndexOf(lastTokenLogits, lastTokenLogits.Max());

            if (nextToken == _eosTokenId || nextToken == _padTokenId)
                break;

            generatedIds.Add(nextToken);
        }

        return generatedIds;
    }

    /// <summary>
    /// Generate tokens using merged decoder model (Xenova export format).
    /// Uses a single decoder_model.onnx with use_cache_branch to switch modes.
    /// </summary>
    private List<int> GenerateTokensMergedDecoder(float[] encoderOutput)
    {
        var generatedIds = new List<int> { _decoderStartTokenId };
        var encoderDims = new[] { 1, _encoderSeqLength, _encoderHiddenSize };

        const float RepetitionPenalty = 1.2f;
        const int RepetitionWindowSize = 10;
        const int MaxGeneratedLength = 50;

        var decoderKVCache = new Dictionary<string, float[]>();
        var encoderKVCache = new Dictionary<string, float[]>();
        int cachedSeqLen = 0;

        for (int i = 0; i < MaxGeneratedLength; i++)
        {
            long[] inputIdsData = i == 0
                ? new[] { (long)_decoderStartTokenId }
                : new[] { (long)generatedIds[^1] };

            var inputIdsTensor = new DenseTensor<long>(inputIdsData, new[] { 1, inputIdsData.Length });
            var encoderTensor = new DenseTensor<float>(encoderOutput, encoderDims);
            var useCacheBranch = new DenseTensor<bool>(new[] { i > 0 }, new[] { 1 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderTensor),
                NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheBranch)
            };

            // Add KV cache tensors
            for (int layer = 0; layer < _numDecoderLayers; layer++)
            {
                var decoderKeyName = $"past_key_values.{layer}.decoder.key";
                var decoderValueName = $"past_key_values.{layer}.decoder.value";
                var encoderKeyName = $"past_key_values.{layer}.encoder.key";
                var encoderValueName = $"past_key_values.{layer}.encoder.value";

                if (i == 0)
                {
                    var emptyDecoder = new float[0];
                    var emptyEncoder = new float[0];
                    inputs.Add(NamedOnnxValue.CreateFromTensor(decoderKeyName,
                        new DenseTensor<float>(emptyDecoder, new[] { 1, _numDecoderHeads, 0, _decoderHeadDim })));
                    inputs.Add(NamedOnnxValue.CreateFromTensor(decoderValueName,
                        new DenseTensor<float>(emptyDecoder, new[] { 1, _numDecoderHeads, 0, _decoderHeadDim })));
                    inputs.Add(NamedOnnxValue.CreateFromTensor(encoderKeyName,
                        new DenseTensor<float>(emptyEncoder, new[] { 1, _numDecoderHeads, 0, _decoderHeadDim })));
                    inputs.Add(NamedOnnxValue.CreateFromTensor(encoderValueName,
                        new DenseTensor<float>(emptyEncoder, new[] { 1, _numDecoderHeads, 0, _decoderHeadDim })));
                }
                else
                {
                    var decoderKeyData = (float[])decoderKVCache[decoderKeyName].Clone();
                    var decoderValueData = (float[])decoderKVCache[decoderValueName].Clone();
                    var encoderKeyData = (float[])encoderKVCache[encoderKeyName].Clone();
                    var encoderValueData = (float[])encoderKVCache[encoderValueName].Clone();

                    inputs.Add(NamedOnnxValue.CreateFromTensor(decoderKeyName,
                        new DenseTensor<float>(decoderKeyData, new[] { 1, _numDecoderHeads, cachedSeqLen, _decoderHeadDim })));
                    inputs.Add(NamedOnnxValue.CreateFromTensor(decoderValueName,
                        new DenseTensor<float>(decoderValueData, new[] { 1, _numDecoderHeads, cachedSeqLen, _decoderHeadDim })));
                    inputs.Add(NamedOnnxValue.CreateFromTensor(encoderKeyName,
                        new DenseTensor<float>(encoderKeyData, new[] { 1, _numDecoderHeads, _encoderSeqLength, _decoderHeadDim })));
                    inputs.Add(NamedOnnxValue.CreateFromTensor(encoderValueName,
                        new DenseTensor<float>(encoderValueData, new[] { 1, _numDecoderHeads, _encoderSeqLength, _decoderHeadDim })));
                }
            }

            try
            {
                using var results = _decoderSession.Run(inputs);
                var resultsList = results.ToList();
                var logits = resultsList.First(r => r.Name == "logits").AsTensor<float>();

                foreach (var result in resultsList)
                {
                    if (result.Name.StartsWith("present."))
                    {
                        var tensor = result.AsTensor<float>();
                        var data = tensor.ToArray();
                        var cacheKey = result.Name.Replace("present.", "past_key_values.");

                        if (result.Name.Contains(".decoder."))
                            decoderKVCache[cacheKey] = data;
                        else if (result.Name.Contains(".encoder.") && i == 0)
                            encoderKVCache[cacheKey] = data;
                    }
                }

                cachedSeqLen = i + 1;

                var vocabSize = logits.Dimensions[2];
                var lastTokenLogits = new float[vocabSize];
                var lastIdx = logits.Dimensions[1] - 1;
                for (int j = 0; j < vocabSize; j++)
                    lastTokenLogits[j] = logits[0, lastIdx, j];

                var recentWindow = generatedIds.Skip(Math.Max(0, generatedIds.Count - RepetitionWindowSize));
                foreach (var tokenId in recentWindow)
                {
                    if (tokenId >= 0 && tokenId < lastTokenLogits.Length)
                        lastTokenLogits[tokenId] /= RepetitionPenalty;
                }

                var nextToken = Array.IndexOf(lastTokenLogits, lastTokenLogits.Max());

                if (nextToken == _eosTokenId || nextToken == _padTokenId)
                    break;

                generatedIds.Add(nextToken);
            }
            catch (Exception)
            {
                break;
            }
        }

        return generatedIds;
    }

    private string DecodeTokens(List<int> tokenIds)
    {
        var tokens = tokenIds
            .Skip(1) // Skip decoder start token
            .Where(id => id != _eosTokenId && id != _padTokenId && id < _vocabulary.Length)
            .Where(id => _vocabulary[id] != null)
            .Select(id => _vocabulary[id])
            .ToList();

        // Join tokens and clean up BPE artifacts
        var text = string.Join("", tokens)
            .Replace("</w>", " ")
            .Replace("Ġ", " ")
            .Replace("▁", " ")  // SentencePiece space marker (used by small model)
            .Trim();

        return text;
    }

    private string[] LoadVocabulary(string vocabPath)
    {
        if (!File.Exists(vocabPath))
        {
            Console.WriteLine($"Warning: Vocabulary file not found at {vocabPath}. Using placeholder.");
            // Return a basic placeholder - in production, this should be the actual vocabulary
            return Enumerable.Range(0, 50265).Select(i => $"[{i}]").ToArray();
        }

        try
        {
            var json = File.ReadAllText(vocabPath);
            // Parse vocab.json - it's typically a dict of token -> id
            // For simplicity, we'll need to invert it
            var vocab = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, int>>(json);
            if (vocab == null || vocab.Count == 0) return Array.Empty<string>();

            var result = new string[vocab.Values.Max() + 1];
            foreach (var kvp in vocab)
            {
                result[kvp.Value] = kvp.Key;
            }
            return result;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to load vocabulary: {ex.Message}");
            return Enumerable.Range(0, 50265).Select(i => $"[{i}]").ToArray();
        }
    }

    public void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
        _decoderWithPastSession?.Dispose();
    }
}
