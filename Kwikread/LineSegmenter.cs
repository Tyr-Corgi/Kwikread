using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Kwikread;

/// <summary>
/// Segments a full-page image into individual text lines for OCR processing.
/// Uses adaptive detection based on actual content - no fixed line height assumptions.
/// </summary>
public class LineSegmenter
{
    private const int AbsoluteMinLineHeight = 15;  // Absolute minimum (tiny text)
    private const int AbsoluteMaxLineHeight = 500; // Absolute maximum (huge text)
    private const int ProjectionSmoothWindow = 3;  // Smaller window for finer detection

    /// <summary>
    /// Segments an image into individual text lines.
    /// Returns a list of cropped line images.
    /// </summary>
    public static List<Image<Rgb24>> SegmentLines(Image<Rgb24> image)
    {
        var lines = new List<Image<Rgb24>>();

        // Calculate horizontal projection
        var rawProjection = CalculateHorizontalProjection(image);
        var projection = SmoothProjection(rawProjection);

        // Detect line boundaries adaptively from the projection
        var lineRegions = DetectLinesAdaptively(projection, image.Height);

        Console.WriteLine($"Detected {lineRegions.Count} text lines");

        // Extract line images
        foreach (var (top, bottom) in lineRegions)
        {
            var cropHeight = bottom - top;
            if (cropHeight < AbsoluteMinLineHeight) continue;

            var lineImage = image.Clone(ctx => ctx.Crop(new Rectangle(0, top, image.Width, cropHeight)));
            lines.Add(lineImage);

            Console.WriteLine($"  Line {lines.Count}: y={top}-{bottom} ({cropHeight}px)");
        }

        if (lines.Count == 0)
        {
            Console.WriteLine("No lines detected. Processing entire image as single region.");
            lines.Add(image.Clone());
        }

        return lines;
    }

    /// <summary>
    /// Adaptively detect text lines by finding gaps (peaks) in the horizontal projection.
    /// No fixed assumptions about line height - detects from actual content.
    /// </summary>
    private static List<(int top, int bottom)> DetectLinesAdaptively(double[] projection, int height)
    {
        // Step 1: Find all significant gaps (local maxima in projection)
        var gaps = FindSignificantGaps(projection, height);

        if (gaps.Count == 0)
        {
            // No clear gaps found - treat entire image as one region
            return new List<(int top, int bottom)> { (0, height) };
        }

        // Step 2: Analyze gap spacing to understand the line structure
        var gapSpacings = new List<int>();
        for (int i = 1; i < gaps.Count; i++)
        {
            gapSpacings.Add(gaps[i] - gaps[i - 1]);
        }

        // Detect the typical line height from gap spacing
        int detectedLineHeight = gapSpacings.Count > 0
            ? (int)gapSpacings.OrderBy(x => x).ElementAt(gapSpacings.Count / 2) // Median
            : height / 10;

        Console.WriteLine($"Detected line height: ~{detectedLineHeight}px (from {gaps.Count} gaps)");

        // Step 3: Build line regions from gaps
        var regions = new List<(int top, int bottom)>();

        // Add region before first gap (if there's content there)
        if (gaps[0] > AbsoluteMinLineHeight)
        {
            AddRegionsFromRange(regions, projection, 0, gaps[0], detectedLineHeight);
        }

        // Add regions between consecutive gaps
        for (int i = 0; i < gaps.Count - 1; i++)
        {
            int regionTop = gaps[i];
            int regionBottom = gaps[i + 1];
            int regionHeight = regionBottom - regionTop;

            if (regionHeight < AbsoluteMinLineHeight)
                continue;

            // If region is much larger than detected line height, subdivide it
            if (regionHeight > detectedLineHeight * 1.8 && detectedLineHeight > AbsoluteMinLineHeight)
            {
                AddRegionsFromRange(regions, projection, regionTop, regionBottom, detectedLineHeight);
            }
            else
            {
                regions.Add((regionTop, regionBottom));
            }
        }

        // Add region after last gap (if there's content there)
        if (height - gaps[gaps.Count - 1] > AbsoluteMinLineHeight)
        {
            AddRegionsFromRange(regions, projection, gaps[gaps.Count - 1], height, detectedLineHeight);
        }

        return regions;
    }

    /// <summary>
    /// Find significant gaps (whitespace between text lines) in the projection.
    /// Uses adaptive thresholding based on the projection's characteristics.
    /// </summary>
    private static List<int> FindSignificantGaps(double[] projection, int height)
    {
        if (projection.Length == 0) return new List<int>();

        // Sort projection values to find percentiles
        var sorted = projection.OrderBy(x => x).ToArray();
        double p25 = sorted[(int)(sorted.Length * 0.25)];
        double p50 = sorted[(int)(sorted.Length * 0.50)]; // median
        double p75 = sorted[(int)(sorted.Length * 0.75)];
        double min = sorted[0];
        double max = sorted[sorted.Length - 1];
        double range = max - min;

        // If there's very little variation, no clear lines exist
        if (range < 0.05)
        {
            Console.WriteLine("Low projection variance - unclear line structure");
            return new List<int>();
        }

        // Gap threshold: between median and 75th percentile
        // This adapts to the actual distribution of whitespace in the image
        double gapThreshold = p50 + (p75 - p50) * 0.5;

        // Find local maxima (peaks) that represent gaps
        var candidates = new List<(int y, double value, double prominence)>();
        const int WindowSize = 4;

        for (int y = WindowSize; y < height - WindowSize; y++)
        {
            double val = projection[y];
            if (val < gapThreshold) continue;

            // Check if local maximum
            bool isMax = true;
            double minNeighbor = double.MaxValue;
            for (int dy = -WindowSize; dy <= WindowSize; dy++)
            {
                if (dy != 0)
                {
                    if (projection[y + dy] > val)
                        isMax = false;
                    minNeighbor = Math.Min(minNeighbor, projection[y + dy]);
                }
            }

            if (isMax)
            {
                // Prominence = how much higher than neighbors
                double prominence = val - minNeighbor;
                candidates.Add((y, val, prominence));
            }
        }

        // Filter by prominence (keep peaks that stand out)
        double avgProminence = candidates.Count > 0 ? candidates.Average(c => c.prominence) : 0;
        var significantCandidates = candidates
            .Where(c => c.prominence >= avgProminence * 0.3)
            .OrderByDescending(c => c.value)
            .ToList();

        // Keep well-spaced gaps
        var gaps = new List<int>();
        int minGapDistance = Math.Max(15, height / 100); // At least 15px or 1% of height

        foreach (var (y, value, prominence) in significantCandidates)
        {
            bool tooClose = gaps.Any(existing => Math.Abs(y - existing) < minGapDistance);
            if (!tooClose)
                gaps.Add(y);
        }

        gaps.Sort();
        return gaps;
    }

    /// <summary>
    /// Subdivide a range into lines based on detected line height.
    /// Uses projection to find optimal boundaries.
    /// </summary>
    private static void AddRegionsFromRange(
        List<(int top, int bottom)> regions,
        double[] projection,
        int rangeTop,
        int rangeBottom,
        int targetLineHeight)
    {
        int rangeHeight = rangeBottom - rangeTop;

        // If range is small enough, add as single region
        if (rangeHeight <= targetLineHeight * 1.5 || targetLineHeight < AbsoluteMinLineHeight)
        {
            if (rangeHeight >= AbsoluteMinLineHeight)
                regions.Add((rangeTop, rangeBottom));
            return;
        }

        // Estimate number of lines in this range
        int numLines = Math.Max(1, (int)Math.Round((double)rangeHeight / targetLineHeight));
        int lineHeight = rangeHeight / numLines;

        // Find optimal boundaries using projection
        var boundaries = new List<int> { rangeTop };

        for (int i = 1; i < numLines; i++)
        {
            int estimatedBoundary = rangeTop + i * lineHeight;

            // Search for best gap near estimated boundary
            int searchRadius = Math.Max(5, lineHeight / 4);
            int searchStart = Math.Max(rangeTop, estimatedBoundary - searchRadius);
            int searchEnd = Math.Min(rangeBottom, estimatedBoundary + searchRadius);

            int bestY = estimatedBoundary;
            double bestVal = -1;

            for (int y = searchStart; y < searchEnd && y < projection.Length; y++)
            {
                if (projection[y] > bestVal)
                {
                    bestVal = projection[y];
                    bestY = y;
                }
            }

            boundaries.Add(bestY);
        }

        boundaries.Add(rangeBottom);

        // Create regions from boundaries
        for (int i = 0; i < boundaries.Count - 1; i++)
        {
            int top = boundaries[i];
            int bottom = boundaries[i + 1];
            if (bottom - top >= AbsoluteMinLineHeight)
                regions.Add((top, bottom));
        }
    }

    /// <summary>
    /// Smooth projection with a moving average to reduce noise.
    /// </summary>
    private static double[] SmoothProjection(double[] projection)
    {
        if (projection.Length == 0) return projection;
        var half = ProjectionSmoothWindow / 2;
        var result = new double[projection.Length];

        for (int i = 0; i < projection.Length; i++)
        {
            double sum = 0;
            int count = 0;
            for (int j = Math.Max(0, i - half); j <= Math.Min(projection.Length - 1, i + half); j++)
            {
                sum += projection[j];
                count++;
            }
            result[i] = count > 0 ? sum / count : projection[i];
        }
        return result;
    }

    /// <summary>
    /// Calculate horizontal projection (ratio of white pixels per row).
    /// Higher values = more whitespace = likely gap between lines.
    /// </summary>
    private static double[] CalculateHorizontalProjection(Image<Rgb24> image)
    {
        var projection = new double[image.Height];

        // Calculate adaptive threshold
        var threshold = CalculateAdaptiveThreshold(image);
        Console.WriteLine($"Brightness threshold: {threshold:F2}");

        for (int y = 0; y < image.Height; y++)
        {
            int whitePixels = 0;

            for (int x = 0; x < image.Width; x++)
            {
                var pixel = image[x, y];
                var brightness = (pixel.R + pixel.G + pixel.B) / 3.0 / 255.0;

                if (brightness > threshold)
                    whitePixels++;
            }

            projection[y] = (double)whitePixels / image.Width;
        }

        return projection;
    }

    /// <summary>
    /// Calculate adaptive brightness threshold to separate text from background.
    /// </summary>
    private static double CalculateAdaptiveThreshold(Image<Rgb24> image)
    {
        // Sample pixels for histogram
        var samples = new List<double>();
        int step = Math.Max(1, Math.Min(image.Width, image.Height) / 100);

        for (int y = 0; y < image.Height; y += step)
        {
            for (int x = 0; x < image.Width; x += step)
            {
                var pixel = image[x, y];
                var brightness = (pixel.R + pixel.G + pixel.B) / 3.0 / 255.0;
                samples.Add(brightness);
            }
        }

        if (samples.Count == 0) return 0.5;

        samples.Sort();

        // Find the gap between dark (text) and light (background)
        var darkPixels = samples.Where(b => b < 0.5).ToList();
        var lightPixels = samples.Where(b => b >= 0.5).ToList();

        if (darkPixels.Count > 0 && lightPixels.Count > 0)
        {
            var maxDark = darkPixels.Max();
            var minLight = lightPixels.Min();
            return (maxDark + minLight) / 2;
        }

        // Fallback to median
        return samples[samples.Count / 2];
    }

    /// <summary>
    /// Preprocess image before line detection.
    /// </summary>
    public static Image<Rgb24> PreprocessImage(Image<Rgb24> image)
    {
        var processed = image.Clone();

        // Resize if very large
        const int MaxWidth = 2000;
        if (processed.Width > MaxWidth)
        {
            var scale = (double)MaxWidth / processed.Width;
            var newHeight = (int)(processed.Height * scale);
            processed.Mutate(x => x.Resize(MaxWidth, newHeight));
            Console.WriteLine($"Resized to {MaxWidth}x{newHeight}");
        }

        // Enhance contrast
        processed.Mutate(x => x.Contrast(1.4f));

        return processed;
    }
}
