using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Kwikread;

/// <summary>
/// Segments a full-page image into individual text lines for OCR processing.
/// </summary>
public class LineSegmenter
{
    private const int MinLineHeight = 20;
    private const int MaxLineHeight = 300;
    private const double WhiteSpaceThreshold = 0.85; // 85% white = line boundary
    private const int ProjectionSmoothWindow = 5;    // Moving average to reduce noise
    private const double SingleRegionMaxFraction = 0.6; // If one region is >60% of page, subdivide it

    /// <summary>
    /// Segments an image into individual text lines.
    /// Returns a list of cropped line images.
    /// </summary>
    public static List<Image<Rgb24>> SegmentLines(Image<Rgb24> image)
    {
        var lines = new List<Image<Rgb24>>();
        
        // Horizontal projection (white pixels per row)
        var rawProjection = CalculateHorizontalProjection(image);
        var projection = SmoothProjection(rawProjection);
        
        var lineRegions = DetectLineRegions(projection, image.Height);
        var totalHeight = image.Height;
        
        // If we got 0 regions (whole page "white") or one huge region, subdivide by projection peaks
        if (lineRegions.Count == 0)
        {
            Console.WriteLine("No regions from threshold. Subdividing full page by projection peaks...");
            lineRegions = SplitRegionByPeaks(projection, 0, totalHeight, totalHeight);
        }
        else if (lineRegions.Count == 1)
        {
            var (top, bottom) = lineRegions[0];
            if ((bottom - top) > totalHeight * SingleRegionMaxFraction)
            {
                Console.WriteLine("Single large region detected. Subdividing by projection peaks...");
                lineRegions = SplitRegionByPeaks(projection, top, bottom, totalHeight);
            }
        }
        
        Console.WriteLine($"Detected {lineRegions.Count} potential text lines");

        // Process regions, splitting oversized ones
        var finalRegions = new List<(int top, int bottom)>();
        foreach (var (top, bottom) in lineRegions)
        {
            var lineHeight = bottom - top;

            if (lineHeight < MinLineHeight)
            {
                Console.WriteLine($"  Skipping tiny region {top}-{bottom} (height {lineHeight}px)");
                continue;
            }

            if (lineHeight > MaxLineHeight)
            {
                // Try to split oversized regions by projection peaks
                Console.WriteLine($"  Splitting oversized region {top}-{bottom} (height {lineHeight}px)...");
                var subRegions = SplitRegionByPeaks(projection, top, bottom, totalHeight);
                foreach (var sub in subRegions)
                {
                    var subHeight = sub.bottom - sub.top;
                    if (subHeight >= MinLineHeight && subHeight <= MaxLineHeight)
                    {
                        finalRegions.Add(sub);
                        Console.WriteLine($"    Sub-region: {sub.top}-{sub.bottom} ({subHeight}px)");
                    }
                    else if (subHeight > MaxLineHeight)
                    {
                        // Still too large - use fixed chunking for this region
                        Console.WriteLine($"    Sub-region {sub.top}-{sub.bottom} still too large ({subHeight}px), chunking...");
                        var chunkHeight = MaxLineHeight / 2;
                        for (int y = sub.top; y < sub.bottom; y += chunkHeight - 10)
                        {
                            var endY = Math.Min(y + chunkHeight, sub.bottom);
                            if (endY - y >= MinLineHeight)
                            {
                                finalRegions.Add((y, endY));
                                Console.WriteLine($"    Chunk: {y}-{endY} ({endY - y}px)");
                            }
                        }
                    }
                }
            }
            else
            {
                finalRegions.Add((top, bottom));
            }
        }

        // Extract final line images (no padding - peaks already define boundaries)
        foreach (var (top, bottom) in finalRegions)
        {
            var cropHeight = bottom - top;

            var lineImage = image.Clone(ctx => ctx.Crop(new Rectangle(0, top, image.Width, cropHeight)));
            lines.Add(lineImage);

            Console.WriteLine($"  Extracted line {lines.Count}: y={top}-{bottom} ({cropHeight}px)");
        }

        if (lines.Count == 0)
        {
            Console.WriteLine("Projection method failed. Using adaptive fixed-height chunking...");
            return SegmentLinesFixedHeight(image);
        }

        return lines;
    }

    /// <summary>
    /// Smooth projection with a moving average to reduce single-row noise.
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
    /// Split a region into lines using a hybrid approach:
    /// 1. Estimate line count from region height
    /// 2. Find the best gap (whitespace peak) near each estimated boundary
    /// </summary>
    private static List<(int top, int bottom)> SplitRegionByPeaks(double[] projection, int regionTop, int regionBottom, int totalHeight)
    {
        int regionHeight = regionBottom - regionTop;

        // Estimate typical line height (handwriting is usually 50-80px at 2000px width)
        const int TypicalLineHeight = 70;
        int estimatedLines = Math.Max(1, (regionHeight + TypicalLineHeight / 2) / TypicalLineHeight);

        // If region is small enough for just a few lines, use simple division
        if (estimatedLines <= 2)
        {
            var simpleRegions = new List<(int top, int bottom)>();
            int lineHeight = regionHeight / estimatedLines;
            for (int i = 0; i < estimatedLines; i++)
            {
                int top = regionTop + i * lineHeight;
                int bottom = (i == estimatedLines - 1) ? regionBottom : regionTop + (i + 1) * lineHeight;
                if (bottom - top >= MinLineHeight)
                    simpleRegions.Add((top, bottom));
            }
            return simpleRegions;
        }

        // For larger regions, find optimal gaps near estimated boundaries
        int targetLineHeight = regionHeight / estimatedLines;
        var boundaries = new List<int> { regionTop };

        for (int i = 1; i < estimatedLines; i++)
        {
            int estimatedBoundary = regionTop + i * targetLineHeight;

            // Search for the best gap (highest projection) within a window around the estimated boundary
            int searchRadius = targetLineHeight / 3;
            int searchStart = Math.Max(regionTop, estimatedBoundary - searchRadius);
            int searchEnd = Math.Min(regionBottom, estimatedBoundary + searchRadius);

            int bestGap = estimatedBoundary;
            double bestValue = -1;

            for (int y = searchStart; y < searchEnd && y < projection.Length; y++)
            {
                if (projection[y] > bestValue)
                {
                    bestValue = projection[y];
                    bestGap = y;
                }
            }

            boundaries.Add(bestGap);
        }

        boundaries.Add(regionBottom);

        // Create regions from boundaries
        var regions = new List<(int top, int bottom)>();
        for (int i = 0; i < boundaries.Count - 1; i++)
        {
            int top = boundaries[i], bottom = boundaries[i + 1];
            if (bottom - top >= MinLineHeight)
                regions.Add((top, bottom));
        }
        return regions;
    }

    /// <summary>
    /// Fallback: Split image into chunks. Uses adaptive height based on image size
    /// so we get a reasonable number of lines (assume ~50px per line + spacing).
    /// </summary>
    private static List<Image<Rgb24>> SegmentLinesFixedHeight(Image<Rgb24> image)
    {
        var lines = new List<Image<Rgb24>>();
        const int TypicalLineWithSpacing = 70; // px per line + gap (handwriting ~50–100)
        int estimatedLines = Math.Max(1, image.Height / TypicalLineWithSpacing);
        int chunkHeight = (image.Height + estimatedLines - 1) / estimatedLines;
        chunkHeight = Math.Clamp(chunkHeight, 40, 250);
        int overlap = Math.Max(5, chunkHeight / 5);
        
        for (int y = 0; y < image.Height; y += Math.Max(1, chunkHeight - overlap))
        {
            var cropHeight = Math.Min(chunkHeight, image.Height - y);
            
            if (cropHeight < MinLineHeight)
                break;
            
            var chunk = image.Clone(ctx => ctx.Crop(new Rectangle(0, y, image.Width, cropHeight)));
            lines.Add(chunk);
            
            Console.WriteLine($"  Extracted chunk {lines.Count}: y={y}-{y + cropHeight} ({cropHeight}px)");
        }
        
        return lines;
    }

    /// <summary>
    /// Calculate horizontal projection (sum of white pixels per row).
    /// Higher values = more white space = likely line boundary.
    /// </summary>
    private static double[] CalculateHorizontalProjection(Image<Rgb24> image)
    {
        var projection = new double[image.Height];
        
        // Apply Otsu's method for adaptive thresholding
        var threshold = CalculateOtsuThreshold(image);
        Console.WriteLine($"Using brightness threshold: {threshold:F2}");
        
        for (int y = 0; y < image.Height; y++)
        {
            int whitePixels = 0;
            
            for (int x = 0; x < image.Width; x++)
            {
                var pixel = image[x, y];
                var brightness = (pixel.R + pixel.G + pixel.B) / 3.0 / 255.0;
                
                // Consider pixel as "white" if brightness > threshold
                if (brightness > threshold)
                {
                    whitePixels++;
                }
            }
            
            projection[y] = (double)whitePixels / image.Width;
        }
        
        return projection;
    }

    /// <summary>
    /// Calculate adaptive threshold using Otsu's method (simplified).
    /// Returns a brightness threshold between 0 and 1.
    /// </summary>
    private static double CalculateOtsuThreshold(Image<Rgb24> image)
    {
        // Sample every 10th pixel for speed
        var samples = new List<double>();
        for (int y = 0; y < image.Height; y += 10)
        {
            for (int x = 0; x < image.Width; x += 10)
            {
                var pixel = image[x, y];
                var brightness = (pixel.R + pixel.G + pixel.B) / 3.0 / 255.0;
                samples.Add(brightness);
            }
        }

        samples.Sort();
        var median = samples[samples.Count / 2];

        // Find actual separation between text and background
        // Use the gap between dark (text) and light (background) pixels
        var darkPixels = samples.Where(b => b < 0.5).ToList();
        var lightPixels = samples.Where(b => b >= 0.5).ToList();

        double threshold;
        if (darkPixels.Count > 0 && lightPixels.Count > 0)
        {
            // Threshold is midpoint between darkest background and lightest text
            var maxDark = darkPixels.Max();
            var minLight = lightPixels.Min();
            threshold = (maxDark + minLight) / 2;
        }
        else
        {
            // Fallback: use median with smaller bias
            threshold = median;
        }

        // Clamp to reasonable range - threshold should separate text from background
        // For most handwriting: text is dark (<0.5), background is light (>0.7)
        return Math.Clamp(threshold, 0.4, 0.85);
    }

    /// <summary>
    /// Detect line regions from horizontal projection.
    /// Returns list of (top, bottom) y-coordinates for each line.
    /// </summary>
    private static List<(int top, int bottom)> DetectLineRegions(double[] projection, int height)
    {
        var regions = new List<(int top, int bottom)>();
        bool inTextRegion = false;
        int regionStart = 0;
        
        for (int y = 0; y < height; y++)
        {
            bool isWhiteSpace = projection[y] > WhiteSpaceThreshold;
            
            if (!inTextRegion && !isWhiteSpace)
            {
                // Start of text region
                inTextRegion = true;
                regionStart = y;
            }
            else if (inTextRegion && isWhiteSpace)
            {
                // End of text region
                inTextRegion = false;
                regions.Add((regionStart, y));
            }
        }
        
        // Handle case where text extends to bottom of image
        if (inTextRegion)
        {
            regions.Add((regionStart, height));
        }
        
        // Merge nearby regions (likely same line)
        return MergeNearbyRegions(regions);
    }

    /// <summary>
    /// Merge regions that are very close together (likely same line of text).
    /// Uses a capped threshold to prevent runaway merging.
    /// </summary>
    private static List<(int top, int bottom)> MergeNearbyRegions(List<(int top, int bottom)> regions)
    {
        if (regions.Count == 0) return regions;

        var merged = new List<(int top, int bottom)>();
        var current = regions[0];

        // Cap merge threshold to prevent runaway merging of large regions
        const int MaxMergeThreshold = 25;

        for (int i = 1; i < regions.Count; i++)
        {
            var next = regions[i];
            var gap = next.top - current.bottom;
            var currentHeight = current.bottom - current.top;

            // If gap is small relative to line height, merge the regions
            // But cap the threshold to prevent creating giant merged regions
            var mergeThreshold = Math.Min(MaxMergeThreshold, Math.Max(8, currentHeight / 4));

            if (gap < mergeThreshold)
            {
                current = (current.top, next.bottom);
            }
            else
            {
                merged.Add(current);
                current = next;
            }
        }

        merged.Add(current);
        return merged;
    }

    /// <summary>
    /// Preprocess image to improve line detection.
    /// - Resize if too large
    /// - Increase contrast
    /// - Optional deskewing
    /// </summary>
    public static Image<Rgb24> PreprocessImage(Image<Rgb24> image)
    {
        var processed = image.Clone();
        
        // Resize if image is very large (speeds up processing)
        const int MaxWidth = 2000;
        if (processed.Width > MaxWidth)
        {
            var scale = (double)MaxWidth / processed.Width;
            var newHeight = (int)(processed.Height * scale);
            processed.Mutate(x => x.Resize(MaxWidth, newHeight));
            Console.WriteLine($"Resized image to {MaxWidth}x{newHeight}");
        }
        
        // Increase contrast to make text more distinct
        processed.Mutate(x => x.Contrast(1.5f));
        
        return processed;
    }
}
