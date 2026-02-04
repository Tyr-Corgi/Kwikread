"""
Text postprocessing functions for Temporal OCR.

Contains text correction, filtering, consensus voting, and quality assessment.
"""

from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from .utils import FrameResult, DetectedLine, AggregatedLine


class FrameQualityAssessor:
    """Assesses frame quality for filtering."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_min: int = 30,
        brightness_max: int = 225,
        min_text_regions: int = 1
    ):
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.min_text_regions = min_text_regions

    def assess(
        self,
        image: np.ndarray,
        num_detections: int = 0,
        avg_confidence: float = 1.0
    ) -> Tuple[float, bool, str]:
        """
        Assess frame quality.

        Returns:
            (quality_score, is_valid, rejection_reason)
        """
        if image is None:
            return 0.0, False, "null_image"

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Blur metric (variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()

        if blur_score < self.blur_threshold:
            return blur_score, False, f"too_blurry ({blur_score:.1f} < {self.blur_threshold})"

        # Brightness check
        mean_brightness = gray.mean()
        if mean_brightness < self.brightness_min:
            return blur_score, False, f"too_dark ({mean_brightness:.1f})"
        if mean_brightness > self.brightness_max:
            return blur_score, False, f"too_bright ({mean_brightness:.1f})"

        # Text presence check
        if num_detections < self.min_text_regions:
            return blur_score, False, f"no_text_detected ({num_detections} regions)"

        # OCR confidence check - reject frames with garbage OCR
        if avg_confidence < 0.3:
            return blur_score, False, f"low_ocr_confidence ({avg_confidence:.2f})"

        # Quality score combines blur and brightness stability
        brightness_penalty = abs(mean_brightness - 128) / 128  # Prefer mid-gray
        quality_score = blur_score * (1 - 0.3 * brightness_penalty) * avg_confidence

        return quality_score, True, "ok"

    def select_best_frames(
        self,
        frame_results: List[FrameResult],
        max_frames: int = 15,
        min_frames: int = 5
    ) -> List[FrameResult]:
        """Select the best frames based on quality score."""
        # Filter valid frames
        valid = [f for f in frame_results if f.is_valid and len(f.lines) > 0]

        if len(valid) <= max_frames:
            return valid

        # Sort by quality and select top N
        valid = sorted(valid, key=lambda x: x.quality_score, reverse=True)
        return valid[:max_frames]


class TemporalLineMatcher:
    """Matches lines across frames for temporal aggregation."""

    def __init__(
        self,
        y_tolerance: float = 0.20,  # 20% of image height (more permissive)
        iou_threshold: float = 0.3,
        text_similarity_weight: float = 0.4,  # Weight text more for matching
        merge_similar_threshold: float = 0.7  # Merge lines with >70% text similarity
    ):
        self.y_tolerance = y_tolerance
        self.iou_threshold = iou_threshold
        self.text_similarity_weight = text_similarity_weight
        self.merge_similar_threshold = merge_similar_threshold

    def match_lines_across_frames(
        self,
        frame_results: List[FrameResult],
        image_height: int
    ) -> Dict[int, List[Tuple[int, DetectedLine]]]:
        """
        Match lines across frames.

        Returns:
            {canonical_line_index: [(frame_index, DetectedLine), ...]}
        """
        if not frame_results:
            return {}

        # Use first valid frame as reference
        reference_frame = None
        for fr in frame_results:
            if fr.lines:
                reference_frame = fr
                break

        if reference_frame is None:
            return {}

        # Initialize canonical lines from reference
        canonical_lines = {
            i: [(reference_frame.frame_index, line)]
            for i, line in enumerate(reference_frame.lines)
        }

        y_tol = image_height * self.y_tolerance

        # Match other frames to canonical lines
        for fr in frame_results:
            if fr.frame_index == reference_frame.frame_index:
                continue

            for line in fr.lines:
                best_match = self._find_best_match(
                    line, canonical_lines, y_tol, image_height
                )

                if best_match is not None:
                    canonical_lines[best_match].append((fr.frame_index, line))
                else:
                    # New line not seen in reference
                    new_idx = len(canonical_lines)
                    canonical_lines[new_idx] = [(fr.frame_index, line)]

        # Re-sort canonical lines by average y-position
        sorted_lines = {}
        for old_idx, matches in canonical_lines.items():
            avg_y = np.mean([m[1].bbox.center_y for m in matches])
            sorted_lines[old_idx] = (avg_y, matches)

        sorted_items = sorted(sorted_lines.items(), key=lambda x: x[1][0])

        return {
            new_idx: matches
            for new_idx, (old_idx, (_, matches)) in enumerate(sorted_items)
        }

    def _find_best_match(
        self,
        line: DetectedLine,
        canonical_lines: Dict[int, List[Tuple[int, DetectedLine]]],
        y_tol: float,
        image_height: int
    ) -> Optional[int]:
        """Find the best matching canonical line."""
        best_idx = None
        best_score = 0.0

        for idx, matches in canonical_lines.items():
            # Use median y-position of canonical line
            canonical_y = np.median([m[1].bbox.center_y for m in matches])

            # Y-distance score
            y_dist = abs(line.bbox.center_y - canonical_y)
            if y_dist > y_tol:
                continue

            y_score = 1 - (y_dist / y_tol)

            # Text similarity score (using best match from canonical)
            text_scores = []
            for _, canonical_line in matches:
                sim = self._text_similarity(line.text, canonical_line.text)
                text_scores.append(sim)
            text_score = max(text_scores) if text_scores else 0

            # Combined score
            score = y_score * (1 - self.text_similarity_weight) + text_score * self.text_similarity_weight

            if score > best_score and score > 0.5:
                best_score = score
                best_idx = idx

        return best_idx

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Levenshtein ratio."""
        if not text1 or not text2:
            return 0.0

        try:
            from rapidfuzz import fuzz
            return fuzz.ratio(text1.lower(), text2.lower()) / 100.0
        except ImportError:
            # Fallback to basic comparison
            return 1.0 if text1.lower() == text2.lower() else 0.0


class TemporalAggregator:
    """Aggregates OCR results across frames."""

    def __init__(
        self,
        consensus_threshold: float = 0.6,
        min_agreement: int = 2,
        use_consensus: bool = True
    ):
        self.consensus_threshold = consensus_threshold
        self.min_agreement = min_agreement
        self.use_consensus = use_consensus

    def aggregate_lines(
        self,
        matched_lines: Dict[int, List[Tuple[int, DetectedLine]]]
    ) -> List[AggregatedLine]:
        """
        Aggregate matched lines into final results.

        Returns:
            List of AggregatedLine objects
        """
        aggregated = []

        for line_idx, matches in sorted(matched_lines.items()):
            if not matches:
                continue

            # Extract candidates
            candidates = [
                (line.text, line.confidence, frame_idx)
                for frame_idx, line in matches
            ]

            # Best-of-N (highest confidence)
            best_candidate = max(candidates, key=lambda x: x[1])
            best_text = best_candidate[0]
            best_conf = best_candidate[1]

            # Consensus voting
            if self.use_consensus and len(candidates) >= self.min_agreement:
                consensus_text, consensus_conf = self._consensus_vote(candidates)
            else:
                consensus_text = best_text
                consensus_conf = best_conf

            # Choose final result
            if consensus_conf >= self.consensus_threshold:
                final_text = consensus_text
                final_conf = consensus_conf
            else:
                final_text = best_text
                final_conf = best_conf

            aggregated.append(AggregatedLine(
                line_index=line_idx,
                final_text=final_text,
                confidence=final_conf,
                consensus_text=consensus_text,
                best_single_text=best_text,
                best_single_confidence=best_conf,
                contributing_frames=[frame_idx for frame_idx, _ in matches],
                all_candidates=candidates
            ))

        return aggregated

    def _consensus_vote(
        self,
        candidates: List[Tuple[str, float, int]]
    ) -> Tuple[str, float]:
        """
        Character-level consensus voting with alignment.

        Returns:
            (consensus_text, confidence)
        """
        if not candidates:
            return "", 0.0

        if len(candidates) == 1:
            return candidates[0][0], candidates[0][1]

        texts = [c[0] for c in candidates]
        confs = [c[1] for c in candidates]

        # Find reference (longest or highest confidence)
        ref_idx = max(range(len(texts)), key=lambda i: (len(texts[i]), confs[i]))
        reference = texts[ref_idx]

        if not reference:
            return "", 0.0

        # Align all texts to reference and vote character-by-character
        try:
            from rapidfuzz import process
            from rapidfuzz.distance import Levenshtein
        except ImportError:
            # Fallback: just return highest confidence
            best_idx = max(range(len(candidates)), key=lambda i: confs[i])
            return texts[best_idx], confs[best_idx]

        # Simple approach: for each position, vote on character
        # Use weighted voting by confidence
        aligned_texts = [self._align_to_reference(reference, t) for t in texts]

        consensus_chars = []
        for pos in range(len(reference)):
            char_votes = {}
            total_weight = 0

            for i, aligned in enumerate(aligned_texts):
                if pos < len(aligned):
                    char = aligned[pos]
                    weight = confs[i]
                    char_votes[char] = char_votes.get(char, 0) + weight
                    total_weight += weight

            if char_votes:
                best_char = max(char_votes.items(), key=lambda x: x[1])
                consensus_chars.append(best_char[0])

        consensus_text = "".join(consensus_chars)

        # Calculate consensus confidence
        # Higher if more agreement, weighted by individual confidences
        avg_conf = np.mean(confs)

        # Calculate agreement score
        similarities = []
        for t in texts:
            sim = Levenshtein.normalized_similarity(consensus_text, t)
            similarities.append(sim)
        agreement = np.mean(similarities)

        consensus_conf = avg_conf * agreement

        return consensus_text.strip(), consensus_conf

    def _align_to_reference(self, reference: str, text: str) -> str:
        """Align text to reference using edit operations."""
        if not reference or not text:
            return text

        # Simple alignment: pad shorter string
        if len(text) < len(reference):
            text = text + " " * (len(reference) - len(text))
        elif len(text) > len(reference):
            text = text[:len(reference)]

        return text
