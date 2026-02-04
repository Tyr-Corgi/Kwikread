"""
Text detection functions for Temporal OCR.

Contains line detection, bounding box processing, and multi-column handling.
"""

from typing import List, Tuple

import cv2
import numpy as np

from .utils import BoundingBox


class LineDetector:
    """Detects and groups text into lines."""

    def __init__(
        self,
        ocr_engine,
        y_overlap_thresh: float = 0.5,
        min_line_height: int = 10,
        padding: int = 12
    ):
        self.ocr_engine = ocr_engine
        self.y_overlap_thresh = y_overlap_thresh
        self.min_line_height = min_line_height
        self.padding = padding

    def detect_lines(
        self,
        image: np.ndarray,
        preprocessed: np.ndarray = None
    ) -> List[Tuple[BoundingBox, List[Tuple[BoundingBox, str, float]]]]:
        """
        Detect text and group into lines.

        Returns:
            List of (line_bbox, [(word_bbox, text, conf), ...])
        """
        # Use original image for detection (OCR engines handle preprocessing)
        detections = self.ocr_engine.detect_and_recognize(image)

        if not detections:
            return []

        # Convert polygons to bounding boxes
        word_boxes = []
        for polygon, text, conf in detections:
            if not text.strip():
                continue

            # Convert polygon to axis-aligned bbox
            pts = np.array(polygon)
            x = int(pts[:, 0].min())
            y = int(pts[:, 1].min())
            w = int(pts[:, 0].max() - x)
            h = int(pts[:, 1].max() - y)

            if h < self.min_line_height:
                continue

            bbox = BoundingBox(x, y, w, h)
            word_boxes.append((bbox, text, conf))

        if not word_boxes:
            return []

        # Group into lines by y-overlap clustering
        lines = self._group_into_lines(word_boxes)

        return lines

    def _group_into_lines(
        self,
        word_boxes: List[Tuple[BoundingBox, str, float]]
    ) -> List[Tuple[BoundingBox, List[Tuple[BoundingBox, str, float]]]]:
        """
        Group word boxes into lines with COLUMN-AWARE detection.
        Handles multi-column layouts and prevents merging items across columns.
        Also handles diagonal text by treating isolated words as separate items.
        """
        if not word_boxes:
            return []

        from sklearn.cluster import DBSCAN

        # Estimate typical dimensions
        heights = [box.height for box, _, _ in word_boxes]
        widths = [box.width for box, _, _ in word_boxes]
        median_height = np.median(heights)
        median_width = np.median(widths)

        # STEP 1: First cluster by Y to find rough horizontal bands
        y_centers = np.array([[box.center_y] for box, _, _ in word_boxes])
        y_eps = median_height * 0.7
        y_clustering = DBSCAN(eps=max(y_eps, 15), min_samples=1).fit(y_centers)

        # STEP 2: Within each Y-cluster, split by large X gaps (column detection)
        # This prevents merging items from different columns
        # Use minimum of median_width * 1.5 OR 100px - whichever is smaller
        # This catches column gaps which are typically >100px
        MAX_WORD_GAP = min(median_width * 1.5, 100)  # Max gap between words in same item

        final_lines = []
        y_groups = {}
        for i, label in enumerate(y_clustering.labels_):
            if label not in y_groups:
                y_groups[label] = []
            y_groups[label].append(word_boxes[i])

        for label, words in y_groups.items():
            # Sort words left-to-right
            words = sorted(words, key=lambda x: x[0].x)

            # Split into separate items based on X gaps
            current_item = [words[0]]
            for i in range(1, len(words)):
                prev_box = words[i-1][0]
                curr_box = words[i][0]

                # Calculate gap between end of previous word and start of current
                gap = curr_box.x - (prev_box.x + prev_box.width)

                if gap > MAX_WORD_GAP:
                    # Large gap - this is a new column/item
                    if current_item:
                        final_lines.append(current_item)
                    current_item = [words[i]]
                else:
                    # Same item, add to current group
                    current_item.append(words[i])

            if current_item:
                final_lines.append(current_item)

        # STEP 3: Create line bounding boxes for each item
        lines = []
        for words in final_lines:
            min_x = min(box.x for box, _, _ in words)
            min_y = min(box.y for box, _, _ in words)
            max_x = max(box.x + box.width for box, _, _ in words)
            max_y = max(box.y + box.height for box, _, _ in words)

            line_bbox = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
            lines.append((line_bbox, words))

        # Sort lines by Y first (top-to-bottom), then X (left-to-right)
        # This gives natural reading order for multi-column layouts
        lines = sorted(lines, key=lambda x: (x[0].center_y // 50, x[0].x))

        return lines

    def crop_line(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        padding: int = None
    ) -> np.ndarray:
        """Crop a line region with padding."""
        if padding is None:
            padding = self.padding

        h, w = image.shape[:2]

        x1 = max(0, bbox.x - padding)
        y1 = max(0, bbox.y - padding)
        x2 = min(w, bbox.x + bbox.width + padding)
        y2 = min(h, bbox.y + bbox.height + padding)

        return image[y1:y2, x1:x2].copy()


def split_wide_detections(
    image: np.ndarray,
    detections: List
) -> List:
    """Split wide detections that may span multiple columns."""
    final_detections = []

    for item in detections:
        polygon = item[0]
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        x1, x2 = int(min(x_coords)), int(max(x_coords))
        y1, y2 = int(min(y_coords)), int(max(y_coords))
        width = x2 - x1
        height = y2 - y1

        if width > 400 and width > height * 3:
            # Extract crop and find text blobs
            pad = 3
            crop = image[max(0,y1-pad):min(image.shape[0],y2+pad),
                        max(0,x1-pad):min(image.shape[1],x2+pad)]

            if crop.size > 0:
                if len(crop.shape) == 3:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                else:
                    gray = crop.copy()

                _, binary = cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.dilate(binary, kernel, iterations=2)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    binary, connectivity=8
                )

                blobs = []
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > 100:
                        bx = stats[i, cv2.CC_STAT_LEFT]
                        bw = stats[i, cv2.CC_STAT_WIDTH]
                        blobs.append((bx, bx + bw))

                if len(blobs) >= 2:
                    blobs.sort(key=lambda b: b[0])

                    max_gap = 0
                    split_x_local = None
                    for i in range(len(blobs) - 1):
                        gap = blobs[i+1][0] - blobs[i][1]
                        if gap > max_gap:
                            max_gap = gap
                            split_x_local = (blobs[i][1] + blobs[i+1][0]) / 2

                    if max_gap > 80 and split_x_local is not None:
                        split_x = x1 + int(split_x_local)
                        poly1 = [[x1, y1], [split_x-5, y1], [split_x-5, y2], [x1, y2]]
                        poly2 = [[split_x+5, y1], [x2, y1], [x2, y2], [split_x+5, y2]]
                        final_detections.append((poly1, "", 0))
                        final_detections.append((poly2, "", 0))
                        continue

        final_detections.append(item)

    return final_detections
