"""
OCR recognition engines for Temporal OCR.

Contains OCR engine wrappers for PaddleOCR, EasyOCR, Tesseract, TrOCR, and Ensemble modes.
"""

import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .detection import split_wide_detections


class OCREngine:
    """Wrapper for OCR engines (PaddleOCR, EasyOCR, Tesseract, TrOCR, Ensemble)."""

    def __init__(
        self,
        engine_name: str = "paddle",
        lang: str = "en",
        model_path: str = None,
        ensemble_base_model: str = "microsoft/trocr-large-handwritten",
        ensemble_weights: Tuple[float, float] = (0.6, 0.4)
    ):
        self.engine_name = engine_name.lower()
        self.lang = lang
        self.engine = None
        self.model_path = model_path
        self.trocr_model = None
        self.trocr_processor = None
        self.detector = None  # For TrOCR, we need a separate detector
        self.ensemble_engine = None  # For ensemble mode
        self.ensemble_base_model = ensemble_base_model
        self.ensemble_weights = ensemble_weights
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the selected OCR engine."""
        if self.engine_name == "paddle":
            try:
                from paddleocr import PaddleOCR
                # use_angle_cls for rotated text, det for detection, rec for recognition
                self.engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=False,
                    show_log=False,
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    rec_batch_num=6
                )
                print(f"[OCR] Initialized PaddleOCR (lang={self.lang})")
            except ImportError:
                print("[OCR] PaddleOCR not available, falling back to EasyOCR")
                self.engine_name = "easyocr"
                self._initialize_engine()

        elif self.engine_name == "easyocr":
            try:
                import easyocr
                self.engine = easyocr.Reader([self.lang], gpu=False, verbose=False)
                print(f"[OCR] Initialized EasyOCR (lang={self.lang})")
            except ImportError:
                print("[OCR] EasyOCR not available, falling back to Tesseract")
                self.engine_name = "tesseract"
                self._initialize_engine()

        elif self.engine_name == "tesseract":
            try:
                import pytesseract
                self.engine = pytesseract
                print("[OCR] Initialized Tesseract (warning: weak for handwriting)")
            except ImportError:
                raise RuntimeError("No OCR engine available. Install paddleocr or easyocr.")

        elif self.engine_name == "trocr":
            self._initialize_trocr()

        elif self.engine_name == "claude":
            self._initialize_claude()

        elif self.engine_name == "ensemble":
            try:
                self.ensemble_engine = EnsembleOCREngine(
                    finetuned_path=self.model_path,
                    base_model=self.ensemble_base_model,
                    lang=self.lang,
                    weights=self.ensemble_weights
                )
                print(f"[OCR] Initialized Ensemble OCR with {len(self.ensemble_engine.models)} models")

            except Exception as e:
                print(f"[OCR] Ensemble initialization failed ({e}), falling back to TrOCR")
                self.engine_name = "trocr"
                self._initialize_engine()

    def _initialize_trocr(self):
        """Initialize TrOCR engine."""
        try:
            import torch
            from transformers import VisionEncoderDecoderModel, TrOCRProcessor

            # Use fine-tuned model if available, otherwise fall back to base model
            if self.model_path:
                model_path = self.model_path
            else:
                # Check for fine-tuned models in order of preference
                script_dir = Path(__file__).parent.parent
                finetuned_paths = [
                    script_dir / "finetune" / "model_v3" / "final",
                    script_dir / "finetune" / "model_v2" / "final",
                    script_dir / "finetune" / "model" / "final",
                ]
                model_path = "microsoft/trocr-large-handwritten"  # Default fallback
                for finetuned in finetuned_paths:
                    if finetuned.exists():
                        model_path = str(finetuned)
                        break

            print(f"[OCR] Loading TrOCR from: {model_path}")
            self.trocr_processor = TrOCRProcessor.from_pretrained(model_path)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_path)
            self.trocr_model.eval()

            # Use MPS on Mac if available, else CPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("[OCR] Using MPS (Metal) acceleration")
            else:
                self.device = torch.device("cpu")
            self.trocr_model = self.trocr_model.to(self.device)

            # Initialize EasyOCR as detector (TrOCR only does recognition)
            import easyocr
            self.detector = easyocr.Reader([self.lang], gpu=False, verbose=False)
            print(f"[OCR] Initialized TrOCR with EasyOCR detector")

        except ImportError as e:
            print(f"[OCR] TrOCR not available ({e}), falling back to EasyOCR")
            self.engine_name = "easyocr"
            self._initialize_engine()

    def _initialize_claude(self):
        """Initialize Claude Vision engine."""
        try:
            import anthropic

            self.claude_client = anthropic.Anthropic()

            # Initialize EasyOCR as detector (Claude does recognition)
            import easyocr
            self.detector = easyocr.Reader([self.lang], gpu=False, verbose=False)
            print(f"[OCR] Initialized Claude Vision with EasyOCR detector")

        except ImportError as e:
            print(f"[OCR] Claude SDK not available ({e}), falling back to EasyOCR")
            self.engine_name = "easyocr"
            self._initialize_engine()

    def detect_and_recognize(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """
        Detect text regions and recognize text.

        Returns:
            List of (polygon_points, text, confidence)
        """
        if image is None or image.size == 0:
            return []

        results = []

        if self.engine_name == "paddle":
            results = self._detect_paddle(image)

        elif self.engine_name == "easyocr":
            results = self._detect_easyocr(image)

        elif self.engine_name == "tesseract":
            results = self._detect_tesseract(image)

        elif self.engine_name == "trocr":
            results = self._detect_trocr(image)

        elif self.engine_name == "claude":
            results = self._detect_claude(image)

        elif self.engine_name == "ensemble":
            # Delegate to ensemble engine
            return self.ensemble_engine.detect_and_recognize(image)

        return results

    def _detect_paddle(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """PaddleOCR detection and recognition."""
        results = []
        try:
            ocr_result = self.engine.ocr(image, cls=True)
            if ocr_result and ocr_result[0]:
                for item in ocr_result[0]:
                    if item is None:
                        continue
                    polygon = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text, conf = item[1]
                    results.append((polygon, text, conf))
        except Exception as e:
            print(f"[OCR] PaddleOCR error: {e}")
        return results

    def _detect_easyocr(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """EasyOCR detection and recognition."""
        results = []
        try:
            # Word-level detection to handle multi-column layouts
            ocr_result = self.engine.readtext(
                image,
                paragraph=False,
                width_ths=0.1,  # Prevent horizontal word merging
                height_ths=0.5
            )
            for item in ocr_result:
                polygon = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = item[1]
                conf = item[2]
                results.append((polygon, text, conf))
        except Exception as e:
            print(f"[OCR] EasyOCR error: {e}")
        return results

    def _detect_tesseract(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """Tesseract detection and recognition."""
        results = []
        try:
            data = self.engine.image_to_data(image, output_type=self.engine.Output.DICT)
            for i, text in enumerate(data['text']):
                if text.strip():
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    conf = float(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0.5
                    polygon = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                    results.append((polygon, text, conf))
        except Exception as e:
            print(f"[OCR] Tesseract error: {e}")
        return results

    def _detect_trocr(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """TrOCR detection (using EasyOCR) and recognition."""
        results = []
        try:
            # Get detections - use moderate merging, we'll split wide ones ourselves
            detections = self.detector.readtext(
                image,
                paragraph=False,
                width_ths=0.3,
                height_ths=0.5
            )

            # Split wide detections using connected component analysis
            final_detections = split_wide_detections(image, detections)

            # Process each detection with TrOCR
            for item in final_detections:
                polygon = item[0]
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))

                # Pad slightly
                pad = 5
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(image.shape[1], x2+pad), min(image.shape[0], y2+pad)

                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    text, conf = self.recognize_line(crop)
                    if text:
                        results.append((polygon, text, conf))
        except Exception as e:
            print(f"[OCR] TrOCR detection error: {e}")
        return results

    def _detect_claude(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """Claude Vision detection and recognition."""
        results = []
        try:
            # Word-level detection to handle multi-column layouts
            detections = self.detector.readtext(
                image,
                paragraph=False,
                width_ths=0.1,
                height_ths=0.5
            )
            for item in detections:
                polygon = item[0]
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))

                pad = 5
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(image.shape[1], x2+pad), min(image.shape[0], y2+pad)

                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    text, conf = self.recognize_line(crop)
                    if text:
                        results.append((polygon, text, conf))
        except Exception as e:
            print(f"[OCR] Claude detection error: {e}")
        return results

    def recognize_line(self, line_image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text in a cropped line image.

        Returns:
            (text, confidence)
        """
        if line_image is None or line_image.size == 0:
            return "", 0.0

        if self.engine_name == "paddle":
            return self._recognize_paddle(line_image)

        elif self.engine_name == "easyocr":
            return self._recognize_easyocr(line_image)

        elif self.engine_name == "tesseract":
            return self._recognize_tesseract(line_image)

        elif self.engine_name == "trocr":
            return self._recognize_trocr(line_image)

        elif self.engine_name == "ensemble":
            # Delegate to ensemble engine
            return self.ensemble_engine.recognize_line(line_image)

        return "", 0.0

    def _recognize_paddle(self, line_image: np.ndarray) -> Tuple[str, float]:
        """PaddleOCR line recognition."""
        try:
            result = self.engine.ocr(line_image, det=False, cls=True)
            if result and result[0]:
                # Combine all recognized text parts
                texts = []
                confs = []
                for item in result[0]:
                    if item:
                        texts.append(item[0])
                        confs.append(item[1])
                if texts:
                    return " ".join(texts), sum(confs) / len(confs)
        except Exception as e:
            print(f"[OCR] PaddleOCR recognize_line error: {e}")
        return "", 0.0

    def _recognize_easyocr(self, line_image: np.ndarray) -> Tuple[str, float]:
        """EasyOCR line recognition."""
        try:
            result = self.engine.readtext(line_image)
            if result:
                texts = [item[1] for item in result]
                confs = [item[2] for item in result]
                return " ".join(texts), sum(confs) / len(confs) if confs else 0.0
        except Exception as e:
            print(f"[OCR] EasyOCR recognize_line error: {e}")
        return "", 0.0

    def _recognize_tesseract(self, line_image: np.ndarray) -> Tuple[str, float]:
        """Tesseract line recognition."""
        try:
            text = self.engine.image_to_string(line_image).strip()
            return text, 0.5  # Tesseract doesn't give confidence easily
        except Exception as e:
            print(f"[OCR] Tesseract recognize_line error: {e}")
        return "", 0.0

    def _recognize_trocr(self, line_image: np.ndarray) -> Tuple[str, float]:
        """TrOCR line recognition."""
        try:
            import torch
            from PIL import Image, ImageOps, ImageEnhance

            # === GENTLE PREPROCESSING FOR HANDWRITING (preserve strokes) ===

            # Step 1: Convert to PIL Image directly (keep color for now)
            if len(line_image.shape) == 3:
                pil_orig = Image.fromarray(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
            else:
                pil_orig = Image.fromarray(line_image).convert("RGB")

            # Step 2: Add padding
            pad = 16
            w, h = pil_orig.size
            padded = Image.new("RGB", (w + 2*pad, h + 2*pad), (255, 255, 255))
            padded.paste(pil_orig, (pad, pad))

            # Step 3: Resize to reasonable height
            w, h = padded.size
            target_height = 96
            scale = target_height / h
            new_w = max(int(w * scale), 64)
            resized = padded.resize((new_w, target_height), Image.LANCZOS)

            # Step 4: Convert to grayscale and enhance contrast
            gray_pil = resized.convert("L")

            # Auto-contrast (stretches histogram)
            gray_pil = ImageOps.autocontrast(gray_pil, cutoff=2)

            # Convert back to RGB
            pil_image = gray_pil.convert("RGB")

            # Boost contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.3)

            # Process and generate with optimized parameters
            pixel_values = self.trocr_processor(
                pil_image, return_tensors="pt"
            ).pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = self.trocr_model.generate(
                    pixel_values,
                    max_length=64,
                    num_beams=5,  # More beams for better decoding
                    early_stopping=True,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,  # Prevent repetition
                )

            text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            # POST-PROCESSING: Clean up TrOCR word output
            # Remove spurious periods/punctuation (common TrOCR artifact)
            text = re.sub(r'\s*[.,;:]\s*', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            # Line-level corrections happen after words are grouped

            # TrOCR doesn't provide confidence directly; estimate based on output
            conf = 0.9
            if not text or len(text) < 2:
                conf = 0.1
            elif any(x in text.lower() for x in ['%', 'na%', 'sc%', '###']):
                conf = 0.1  # Garbage output

            return text, conf

        except Exception as e:
            print(f"[OCR] TrOCR recognition error: {e}")
            import traceback
            traceback.print_exc()

        return "", 0.0


class EnsembleOCREngine:
    """
    Ensemble OCR engine that combines predictions from multiple TrOCR models
    using character-level voting (similar to temporal consensus).

    Runs:
    1. Fine-tuned model (finetune/model_v3/final)
    2. Base model (trocr-large-handwritten or trocr-base-handwritten)

    Uses weighted character-level voting to combine results.
    """

    def __init__(
        self,
        finetuned_path: str = None,
        base_model: str = "microsoft/trocr-large-handwritten",
        lang: str = "en",
        weights: Tuple[float, float] = (0.6, 0.4)  # (finetuned, base) weights
    ):
        self.lang = lang
        self.weights = weights
        self.models = []
        self.processors = []
        self.model_names = []
        self.device = None
        self.detector = None

        self._initialize_ensemble(finetuned_path, base_model)

    def _initialize_ensemble(self, finetuned_path: str, base_model: str):
        """Initialize all models in the ensemble."""
        import torch
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor

        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[Ensemble] Using MPS (Metal) acceleration")
        else:
            self.device = torch.device("cpu")
            print("[Ensemble] Using CPU")

        # Load fine-tuned model
        if finetuned_path is None:
            script_dir = Path(__file__).parent.parent
            finetuned_paths = [
                script_dir / "finetune" / "model_v3" / "final",
                script_dir / "finetune" / "model_v2" / "final",
                script_dir / "finetune" / "model" / "final",
            ]
            for fp in finetuned_paths:
                if fp.exists():
                    finetuned_path = str(fp)
                    break

        if finetuned_path and Path(finetuned_path).exists():
            print(f"[Ensemble] Loading fine-tuned model: {finetuned_path}")
            try:
                processor = TrOCRProcessor.from_pretrained(finetuned_path)
                model = VisionEncoderDecoderModel.from_pretrained(finetuned_path)
                model.eval()
                model = model.to(self.device)
                self.models.append(model)
                self.processors.append(processor)
                self.model_names.append("finetuned")
            except Exception as e:
                print(f"[Ensemble] Failed to load fine-tuned model: {e}")
        else:
            print(f"[Ensemble] Fine-tuned model not found, skipping")

        # Load base model
        print(f"[Ensemble] Loading base model: {base_model}")
        try:
            processor = TrOCRProcessor.from_pretrained(base_model)
            model = VisionEncoderDecoderModel.from_pretrained(base_model)
            model.eval()
            model = model.to(self.device)
            self.models.append(model)
            self.processors.append(processor)
            self.model_names.append("base")
        except Exception as e:
            print(f"[Ensemble] Failed to load base model: {e}")

        if not self.models:
            raise RuntimeError("No models loaded for ensemble")

        print(f"[Ensemble] Loaded {len(self.models)} models: {self.model_names}")

        # Initialize detector (EasyOCR for text detection)
        try:
            import easyocr
            self.detector = easyocr.Reader([self.lang], gpu=False, verbose=False)
            print("[Ensemble] Initialized EasyOCR detector")
        except ImportError:
            raise RuntimeError("EasyOCR required for ensemble detection")

    def detect_and_recognize(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """
        Detect text regions and recognize using ensemble voting.

        Returns:
            List of (polygon_points, text, confidence)
        """
        if image is None or image.size == 0:
            return []

        results = []

        try:
            # Get detections from EasyOCR
            detections = self.detector.readtext(
                image,
                paragraph=False,
                width_ths=0.3,
                height_ths=0.5
            )

            # Split wide detections (same logic as TrOCR engine)
            final_detections = split_wide_detections(image, detections)

            # Process each detection with ensemble
            for item in final_detections:
                polygon = item[0]
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))

                # Pad slightly
                pad = 5
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(image.shape[1], x2+pad), min(image.shape[0], y2+pad)

                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    text, conf = self.recognize_line(crop)
                    if text:
                        results.append((polygon, text, conf))
        except Exception as e:
            print(f"[Ensemble] Detection error: {e}")

        return results

    def recognize_line(self, line_image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text using ensemble voting.

        Runs each model and combines results using character-level voting.

        Returns:
            (text, confidence)
        """
        if line_image is None or line_image.size == 0:
            return "", 0.0

        import torch
        from PIL import Image, ImageOps, ImageEnhance

        # Preprocess image (same as single TrOCR)
        if len(line_image.shape) == 3:
            pil_orig = Image.fromarray(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
        else:
            pil_orig = Image.fromarray(line_image).convert("RGB")

        # Add padding
        pad = 16
        w, h = pil_orig.size
        padded = Image.new("RGB", (w + 2*pad, h + 2*pad), (255, 255, 255))
        padded.paste(pil_orig, (pad, pad))

        # Resize
        w, h = padded.size
        target_height = 96
        scale = target_height / h
        new_w = max(int(w * scale), 64)
        resized = padded.resize((new_w, target_height), Image.LANCZOS)

        # Convert to grayscale and enhance
        gray_pil = resized.convert("L")
        gray_pil = ImageOps.autocontrast(gray_pil, cutoff=2)
        pil_image = gray_pil.convert("RGB")

        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)

        # Run inference on each model
        predictions = []
        for i, (model, processor) in enumerate(zip(self.models, self.processors)):
            try:
                pixel_values = processor(
                    pil_image, return_tensors="pt"
                ).pixel_values.to(self.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=64,
                        num_beams=5,
                        early_stopping=True,
                        length_penalty=1.0,
                        no_repeat_ngram_size=3,
                    )

                text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()

                # Clean up
                text = re.sub(r'\s*[.,;:]\s*', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()

                # Estimate confidence
                conf = 0.9
                if not text or len(text) < 2:
                    conf = 0.1
                elif any(x in text.lower() for x in ['%', 'na%', 'sc%', '###']):
                    conf = 0.1

                # Apply model weight
                weight = self.weights[i] if i < len(self.weights) else 0.5
                predictions.append((text, conf, weight, self.model_names[i]))

            except Exception as e:
                print(f"[Ensemble] Model {self.model_names[i]} error: {e}")

        if not predictions:
            return "", 0.0

        if len(predictions) == 1:
            return predictions[0][0], predictions[0][1]

        # Ensemble voting
        ensemble_text, ensemble_conf = self._ensemble_vote(predictions)
        return ensemble_text, ensemble_conf

    def _ensemble_vote(
        self,
        predictions: List[Tuple[str, float, float, str]]
    ) -> Tuple[str, float]:
        """
        Character-level ensemble voting (similar to temporal consensus).

        Args:
            predictions: List of (text, confidence, weight, model_name)

        Returns:
            (consensus_text, confidence)
        """
        if not predictions:
            return "", 0.0

        if len(predictions) == 1:
            return predictions[0][0], predictions[0][1]

        texts = [p[0] for p in predictions]
        confs = [p[1] for p in predictions]
        weights = [p[2] for p in predictions]

        # Combined weight = confidence * model_weight
        combined_weights = [c * w for c, w in zip(confs, weights)]

        # Find reference (longest text with highest combined weight)
        ref_idx = max(range(len(texts)),
                      key=lambda i: (len(texts[i]), combined_weights[i]))
        reference = texts[ref_idx]

        if not reference:
            return "", 0.0

        # Align all texts to reference
        aligned_texts = [self._align_to_reference(reference, t) for t in texts]

        # Vote character by character
        consensus_chars = []
        for pos in range(len(reference)):
            char_votes = {}
            total_weight = 0

            for i, aligned in enumerate(aligned_texts):
                if pos < len(aligned):
                    char = aligned[pos]
                    weight = combined_weights[i]
                    char_votes[char] = char_votes.get(char, 0) + weight
                    total_weight += weight

            if char_votes:
                best_char = max(char_votes.items(), key=lambda x: x[1])
                consensus_chars.append(best_char[0])

        consensus_text = "".join(consensus_chars).strip()

        # Calculate consensus confidence
        try:
            from rapidfuzz.distance import Levenshtein

            similarities = []
            for t, w in zip(texts, combined_weights):
                sim = Levenshtein.normalized_similarity(consensus_text, t)
                similarities.append(sim * w)

            agreement = sum(similarities) / sum(combined_weights) if combined_weights else 0
            avg_conf = np.mean(confs)
            consensus_conf = avg_conf * agreement

        except ImportError:
            # Fallback
            consensus_conf = np.mean(confs)

        return consensus_text, consensus_conf

    def _align_to_reference(self, reference: str, text: str) -> str:
        """Align text to reference by padding/truncating."""
        if not reference or not text:
            return text

        if len(text) < len(reference):
            text = text + " " * (len(reference) - len(text))
        elif len(text) > len(reference):
            text = text[:len(reference)]

        return text
