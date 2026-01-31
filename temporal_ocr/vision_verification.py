"""
Vision LLM Verification Module for Temporal OCR

Uses local vision-capable LLMs (Qwen2.5-VL, LLaVA) to verify and correct
TrOCR recognition errors by leveraging semantic understanding.

The key insight: Vision LLMs understand that "Oat milk" is a valid grocery
item while "out milk" is not - something TrOCR cannot determine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum
import numpy as np
import cv2
import base64


class VerificationMode(Enum):
    """Vision verification operating modes."""
    DISABLED = "disabled"
    VERIFY_LOW_CONFIDENCE = "verify_low"
    VERIFY_ALL = "verify_all"
    PRIMARY = "primary"  # Vision LLM is primary, TrOCR fallback


@dataclass
class VerificationResult:
    """Result from vision LLM verification."""
    original_text: str
    original_confidence: float
    verified_text: str
    verified_confidence: float
    vision_text: Optional[str]
    vision_confidence: Optional[float]
    agreement_score: float
    source: str  # "trocr", "vision", "consensus"


class VisionVerifierBase(ABC):
    """Abstract base class for vision LLM verifiers."""

    @abstractmethod
    def recognize_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Recognize text in a cropped image using vision LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the verifier is available and ready."""
        pass


class OllamaVisionVerifier(VisionVerifierBase):
    """Local vision LLM verifier using Ollama.

    Tested models (in order of OCR accuracy):
    - llava:7b - Best overall (5/6 test cases correct)
    - moondream - Good with describe prompt (4/6 correct)
    - qwen2.5-vl:7b - Requires more VRAM
    """

    def __init__(
        self,
        model: str = "llava:7b",  # LLaVA tested best for OCR
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0  # Increased for slower machines
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._available is not None:
            return self._available

        try:
            import requests
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our model or a variant is available
                self._available = any(
                    self.model.split(":")[0] in name
                    for name in model_names
                )
                if not self._available:
                    print(f"[Vision] Model {self.model} not found. Available: {model_names}")
            else:
                self._available = False
        except Exception as e:
            print(f"[Vision] Ollama not available: {e}")
            self._available = False

        return self._available

    def recognize_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Recognize text using Ollama vision model.

        Uses tested prompts that work well with LLaVA for grocery list OCR.
        Extracts quoted text from responses for clean output.
        """
        import requests
        import re

        if not self.is_available():
            return "", 0.0

        # Convert image to base64
        _, buffer = cv2.imencode('.png', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Tested prompt - works well with LLaVA for handwritten text
        prompt = "Read the handwritten text in this image and tell me exactly what it says."

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 100
                    }
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                raw_response = response.json().get("response", "").strip()
                text = self._extract_text(raw_response)
                confidence = self._estimate_confidence(text, raw_response)
                return text, confidence
            else:
                print(f"[Vision] Ollama error: {response.status_code}")
                return "", 0.0

        except Exception as e:
            print(f"[Vision] Recognition error: {e}")
            return "", 0.0

    def _extract_text(self, raw_response: str) -> str:
        """Extract the actual text from vision model response.

        Vision models often wrap the text in quotes or include explanations.
        This extracts just the recognized text.
        """
        import re

        if not raw_response:
            return ""

        # Strategy 1: Look for quoted text (most common pattern)
        # e.g., 'reads: "Oat Milk"' -> 'Oat Milk'
        quotes = re.findall(r'"([^"]+)"', raw_response)
        if quotes:
            text = quotes[0]
        else:
            # Strategy 2: Look for text after common patterns
            patterns = [
                r'(?:reads|says|written|text is)[:\s]+["\']?([A-Za-z][A-Za-z\s&]+)',
                r'(?:reads|says)[:\s]+(.+?)(?:\.|$)',
            ]
            text = ""
            for pattern in patterns:
                match = re.search(pattern, raw_response, re.I)
                if match:
                    text = match.group(1).strip()
                    break

            # Strategy 3: If still empty, use first line cleaned up
            if not text:
                text = raw_response.split('\n')[0].strip()

        # Clean up extracted text
        text = text.strip('"\'`.,')
        # Remove trailing punctuation that wasn't in original
        text = re.sub(r'[.!?]+$', '', text)

        return text

    def _estimate_confidence(self, text: str, raw_response: str = "") -> float:
        """Estimate confidence from output characteristics.

        Higher confidence when:
        - Text is clean alphabetic characters
        - Response contains definitive language ("reads", "says")
        - Text is reasonable length for grocery item

        Lower confidence when:
        - Response contains uncertain language
        - Text is very short or very long
        - Text contains unusual characters
        """
        if not text:
            return 0.0
        if len(text) < 2:
            return 0.3

        # Check raw response for uncertainty
        uncertain_phrases = ['cannot', 'unclear', 'sorry', 'unable', 'difficult',
                           "can't", "don't see", 'not sure', 'appears to be']
        if raw_response and any(x in raw_response.lower() for x in uncertain_phrases):
            return 0.3

        # Check for confident response patterns
        confident_phrases = ['reads:', 'says:', 'written:', 'the text is']
        has_confident_pattern = raw_response and any(x in raw_response.lower() for x in confident_phrases)

        # Clean text analysis
        clean_text = text.replace(' ', '').replace('-', '').replace('&', '')
        is_clean_alpha = clean_text.isalpha()

        # Length check (grocery items typically 3-40 chars)
        reasonable_length = 3 <= len(text) <= 40

        # Calculate confidence
        if is_clean_alpha and reasonable_length and has_confident_pattern:
            return 0.95
        elif is_clean_alpha and reasonable_length:
            return 0.85
        elif is_clean_alpha:
            return 0.7
        elif reasonable_length:
            return 0.6
        return 0.5


class MockVisionVerifier(VisionVerifierBase):
    """Mock verifier for testing without Ollama."""

    def __init__(self):
        # Known corrections for testing
        self.corrections = {
            "out milk": "Oat Milk",
            "problem bars": "Protein Bars",
            "apple since": "Apple Sauce",
            "trail unit": "Trail Mix",
            "french tries": "French Fries",
            "philt lbury cookie doug": "Pillsbury Cookie Dough",
            "tortila": "Tortilla",
            "cheese sticks": "Cheese Sticks",  # Handle truncation
        }

    def is_available(self) -> bool:
        return True

    def recognize_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Mock recognition - would need TrOCR text to correct."""
        # This is just for testing the pipeline
        return "", 0.0

    def correct_text(self, trocr_text: str) -> Tuple[str, float]:
        """Correct known TrOCR errors."""
        lower = trocr_text.lower().strip()
        if lower in self.corrections:
            return self.corrections[lower], 0.95
        return trocr_text, 0.5


class VisionVerificationEngine:
    """
    Orchestrates vision LLM verification for OCR results.
    """

    def __init__(
        self,
        verifier: VisionVerifierBase,
        mode: VerificationMode = VerificationMode.VERIFY_ALL,
        confidence_threshold: float = 0.7
    ):
        self.verifier = verifier
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self._stats = {"verified": 0, "corrected": 0, "passed": 0}

    def verify_recognition(
        self,
        image: np.ndarray,
        trocr_text: str,
        trocr_confidence: float
    ) -> VerificationResult:
        """
        Verify/correct a TrOCR recognition result.
        """
        # Mode: Disabled
        if self.mode == VerificationMode.DISABLED:
            return self._pass_through(trocr_text, trocr_confidence)

        # Mode: Verify only low confidence
        if self.mode == VerificationMode.VERIFY_LOW_CONFIDENCE:
            if trocr_confidence >= self.confidence_threshold:
                self._stats["passed"] += 1
                return self._pass_through(trocr_text, trocr_confidence)

        # Get vision LLM recognition
        self._stats["verified"] += 1
        vision_text, vision_confidence = self.verifier.recognize_text(image)

        # If vision failed, fall back to TrOCR
        if not vision_text or vision_confidence < 0.3:
            return self._pass_through(trocr_text, trocr_confidence)

        # Calculate agreement
        agreement = self._calculate_agreement(trocr_text, vision_text)

        # Reconcile results
        final_text, final_confidence, source = self._reconcile(
            trocr_text, trocr_confidence,
            vision_text, vision_confidence,
            agreement
        )

        if source == "vision" and final_text != trocr_text:
            self._stats["corrected"] += 1
            print(f"[Vision] Corrected: '{trocr_text}' -> '{final_text}'")

        return VerificationResult(
            original_text=trocr_text,
            original_confidence=trocr_confidence,
            verified_text=final_text,
            verified_confidence=final_confidence,
            vision_text=vision_text,
            vision_confidence=vision_confidence,
            agreement_score=agreement,
            source=source
        )

    def _pass_through(self, text: str, confidence: float) -> VerificationResult:
        """Pass through TrOCR result without verification."""
        return VerificationResult(
            original_text=text,
            original_confidence=confidence,
            verified_text=text,
            verified_confidence=confidence,
            vision_text=None,
            vision_confidence=None,
            agreement_score=1.0,
            source="trocr"
        )

    def _calculate_agreement(self, text1: str, text2: str) -> float:
        """Calculate agreement score between two texts."""
        if not text1 or not text2:
            return 0.0

        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        if t1 == t2:
            return 1.0

        try:
            from rapidfuzz import fuzz
            return fuzz.ratio(t1, t2) / 100.0
        except ImportError:
            # Simple fallback
            common = set(t1.split()) & set(t2.split())
            total = set(t1.split()) | set(t2.split())
            return len(common) / len(total) if total else 0.0

    def _reconcile(
        self,
        trocr_text: str,
        trocr_conf: float,
        vision_text: str,
        vision_conf: float,
        agreement: float
    ) -> Tuple[str, float, str]:
        """
        Reconcile TrOCR and vision LLM results.

        Strategy:
        - Primary mode: Always prefer vision LLM
        - High vision confidence (>0.9): Trust vision (it has semantic understanding)
        - High agreement (>0.95): Both agree, use TrOCR
        - Low agreement (<0.5): Trust vision LLM
        - Medium: Use higher confidence

        Key insight: Vision LLMs understand "oat milk" is valid while "out milk"
        is not. When vision is confident, trust its semantic understanding.
        """
        # Primary mode: Vision LLM takes precedence
        if self.mode == VerificationMode.PRIMARY:
            if vision_text and vision_conf > 0.3:
                return vision_text, vision_conf, "vision"
            return trocr_text, trocr_conf, "trocr"

        # High vision confidence: Trust vision's semantic understanding
        # This catches cases like "out milk" vs "oat milk" (89% similar but vision is right)
        if vision_conf >= 0.9 and agreement < 0.95:
            return vision_text, vision_conf, "vision"

        # Very high agreement (>0.95): Both genuinely agree
        if agreement >= 0.95:
            return trocr_text, min(1.0, trocr_conf * 1.1), "consensus"

        # Low agreement: Trust vision LLM's semantic understanding
        if agreement < 0.5 and vision_conf > 0.5:
            return vision_text, vision_conf, "vision"

        # Medium agreement: Use higher confidence
        if vision_conf > trocr_conf:
            return vision_text, vision_conf, "vision"

        return trocr_text, trocr_conf, "trocr"

    def get_stats(self) -> dict:
        """Get verification statistics."""
        return self._stats.copy()


def create_verifier(
    verifier_type: str = "ollama",
    model: str = None,
    mode: str = "verify_all"
) -> Optional[VisionVerificationEngine]:
    """
    Factory function to create a vision verification engine.

    Args:
        verifier_type: "ollama", "mock", or None
        model: Model name (e.g., "qwen2.5-vl:7b", "llava:13b")
        mode: "disabled", "verify_low", "verify_all", "primary"

    Returns:
        VisionVerificationEngine or None if disabled
    """
    if verifier_type is None or verifier_type == "none":
        return None

    mode_map = {
        "disabled": VerificationMode.DISABLED,
        "verify_low": VerificationMode.VERIFY_LOW_CONFIDENCE,
        "verify_all": VerificationMode.VERIFY_ALL,
        "primary": VerificationMode.PRIMARY
    }

    verification_mode = mode_map.get(mode, VerificationMode.VERIFY_ALL)

    if verifier_type == "ollama":
        model = model or "llava:7b"  # LLaVA tested best for handwritten OCR
        verifier = OllamaVisionVerifier(model=model)
        if not verifier.is_available():
            print(f"[Vision] Warning: Ollama not available, verification disabled")
            print(f"[Vision] To enable: ollama pull {model}")
            return None
        print(f"[Vision] Initialized Ollama verifier with {model}")
    elif verifier_type == "mock":
        verifier = MockVisionVerifier()
        print("[Vision] Using mock verifier (for testing only)")
    else:
        print(f"[Vision] Unknown verifier type: {verifier_type}")
        return None

    return VisionVerificationEngine(verifier=verifier, mode=verification_mode)
