"""
Fast OCR Server - Keeps TrOCR model loaded for instant inference.

Usage:
  # Start server (in background)
  python ocr_server.py serve &

  # Process image (uses running server)
  python ocr_server.py process image.jpg

  # Process with selective vision verification (1-2 lowest confidence lines)
  python ocr_server.py process image.jpg --verify

  # Or via HTTP
  curl -X POST http://localhost:8765/ocr -F "image=@image.jpg"
"""

import sys
import json
import time
import socket
import struct
import base64
import threading
import requests
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

# Server config
HOST = 'localhost'
PORT = 8765
SOCKET_PATH = '/tmp/ocr_server.sock'
OLLAMA_URL = 'http://localhost:11434/api/generate'


class OCRServer:
    """Persistent OCR server with pre-loaded models."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.detector = None  # EasyOCR for line detection
        self._loaded = False

    def load_models(self):
        """Load all models once at startup."""
        if self._loaded:
            return

        print("[Server] Loading models...")
        t0 = time.time()

        # TrOCR for recognition
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        self.processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-large-handwritten'
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-large-handwritten'
        )

        # Use MPS (Metal) on Mac, CUDA on Linux/Windows, CPU fallback
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model.to(self.device)
        self.model.eval()

        # EasyOCR for detection (text boxes only)
        import easyocr
        self.detector = easyocr.Reader(['en'], gpu=self.device != 'cpu')

        # Strikethrough detector
        try:
            from strikethrough_filter import StrikethroughDetector, filter_strikethrough_text
            self.strike_detector = StrikethroughDetector()
            self.filter_strike = filter_strikethrough_text
        except:
            self.strike_detector = None
            self.filter_strike = None

        t1 = time.time()
        print(f"[Server] Models loaded in {t1-t0:.1f}s (device: {self.device})")
        self._loaded = True

    def detect_lines(self, image: np.ndarray) -> List[Tuple[list, str, float]]:
        """Detect text lines using EasyOCR."""
        results = self.detector.readtext(image)
        return results

    def recognize_batch(self, images: List[Image.Image]) -> List[str]:
        """Batch recognize text from PIL images."""
        import torch

        if not images:
            return []

        with torch.no_grad():
            pixel_values = self.processor(
                images=images,
                return_tensors='pt',
                padding=True
            ).pixel_values.to(self.device)

            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=50
            )

            texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

        return texts

    def verify_with_vision(self, crop_img: np.ndarray, trocr_text: str) -> str:
        """Verify/correct text using LLaVA vision model."""
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', crop_img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        prompt = f"""Look at this handwritten text image. The OCR read it as: "{trocr_text}"
Is this correct? If not, what does it actually say?
Reply with ONLY the corrected text, nothing else. If correct, reply with the same text."""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    'model': 'llava:7b',
                    'prompt': prompt,
                    'images': [img_b64],
                    'stream': False,
                    'options': {'temperature': 0.1}
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                # Clean up common LLaVA response patterns
                if result.lower().startswith('the text'):
                    result = result.split(':')[-1].strip().strip('"\'')
                return result if result else trocr_text
        except Exception as e:
            print(f"[Vision] Error: {e}")

        return trocr_text

    def process_image(self, image_path: str, verify: bool = False, max_verify: int = 2) -> dict:
        """Full OCR pipeline on an image.

        Args:
            image_path: Path to image file
            verify: Enable selective LLaVA verification on low-confidence lines
            max_verify: Maximum number of lines to verify (default: 2)
        """
        t0 = time.time()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not load: {image_path}"}

        # Detect text regions
        t1 = time.time()
        detections = self.detect_lines(image)
        t2 = time.time()

        if not detections:
            return {
                "image_path": image_path,
                "lines": [],
                "full_text": "",
                "timing": {"total": time.time() - t0}
            }

        # Group into lines and crop
        from sklearn.cluster import DBSCAN

        # Get bounding boxes with confidence
        boxes = []
        for det in detections:
            pts = np.array(det[0])
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            cy = y + h // 2
            boxes.append((x, y, w, h, cy, det[1], det[2]))  # det[2] is confidence

        # Cluster by y-center
        if len(boxes) > 1:
            y_centers = np.array([[b[4]] for b in boxes])
            clustering = DBSCAN(eps=30, min_samples=1).fit(y_centers)
            labels = clustering.labels_
        else:
            labels = [0]

        # Group boxes by line, track confidence per line
        lines = {}
        for i, (x, y, w, h, cy, text, conf) in enumerate(boxes):
            label = labels[i]
            if label not in lines:
                lines[label] = []
            lines[label].append((x, y, w, h, text, conf))

        # Sort lines top to bottom, words left to right
        sorted_lines = []
        line_confidences = []
        for label in sorted(lines.keys(), key=lambda l: np.mean([b[1] for b in lines[l]])):
            line_boxes = sorted(lines[label], key=lambda b: b[0])

            # Merge boxes into line bbox
            xs = [b[0] for b in line_boxes]
            ys = [b[1] for b in line_boxes]
            x2s = [b[0] + b[2] for b in line_boxes]
            y2s = [b[1] + b[3] for b in line_boxes]

            line_bbox = (min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys))
            sorted_lines.append((line_bbox, line_boxes))

            # Average confidence for the line
            avg_conf = np.mean([b[5] for b in line_boxes])
            line_confidences.append(avg_conf)

        # Crop lines with padding
        padding = 12
        h_img, w_img = image.shape[:2]
        crops = []
        crop_bboxes = []

        for line_bbox, _ in sorted_lines:
            x, y, w, h = line_bbox
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)

            crop = image[y1:y2, x1:x2]
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crops.append(pil_crop)
            crop_bboxes.append((x1, y1, x2-x1, y2-y1, crop))

        t3 = time.time()

        # Batch recognize
        texts = self.recognize_batch(crops)
        t4 = time.time()

        # Determine which lines to verify (lowest confidence)
        lines_to_verify = set()
        if verify and max_verify > 0:
            # Get indices of lowest confidence lines
            sorted_by_conf = sorted(range(len(line_confidences)),
                                   key=lambda i: line_confidences[i])
            lines_to_verify = set(sorted_by_conf[:max_verify])
            print(f"[Verify] Will verify {len(lines_to_verify)} lowest confidence lines")

        # Apply strikethrough filtering and optional vision verification
        results = []
        t_verify = 0
        verified_count = 0
        for i, (bbox_info, text) in enumerate(zip(crop_bboxes, texts)):
            x, y, w, h, crop_img = bbox_info
            confidence = line_confidences[i]

            # Filter strikethrough
            if self.strike_detector and self.filter_strike:
                filtered, had_strike, ratio = self.filter_strike(
                    crop_img, text, self.strike_detector
                )
                if had_strike:
                    text = filtered

            # Selective vision verification
            verified = False
            if i in lines_to_verify:
                t_v0 = time.time()
                original = text
                text = self.verify_with_vision(crop_img, text)
                t_v1 = time.time()
                t_verify += (t_v1 - t_v0)
                verified_count += 1
                if text != original:
                    print(f"[Verify] Line {i}: '{original}' -> '{text}'")
                verified = True

            results.append({
                "index": i,
                "bbox": [x, y, w, h],
                "text": text.strip(),
                "confidence": round(confidence, 3),
                "verified": verified
            })

        t5 = time.time()

        timing = {
            "detection": round(t2 - t1, 2),
            "grouping": round(t3 - t2, 2),
            "recognition": round(t4 - t3, 2),
            "filtering": round(t5 - t4 - t_verify, 2),
            "total": round(t5 - t0, 2)
        }
        if verify:
            timing["verification"] = round(t_verify, 2)
            timing["verified_lines"] = verified_count

        return {
            "image_path": image_path,
            "lines": results,
            "full_text": "\n".join(r["text"] for r in results),
            "timing": timing
        }

    def serve_socket(self):
        """Run as Unix socket server for fastest IPC."""
        import os

        # Remove old socket
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        self.load_models()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(5)

        print(f"[Server] Listening on {SOCKET_PATH}")
        print("[Server] Ready for requests!")

        while True:
            conn, _ = server.accept()
            try:
                # Read message length
                raw_len = conn.recv(4)
                if not raw_len:
                    continue
                msg_len = struct.unpack('>I', raw_len)[0]

                # Read message
                data = b''
                while len(data) < msg_len:
                    chunk = conn.recv(min(4096, msg_len - len(data)))
                    if not chunk:
                        break
                    data += chunk

                request = json.loads(data.decode())

                # Process
                if request.get('action') == 'ocr':
                    result = self.process_image(
                        request['image_path'],
                        verify=request.get('verify', False),
                        max_verify=request.get('max_verify', 2)
                    )
                elif request.get('action') == 'ping':
                    result = {"status": "ok"}
                else:
                    result = {"error": "Unknown action"}

                # Send response
                response = json.dumps(result).encode()
                conn.sendall(struct.pack('>I', len(response)) + response)

            except Exception as e:
                error = json.dumps({"error": str(e)}).encode()
                conn.sendall(struct.pack('>I', len(error)) + error)
            finally:
                conn.close()


def send_request(action: str, **kwargs) -> dict:
    """Send request to running server."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)

    request = {"action": action, **kwargs}
    data = json.dumps(request).encode()
    sock.sendall(struct.pack('>I', len(data)) + data)

    # Read response
    raw_len = sock.recv(4)
    msg_len = struct.unpack('>I', raw_len)[0]

    response = b''
    while len(response) < msg_len:
        chunk = sock.recv(min(4096, msg_len - len(response)))
        response += chunk

    sock.close()
    return json.loads(response.decode())


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'serve':
        server = OCRServer()
        server.serve_socket()

    elif cmd == 'process':
        if len(sys.argv) < 3:
            print("Usage: python ocr_server.py process <image_path> [--verify] [--max-verify N]")
            return

        image_path = sys.argv[2]

        # Parse optional flags
        verify = '--verify' in sys.argv
        max_verify = 2
        if '--max-verify' in sys.argv:
            idx = sys.argv.index('--max-verify')
            if idx + 1 < len(sys.argv):
                max_verify = int(sys.argv[idx + 1])

        t0 = time.time()
        result = send_request(
            'ocr',
            image_path=str(Path(image_path).absolute()),
            verify=verify,
            max_verify=max_verify
        )
        t1 = time.time()

        print(json.dumps(result, indent=2))
        print(f"\n[Client] Total time: {t1-t0:.2f}s")
        if verify:
            print(f"[Client] Verified {result.get('timing', {}).get('verified_lines', 0)} lines with LLaVA")

    elif cmd == 'ping':
        result = send_request('ping')
        print(result)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()
