"""
Bird Detection Model Abstraction
Preserves original YOLO11 5-class model and class names.
Accepts an external OpenCV frame (does NOT open its own camera).
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

try:
    import laptop.config as config
except ImportError:
    import config

class BirdDetector:
    def __init__(self, model_path=None, confidence=None):
        self.model_path = model_path or config.BIRD_MODEL_PATH
        self.confidence = confidence if confidence is not None else config.BIRD_CONFIDENCE
        
        # Target classes from the trained model:
        # {0: 'Crow', 1: 'Common Myna', 2: 'Rose Ringed Parakeet', 3: 'Peacock', 4: 'Pigeon'}
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Bird model weights not found at: {self.model_path}")
            
        print(f"[BirdDetector] Loading model from {self.model_path}...")
        self.model = YOLO(self.model_path, task="detect")
        self.names = self.model.names
        print(f"[BirdDetector] Loaded successfully with classes: {self.names}")

        # Target bird classes to track (supports both dedicated custom bird models and COCO yolo11n where class 14 is 'bird')
        self.target_birds = [
            "bird", "crow", "common myna", "rose ringed parakeet",
            "peacock", "pigeon", "sparrow", "parrot"
        ]
        self.bird_class_ids = [
            cls_id for cls_id, name in self.names.items()
            if name.lower() in self.target_birds or "bird" in name.lower()
        ]
        print(f"[BirdDetector] Active bird tracking class IDs: {self.bird_class_ids}")

    def detect(self, frame: np.ndarray) -> list:
        """
        Run inference on an OpenCV BGR frame.
        Returns a list of normalized detection dicts:
        [
            {
                "type": "bird",
                "label": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
        ]
        """
        if frame is None or self.model is None:
            return []

        # Predict with classes filter to ensure non-bird objects (person, chair, etc.) are excluded at YOLO layer
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=config.IOU_THRESHOLD,
            classes=self.bird_class_ids if self.bird_class_ids else None,
            imgsz=640,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw_label = self.names.get(cls_id, f"Bird_{cls_id}")
            label = "Bird" if raw_label.lower() == "bird" else raw_label

            detections.append({
                "type": "bird",
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

        return detections

