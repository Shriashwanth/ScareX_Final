"""
Tomato Maturity Detection Model Abstraction
Preserves original YOLO11 3-class maturity model and class mapping.
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

class TomatoDetector:
    def __init__(self, model_path=None, confidence=None):
        self.model_path = model_path or config.TOMATO_MODEL_PATH
        self.confidence = confidence if confidence is not None else config.TOMATO_CONFIDENCE
        
        # Classes: {0: 'fully_ripened', 1: 'green', 2: 'half_ripened'} or {0: 'b_fully_ripened', ...}
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Tomato model weights not found at: {self.model_path}")

        print(f"[TomatoDetector] Loading model from {self.model_path}...")
        self.model = YOLO(self.model_path, task="detect")
        self.names = self.model.names
        print(f"[TomatoDetector] Loaded successfully with classes: {self.names}")

    @staticmethod
    def verify_tomato_patch(frame: np.ndarray, bbox: list, label: str) -> tuple:
        """
        Multispectral and morphological verification of candidate tomato patch.
        Rejects human faces/skin, walls, clothing, furniture, and non-fruit objects.
        Returns: (is_valid: bool, reason: str)
        """
        if frame is None or frame.size == 0 or len(bbox) != 4:
            return False, "invalid_frame_or_bbox"

        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # 1. Size Constraints: reject micro-artifacts or oversized background regions
        if w < 16 or h < 16:
            return False, f"too_small_{w}x{h}"
        if (w * h) > (0.42 * w_frame * h_frame):
            return False, f"too_large_region_{(w*h)/(w_frame*h_frame):.2f}"

        # 2. Aspect Ratio: genuine tomatoes are spherical or oval (0.60 <= w/h <= 1.55)
        aspect = w / float(h)
        if aspect < 0.60 or aspect > 1.55:
            return False, f"abnormal_aspect_ratio_{aspect:.2f}"

        # 3. Central Core Extraction (18% margin to ignore background walls or holding fingers)
        margin_x = int(w * 0.18)
        margin_y = int(h * 0.18)
        cx1 = max(0, x1 + margin_x)
        cx2 = min(w_frame, x2 - margin_x)
        cy1 = max(0, y1 + margin_y)
        cy2 = min(h_frame, y2 - margin_y)

        if cx2 <= cx1 or cy2 <= cy1:
            crop = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
        else:
            crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            return False, "empty_crop"

        # 4. Colorimetric Analysis (BGR, CIELAB, HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bgr_mean = crop.mean(axis=(0, 1))
        B, G, R = float(bgr_mean[0]), float(bgr_mean[1]), float(bgr_mean[2])

        L_mean, a_mean, b_mean = lab.mean(axis=(0, 1))
        H_mean, S_mean, V_mean = hsv.mean(axis=(0, 1))
        a_star = float(a_mean - 128.0)

        # Wall / neutral gray / dark shadow rejection
        if S_mean < 45.0 or V_mean < 35.0:
            return False, f"wall_or_neutral_low_sat_{S_mean:.1f}"

        clean_label = label.lower().replace("b_", "")

        # Discrimination per tomato maturity class:
        if "fully_ripened" in clean_label or ("ripened" in clean_label and "half" not in clean_label):
            # Fully ripened tomato: high a* (redness > 24.0), high saturation (>= 110), dominant R
            # Human face/skin has a* <= 19.0 and S <= 150 across all Fitzpatrick skin tones
            if a_star < 24.0:
                return False, f"face_or_skin_low_a_star_{a_star:.1f}"
            if S_mean < 110.0:
                return False, f"insufficient_red_saturation_{S_mean:.1f}"
            if R <= G * 1.25 or R <= B * 1.35:
                return False, f"non_red_rgb_profile"
            return True, "valid_fully_ripened"

        elif "half_ripened" in clean_label:
            # Half-ripened tomato (orange/yellow-green turning phase)
            if a_star < 20.0:
                return False, f"half_low_a_star_{a_star:.1f}"
            if S_mean < 130.0:
                return False, f"half_insufficient_saturation_{S_mean:.1f}"
            if not (4.0 <= H_mean <= 30.0):
                return False, f"half_out_of_hue_range_{H_mean:.1f}"
            if R <= B * 1.30:
                return False, f"half_insufficient_red_ratio"
            return True, "valid_half_ripened"

        elif "green" in clean_label:
            # Green tomato: negative a* (chlorophyll green), dominant G channel, Hue 32-88
            if a_star > -8.0:
                return False, f"green_not_chlorophyll_a_star_{a_star:.1f}"
            if S_mean < 75.0:
                return False, f"green_insufficient_saturation_{S_mean:.1f}"
            if not (32.0 <= H_mean <= 88.0):
                return False, f"green_out_of_hue_range_{H_mean:.1f}"
            if G <= R * 1.10 or G <= B * 1.15:
                return False, f"green_non_dominant_g_channel"
            return True, "valid_green"

        return False, f"unknown_class_{clean_label}"

    def detect(self, frame: np.ndarray) -> list:
        """
        Run inference on an OpenCV BGR frame.
        Applies YOLO11 model + multispectral verification filter.
        Returns a list of validated tomato detection dicts.
        """
        if frame is None or self.model is None:
            return []

        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=config.IOU_THRESHOLD,
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
            raw_label = self.names.get(cls_id, f"Class_{cls_id}")
            # Normalize label (remove potential 'b_' prefix from laboro annotations)
            clean_label = raw_label.replace("b_", "")

            # Multispectral & morphological discrimination filter
            # Eliminates human faces, walls, clothing, hands, and background clutter
            is_valid, reject_reason = TomatoDetector.verify_tomato_patch(frame, [x1, y1, x2, y2], clean_label)
            if not is_valid:
                # Silently reject non-tomato false positives
                continue

            detections.append({
                "type": "tomato",
                "label": clean_label,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

        return detections

    @staticmethod
    def group_into_rows(detections: list, y_threshold: int = 100) -> list:
        """
        Spatial clustering of tomato detections into field rows based on vertical centroid (cy).
        """
        if not detections:
            return []

        # Calculate centroids
        items = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            items.append({
                "label": d["label"],
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0
            })

        items.sort(key=lambda item: item["cy"])
        rows = []
        current_row = [items[0]]

        for i in range(1, len(items)):
            d = items[i]
            avg_y = sum(x["cy"] for x in current_row) / len(current_row)
            if abs(d["cy"] - avg_y) <= y_threshold:
                current_row.append(d)
            else:
                rows.append(current_row)
                current_row = [d]
        rows.append(current_row)

        row_stats = []
        for idx, row in enumerate(rows):
            counts = {"fully_ripened": 0, "half_ripened": 0, "green": 0}
            for item in row:
                lbl = item["label"]
                if lbl in counts:
                    counts[lbl] += 1
                else:
                    counts["green"] += 1

            total = sum(counts.values())
            full_pct = (counts["fully_ripened"] / total * 100) if total > 0 else 0
            half_pct = (counts["half_ripened"] / total * 100) if total > 0 else 0
            green_pct = (counts["green"] / total * 100) if total > 0 else 0

            priority = "LOW"
            if full_pct >= 50:
                priority = "HIGH"
            elif full_pct >= 25:
                priority = "MEDIUM"

            row_stats.append({
                "row_number": idx + 1,
                "total": total,
                "counts": counts,
                "green_pct": round(green_pct, 1),
                "half_pct": round(half_pct, 1),
                "full_pct": round(full_pct, 1),
                "priority": priority
            })

        return row_stats
