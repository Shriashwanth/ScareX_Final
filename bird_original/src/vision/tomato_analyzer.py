import cv2
import time
import json
from ultralytics import YOLO

class TomatoAnalyzer:
    def __init__(self, weights_path="yolo11n_ncnn_model"):
        self.weights_path = weights_path
        self.model = YOLO(self.weights_path)
        self.cap = None
        self.running = False
        self.session_stats = {'fully_ripened': 0, 'green': 0, 'half_ripened': 0}
        self.best_row_stats = []
        self.max_total_detected = 0
        self.frame_count = 0
        self.start_time = 0
        self.current_frame = None
        self.class_names = {0: 'fully_ripened', 1: 'half_ripened', 2: 'green'}
        
    def start(self, source):
        self.stop()
        self.session_stats = {'fully_ripened': 0, 'green': 0, 'half_ripened': 0}
        self.best_row_stats = []
        self.max_total_detected = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Determine source
        if str(source).isdigit():
            source_val = int(source)
        else:
            source_val = source
            
        self.cap = cv2.VideoCapture(source_val)
        if not self.cap.isOpened():
            print(f"Error: Could not open source {source_val}.")
            return False
            
        self.running = True
        return True
        
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def group_into_rows(self, detections, y_threshold=100):
        if not detections:
            return []
        
        # Sort by Y coordinate
        detections.sort(key=lambda d: d['cy'])
        
        rows = []
        current_row = [detections[0]]
        
        for i in range(1, len(detections)):
            d = detections[i]
            # If within threshold of the average Y of the current row, add to current row
            avg_y = sum(item['cy'] for item in current_row) / len(current_row)
            if abs(d['cy'] - avg_y) <= y_threshold:
                current_row.append(d)
            else:
                rows.append(current_row)
                current_row = [d]
        rows.append(current_row)
        
        row_stats = []
        for idx, row in enumerate(rows):
            counts = {'fully_ripened': 0, 'green': 0, 'half_ripened': 0}
            for d in row:
                counts[d['class_name']] += 1
            
            total = sum(counts.values())
            full_pct = (counts['fully_ripened'] / total * 100) if total > 0 else 0
            
            priority = "LOW"
            if full_pct >= 50:
                priority = "HIGH"
            elif full_pct >= 25:
                priority = "MEDIUM"
                
            row_stats.append({
                "row_number": idx + 1,
                "total": total,
                "counts": counts,
                "green_pct": round((counts['green'] / total * 100) if total > 0 else 0, 1),
                "half_pct": round((counts['half_ripened'] / total * 100) if total > 0 else 0, 1),
                "full_pct": round(full_pct, 1),
                "priority": priority
            })
            
        return row_stats
            
    def get_frame(self):
        if not self.running or not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return None
            
        self.frame_count += 1
        
        # Run inference
        results = self.model.predict(frame, conf=0.60, iou=0.45, verbose=False)
        result = results[0]
        
        frame_counts = {'fully_ripened': 0, 'green': 0, 'half_ripened': 0}
        frame_detections = []
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.class_names.get(cls_id, f"Class {cls_id}")
            
            cy = (y1 + y2) / 2
            cx = (x1 + x2) / 2
            
            frame_counts[class_name] += 1
            frame_detections.append({
                "class_name": class_name,
                "cx": cx,
                "cy": cy
            })
            
            # Draw
            color = (0, 255, 0) if cls_id == 2 else (0, 0, 255) if cls_id == 0 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
        # Update max-observed session logic
        for k in self.session_stats:
            if frame_counts[k] > self.session_stats[k]:
                self.session_stats[k] = frame_counts[k]
                
        # For row-wise analysis, capture the rows of the frame with the highest total detections
        total_frame = sum(frame_counts.values())
        if total_frame > self.max_total_detected:
            self.max_total_detected = total_frame
            self.best_row_stats = self.group_into_rows(frame_detections)
            
        # Draw session stats summary on video
        summary = f"Max Observed - Fully: {self.session_stats['fully_ripened']} | Green: {self.session_stats['green']} | Half: {self.session_stats['half_ripened']}"
        cv2.putText(frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        self.current_frame = frame
        return frame
