import cv2
import time
import argparse
import json
from ultralytics import YOLO

CLASS_NAMES = {
    0: 'b_fully_ripened',
    1: 'b_half_ripened',
    2: 'b_green'
}

def group_into_rows(detections, y_threshold=100):
    if not detections:
        return []
    detections.sort(key=lambda d: d['cy'])
    rows = []
    current_row = [detections[0]]
    for i in range(1, len(detections)):
        d = detections[i]
        avg_y = sum(item['cy'] for item in current_row) / len(current_row)
        if abs(d['cy'] - avg_y) <= y_threshold:
            current_row.append(d)
        else:
            rows.append(current_row)
            current_row = [d]
    rows.append(current_row)
    
    row_stats = []
    for idx, row in enumerate(rows):
        counts = {0: 0, 1: 0, 2: 0}
        for d in row:
            counts[d['cls_id']] += 1
        total = sum(counts.values())
        full_pct = (counts[0] / total * 100) if total > 0 else 0
        priority = "LOW"
        if full_pct >= 50:
            priority = "HIGH"
        elif full_pct >= 25:
            priority = "MEDIUM"
            
        row_stats.append({
            "row_number": idx + 1,
            "total": total,
            "counts": counts,
            "green_pct": round((counts[2] / total * 100) if total > 0 else 0, 1),
            "half_pct": round((counts[1] / total * 100) if total > 0 else 0, 1),
            "full_pct": round(full_pct, 1),
            "priority": priority
        })
    return row_stats

def main():
    parser = argparse.ArgumentParser(description="Real-time Tomato Maturity Detection")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--source", type=str, default="0", help="Webcam source ID or video file path")
    parser.add_argument("--weights", type=str, default="yolo11n_ncnn_model", help="Path to model weights")
    args = parser.parse_args()

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.logging.db_logger import db_logger

    source_val = args.source
    if source_val.isdigit():
        source_val = int(source_val)

    print(f"Loading model from {args.weights}...")
    try:
        model = YOLO(args.weights)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Error: Could not open source {source_val}.")
        return

    print(f"Starting inference on {source_val}... Press 'q' to quit.")

    prev_time = 0
    max_session_detections = {0: 0, 1: 0, 2: 0}
    best_row_stats = []
    max_total_detected = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to capture image.")
            break

        results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
        new_time = time.time()
        fps = 1 / (new_time - prev_time) if prev_time > 0 else 0
        prev_time = new_time
        
        result = results[0]
        frame_count += 1
        
        frame_counts = {0: 0, 1: 0, 2: 0}
        frame_detections = []
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            frame_counts[cls_id] += 1
            frame_detections.append({'cls_id': cls_id, 'cy': (y1+y2)/2, 'cx': (x1+x2)/2})
            
            class_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            label = f"{class_name} {conf:.2f}"
            color = (0, 255, 0) if cls_id == 2 else (0, 0, 255) if cls_id == 0 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        for k in max_session_detections:
            if frame_counts[k] > max_session_detections[k]:
                max_session_detections[k] = frame_counts[k]
                
        total_frame = sum(frame_counts.values())
        if total_frame > max_total_detected:
            max_total_detected = total_frame
            best_row_stats = group_into_rows(frame_detections)

        summary = f"Max Observed - Fully: {max_session_detections[0]} | Green: {max_session_detections[2]} | Half: {max_session_detections[1]}"
        cv2.putText(frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Tomato Maturity Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n--- FINAL SESSION ANALYSIS ---")
    print(summary)
    
    total = sum(max_session_detections.values())
    full_pct = (max_session_detections[0] / total * 100) if total > 0 else 0
    priority = "LOW"
    if full_pct >= 50:
        priority = "HIGH"
    elif full_pct >= 25:
        priority = "MEDIUM"
        
    print(f"Overall Harvesting Priority: {priority}")
    print("\nRow-wise Analysis:")
    for row in best_row_stats:
        print(f"Row {row['row_number']} | Total: {row['total']} | Fully: {row['full_pct']}% | Priority: {row['priority']}")
    
    print("\nDisclaimer: Percentages are calculated from tomatoes detected within the camera view. Hidden, occluded, or out-of-frame tomatoes may not be detected.")
    print("Disclaimer: Harvesting priority is a prototype decision-support indicator based on detected maturity distribution and is not a guaranteed agronomic prediction.\n")

    db_logger.log_event(
        event_type="MONITORING_SESSION_END",
        species="Tomato",
        confidence=1.0,
        alarm_played="None",
        system_status=f"Processed {frame_count} frames. Row Data: {json.dumps(best_row_stats)}",
        detection_source=str(source_val),
        duration=0,
        green_count=max_session_detections[2],
        half_ripened_count=max_session_detections[1],
        fully_ripened_count=max_session_detections[0],
        harvest_priority=priority
    )
    print("Session saved successfully to SQLite.")

if __name__ == "__main__":
    main()
