import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO
import os

def test_image_detection():
    # 1. Download a sample image of a bird
    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    print("Downloading test image from unsplash...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    
    # 2. Load the model
    print("Loading YOLOv11n model...")
    model = YOLO("yolo11n.pt")
    
    # 3. Run inference
    print("Running inference...")
    results = model(img)
    
    # 4. Check results and draw boxes
    detected = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            print(f"Detected: {class_name} with {conf*100:.1f}% confidence")
            
            if class_name.lower() == 'bird' or class_name.lower() == 'crow':
                detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"Bird (Simulated Test) {conf*100:.1f}%"
                cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
    if detected:
        print("Success! A bird was detected.")
        cv2.imwrite("test_output.jpg", img)
        print("Saved output with bounding boxes to test_output.jpg")
    else:
        print("Failed to detect a bird.")

if __name__ == "__main__":
    test_image_detection()
