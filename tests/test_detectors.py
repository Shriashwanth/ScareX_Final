"""
Test script to verify both BirdDetector and TomatoDetector abstractions.
Uses a mock frame (or test image) without opening camera.
"""
import sys
import os
import numpy as np
import cv2

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laptop.models.bird.detector import BirdDetector
from laptop.models.tomato.detector import TomatoDetector

def test_detectors():
    print("--- Testing BirdDetector ---")
    bird_detector = BirdDetector()
    
    print("--- Testing TomatoDetector ---")
    tomato_detector = TomatoDetector()

    # Create synthetic test frame (640x640x3)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    # Add a colored circle
    cv2.circle(dummy_frame, (320, 320), 100, (0, 0, 255), -1)

    print("Running bird_detector.detect(dummy_frame)...")
    b_dets = bird_detector.detect(dummy_frame)
    print(f"Bird detections on dummy frame: {b_dets} (expected 0 or low-confidence)")

    print("Running tomato_detector.detect(dummy_frame)...")
    t_dets = tomato_detector.detect(dummy_frame)
    print(f"Tomato detections on dummy frame: {t_dets}")

    print("\nDetectors successfully instantiated and executed inference on shared frame!")

if __name__ == "__main__":
    test_detectors()
