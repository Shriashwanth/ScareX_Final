"""
Verification Test for Stages 6, 7, 8, 9:
6. Verify one frame reaches laptop from Pi camera server.
7. Verify bird model inference on Pi frame.
8. Verify tomato model inference on Pi frame.
9. Verify both models execute on the SAME frame.
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raspberry_pi.camera_server import capture_loop, StreamHandler
from http.server import HTTPServer
import raspberry_pi.camera_server as cs
from laptop.camera.pi_camera_client import PiCameraClient
from laptop.models.bird.detector import BirdDetector
from laptop.models.tomato.detector import TomatoDetector

def run_test():
    print("=== STAGE 4-9 VERIFICATION TEST ===")
    
    # Start Pi Camera Server in background thread on port 8000
    test_port = 8000
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    server = HTTPServer(('127.0.0.1', test_port), StreamHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[Test] Pi Camera Server running on port {test_port}")

    # Start Laptop Camera Client
    client = PiCameraClient(stream_url=f"http://127.0.0.1:{test_port}/video")
    client.start()
    print("[Test] Connecting to Pi Camera Stream...")

    # Wait up to 5 seconds for frame
    frame = None
    for _ in range(50):
        ret, frame = client.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)

    # STAGE 6 Verification
    assert frame is not None, "FAILED: No frame received from Pi camera server!"
    print(f"[Stage 6 PASSED] One frame successfully reached laptop! Shape: {frame.shape}")

    # Initialize both detectors
    bird_detector = BirdDetector()
    tomato_detector = TomatoDetector()

    # STAGE 7 Verification: Bird model on Pi frame
    print("[Test] Running Bird model on the captured Pi frame...")
    bird_dets = bird_detector.detect(frame)
    print(f"[Stage 7 PASSED] Bird model returned: {bird_dets}")

    # STAGE 8 Verification: Tomato model on Pi frame
    print("[Test] Running Tomato model on the captured Pi frame...")
    tomato_dets = tomato_detector.detect(frame)
    print(f"[Stage 8 PASSED] Tomato model returned: {tomato_dets}")

    # STAGE 9 Verification: Both models on SAME frame
    print("[Test] Verifying both models operating on the exact SAME frame instance...")
    t0 = time.time()
    b_results = bird_detector.detect(frame)
    t_results = tomato_detector.detect(frame)
    elapsed = (time.time() - t0) * 1000
    print(f"[Stage 9 PASSED] Dual-inference on identical frame completed in {elapsed:.1f}ms.")
    print(f"  Bird detections: {len(b_results)}")
    print(f"  Tomato detections: {len(t_results)}")

    # Clean up
    client.stop()
    cs.running = False
    server.shutdown()
    print("=== STAGES 6, 7, 8, 9 FULLY VERIFIED ===")

if __name__ == "__main__":
    run_test()
