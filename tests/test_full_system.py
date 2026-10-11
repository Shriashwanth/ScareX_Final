"""
Full End-to-End System Integration Test
Verifies Pi servers, camera client, dual YOLO detectors, decision engine,
Flask dashboard, API endpoints, and PDF generation simultaneously.
"""
import sys
import os
import time
import threading
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import raspberry_pi.camera_server as cs
from http.server import HTTPServer
from raspberry_pi.command_server import app as pi_app
from laptop.camera.pi_camera_client import PiCameraClient
from laptop.models.bird.detector import BirdDetector
from laptop.models.tomato.detector import TomatoDetector
from laptop.fusion.decision_engine import decision_engine
from laptop.communication.pi_client import pi_client
from laptop.dashboard.app import app as dashboard_app, set_camera_client, update_annotated_frame

def run_e2e_test():
    print("=== STARTING COMPLETE SYSTEM INTEGRATION TEST ===")
    
    # 1. Start Pi Camera Server (:8000)
    capture_thread = threading.Thread(target=cs.capture_loop, daemon=True)
    capture_thread.start()
    cam_server = HTTPServer(('127.0.0.1', 8000), cs.StreamHandler)
    threading.Thread(target=cam_server.serve_forever, daemon=True).start()
    print("[1] Pi Camera Server running on :8000")

    # 2. Start Pi Command Server (:9000)
    threading.Thread(target=lambda: pi_app.run(host="127.0.0.1", port=9000, debug=False, use_reloader=False), daemon=True).start()
    print("[2] Pi Command Server running on :9000")

    time.sleep(1.0)

    # 3. Start Laptop Pi Camera Client
    cam_client = PiCameraClient("http://127.0.0.1:8000/video")
    set_camera_client(cam_client)
    cam_client.start()
    print("[3] Laptop Pi Camera Client started")

    # 4. Start Heartbeat
    pi_client.start_heartbeat()
    print("[4] Heartbeat loop started")

    # 5. Start Laptop Dashboard (:5000)
    threading.Thread(target=lambda: dashboard_app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False), daemon=True).start()
    print("[5] Integrated Web Dashboard running on :5000")

    # 6. Initialize Models
    print("[6] Initializing Bird and Tomato Detectors...")
    bird_det = BirdDetector()
    tomato_det = TomatoDetector()

    # 7. Process several frames
    print("[7] Ingesting and processing frames across both models...")
    processed_count = 0
    for _ in range(15):
        ret, frame = cam_client.read()
        if ret and frame is not None:
            b_res = bird_det.detect(frame)
            t_res = tomato_det.detect(frame)
            decision_engine.process_detections(b_res, t_res)
            update_annotated_frame(frame)
            processed_count += 1
        time.sleep(0.1)

    assert processed_count > 0, "No frames processed in inference loop!"
    print(f"  Successfully processed {processed_count} frames across both models.")

    # 8. Test Dashboard HTTP Endpoints
    print("[8] Testing Dashboard REST API Endpoints...")
    res = requests.get("http://127.0.0.1:5000/api/status", timeout=2.0)
    assert res.status_code == 200, f"Dashboard /api/status failed: {res.status_code}"
    status_json = res.json()
    print(f"  /api/status OK. Camera connected: {status_json['camera']['connected']}, Pi connected: {status_json['pi_node']['connected']}")

    # 9. Test PDF Export
    print("[9] Testing PDF Report generation...")
    pdf_res = requests.get("http://127.0.0.1:5000/api/report/pdf", timeout=5.0)
    assert pdf_res.status_code == 200, f"PDF export failed: {pdf_res.status_code}"
    assert len(pdf_res.content) > 1000, "PDF content is unexpectedly small"
    print(f"  PDF Report successfully generated and downloaded ({len(pdf_res.content)} bytes).")

    # 10. Clean up
    cam_client.stop()
    pi_client.stop_heartbeat()
    cs.running = False
    cam_server.shutdown()
    print("=== END-TO-END SYSTEM INTEGRATION TEST PASSED 100%! ===")

if __name__ == "__main__":
    run_e2e_test()
