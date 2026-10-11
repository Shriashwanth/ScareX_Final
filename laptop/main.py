"""
ScareX Integrated System - Laptop Main Orchestrator
Consumes camera stream (either Laptop Webcam for dev/testing or Pi Camera for deployment),
executes independent Bird & Tomato YOLO models on the SAME frame,
combines findings via the Decision/Fusion Engine, controls the Pi edge node, and serves the Web Dashboard.
"""
import sys
import os
import time
import threading
import cv2

# Ensure laptop and root packages are in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
for p in [BASE_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import laptop.config as config
    from laptop.camera.pi_camera_client import PiCameraClient
    from laptop.camera.laptop_camera_client import LaptopCameraClient
    from laptop.models.bird.detector import BirdDetector
    from laptop.models.tomato.detector import TomatoDetector
    from laptop.fusion.decision_engine import decision_engine
    from laptop.communication.pi_client import pi_client
    from laptop.database.db import db
    from laptop.dashboard.app import app, set_camera_client, update_annotated_frame, shared_state
except ImportError:
    import config
    from camera.pi_camera_client import PiCameraClient
    from camera.laptop_camera_client import LaptopCameraClient
    from models.bird.detector import BirdDetector
    from models.tomato.detector import TomatoDetector
    from fusion.decision_engine import decision_engine
    from communication.pi_client import pi_client
    from database.db import db
    from dashboard.app import app, set_camera_client, update_annotated_frame, shared_state

def draw_detections(frame, bird_dets, tomato_dets, mode="both"):
    """
    Renders bounding boxes and status labels on a copy of the frame based on active overlay mode.
    """
    annotated = frame.copy()

    # Draw Tomato Bounding Boxes
    if mode in ["both", "tomato_only"]:
        for t in tomato_dets:
            x1, y1, x2, y2 = t["bbox"]
            lbl = t["label"]
            conf = t["confidence"]

            if lbl == "green":
                color = (0, 220, 0)       # Green
            elif lbl == "half_ripened":
                color = (0, 180, 255)     # Amber/Yellow
            else:
                color = (0, 0, 255)       # Red (fully ripened)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            tag = f"{lbl} {conf*100:.0f}%"
            (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated, tag, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw Bird Bounding Boxes
    if mode in ["both", "bird_only"]:
        for b in bird_dets:
            x1, y1, x2, y2 = b["bbox"]
            lbl = b["label"]
            conf = b["confidence"]
            color = (255, 100, 0)  # Bright cyan/blue border

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            tag = f"BIRD {conf*100:.1f}%" if lbl.lower() == "bird" else f"BIRD: {lbl} {conf*100:.1f}%"
            (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - 24), (x1 + w, y1), color, -1)
            cv2.putText(annotated, tag, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated

def main():
    print("==========================================================")
    print("        ScareX Integrated Autonomous Surveillance         ")
    print("==========================================================")
    print(f"Camera Mode: {config.CAMERA_MODE.upper()}")
    if config.CAMERA_MODE.upper() == "LAPTOP":
        print(f"Camera Source: Laptop Webcam (Device {config.WEBCAM_INDEX}) [Development Mode]")
    else:
        print(f"Camera Source: Pi Camera Network Stream ({config.CAMERA_STREAM_URL}) [Deployment Mode]")
    print(f"Target Pi Node: {config.PI_IP}")
    print(f"Dashboard will be hosted at: http://127.0.0.1:{config.DASHBOARD_PORT}")
    print("----------------------------------------------------------")

    # 1. Initialize Camera Client based on CAMERA_MODE
    if config.CAMERA_MODE.upper() == "LAPTOP":
        print("[Main] Initializing Laptop Webcam Client (Development Mode)...")
        camera_client = LaptopCameraClient()
    else:
        print("[Main] Initializing Pi Camera Network Client (Deployment Mode)...")
        camera_client = PiCameraClient()

    set_camera_client(camera_client)
    camera_client.start()

    # 2. Initialize Bird Model
    print("[Main] Initializing Bird Detection Model...")
    bird_detector = BirdDetector()

    # 3. Initialize Tomato Model
    print("[Main] Initializing Tomato Maturity Model...")
    tomato_detector = TomatoDetector()

    # 3.5. Start Microphone Bird Audio Recognition
    print("[Main] Starting Real-Time Microphone Bird Audio Recognition...")
    try:
        from laptop.audio.bird_audio_listener import bird_audio_listener
        bird_audio_listener.start()
    except Exception as e:
        print(f"[Main] Microphone audio listener note: {e}")

    # 4. Start Heartbeat (if connecting to Raspberry Pi or simulation)
    if config.CAMERA_MODE.upper() == "PI" or os.getenv("SCAREX_ENABLE_HEARTBEAT", "true").lower() == "true":
        print("[Main] Starting Raspberry Pi communication watchdog heartbeat...")
        pi_client.start_heartbeat()

    # 5. Start Integrated Flask Dashboard
    print(f"[Main] Starting Web Dashboard on port {config.DASHBOARD_PORT}...")
    dashboard_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=config.DASHBOARD_PORT, debug=False, use_reloader=False),
        daemon=True
    )
    dashboard_thread.start()

    db.log_event("SYSTEM_INIT", robot_action="SYSTEM_READY", details=f"Mode: {config.CAMERA_MODE.upper()}")
    print("\n[Main] All subsystems active! Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to terminate.")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            # Check if monitoring is active from dashboard
            if not shared_state["system_active"]:
                time.sleep(0.1)
                continue

            # Ingest ONE shared frame from active camera client
            success, frame = camera_client.read()
            if not success or frame is None:
                time.sleep(0.04)
                continue

            # Pass the EXACT SAME frame to both models independently
            bird_dets = bird_detector.detect(frame)
            tomato_dets = tomato_detector.detect(frame)

            # Pass simultaneous detections to decision & fusion engine
            decision_engine.process_detections(bird_dets, tomato_dets)

            # Compute FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-6))
            prev_time = now

            # Render annotations according to selected dashboard overlay mode
            mode = shared_state.get("overlay_mode", "both")
            if mode != "raw":
                annotated = draw_detections(frame, bird_dets, tomato_dets, mode=mode)
            else:
                annotated = frame.copy()

            # Add FPS and camera mode watermark
            cam_label = "LAPTOP WEBCAM" if config.CAMERA_MODE.upper() == "LAPTOP" else "PI CAMERA"
            cv2.putText(annotated, f"FPS: {fps:.1f} | Source: {cam_label}", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Update live feed for dashboard users
            update_annotated_frame(annotated)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Main] Shutting down ScareX Laptop System...")
    finally:
        try:
            from laptop.audio.bird_audio_listener import bird_audio_listener
            bird_audio_listener.stop()
        except Exception:
            pass
        pi_client.stop_heartbeat()
        camera_client.stop()
        db.log_event("SYSTEM_SHUTDOWN", robot_action="CLEAN_EXIT")
        print("[Main] Shutdown complete.")

if __name__ == "__main__":
    main()
