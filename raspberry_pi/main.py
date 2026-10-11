"""
ScareX Standalone Raspberry Pi 5 - Master Autonomous Orchestrator
Runs all AI inference (Bird YOLO11n, Tomato YOLO11n, Acoustic Classifier),
direct camera processing, decision-making, hardware actuation, SQLite logging,
and Web Dashboard locally on Raspberry Pi 5 without needing a laptop AI hub.
"""
import sys
import os
import time
import signal
import threading
import cv2
import psutil

# Ensure raspberry_pi and workspace root are in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
for p in [BASE_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import raspberry_pi.config as config
from raspberry_pi.camera import camera_pipeline
from raspberry_pi.models.bird.detector import BirdDetector
from raspberry_pi.models.tomato.detector import TomatoDetector
from raspberry_pi.audio_listener import bird_audio_listener
from raspberry_pi.fusion.decision_engine import decision_engine
from raspberry_pi.hardware import hardware
from raspberry_pi.audio import audio_player
from raspberry_pi.database.db import db
from raspberry_pi.dashboard.app import app, update_annotated_frame, shared_state, get_local_ip

running = True

def signal_handler(sig, frame):
    global running
    print("\n[ScareX-Pi5] Shutdown signal received. Performing graceful exit...")
    running = False

def draw_detections(frame, bird_dets, tomato_dets, mode="both"):
    """
    Renders high-contrast bounding boxes and status labels on a copy of the frame.
    """
    annotated = frame.copy()

    # Draw Tomato Bounding Boxes
    if mode in ["both", "tomato_only"]:
        for t in (tomato_dets or []):
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
            (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - 18), (x1 + w, y1), color, -1)
            cv2.putText(annotated, tag, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Draw Bird Bounding Boxes
    if mode in ["both", "bird_only"]:
        for b in (bird_dets or []):
            x1, y1, x2, y2 = b["bbox"]
            lbl = b["label"]
            conf = b["confidence"]
            color = (255, 100, 0)  # Bright blue/cyan

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            tag = f"BIRD {conf*100:.1f}%" if lbl.lower() == "bird" else f"BIRD: {lbl} {conf*100:.1f}%"
            (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - 22), (x1 + w, y1), color, -1)
            cv2.putText(annotated, tag, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return annotated

def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pi_ip = get_local_ip()

    print("==========================================================")
    print("      ScareX Standalone - Raspberry Pi 5 Autonomous Node  ")
    print("==========================================================")
    print(f"Host IP Address:        http://{pi_ip}:{config.DASHBOARD_PORT}")
    print(f"Local mDNS Access:      http://raspberrypi.local:{config.DASHBOARD_PORT}")
    print(f"Camera Mode:            Local Raspberry Pi Capture ({config.CAMERA_SOURCE})")
    print(f"Database:               Persistent SQLite ({config.DB_PATH})")
    print(f"Hardware Drivers:       {'Mock (Dev/Simulation)' if hardware.is_mock else 'Raspberry Pi 5 GPIO (lgpio)'}")
    print("==========================================================")

    # 1. Start Local Camera Pipeline
    print("[Main] Starting local camera pipeline...")
    camera_pipeline.start()

    # 2. Initialize Bird Model (NCNN / PyTorch)
    print(f"[Main] Loading Bird Model from {config.BIRD_MODEL_PATH}...")
    bird_detector = BirdDetector(model_path=config.BIRD_MODEL_PATH, confidence=config.BIRD_CONFIDENCE)

    # 3. Initialize Tomato Model (NCNN / PyTorch)
    print(f"[Main] Loading Tomato Model from {config.TOMATO_MODEL_PATH}...")
    tomato_detector = TomatoDetector(model_path=config.TOMATO_MODEL_PATH, confidence=config.TOMATO_CONFIDENCE)

    # 4. Start Microphone Acoustic Bird Classifier (if enabled)
    if getattr(config, "ENABLE_AUDIO_CLASSIFIER", True):
        print("[Main] Initializing Microphone Acoustic Bird Classifier...")
        try:
            bird_audio_listener.start()
        except Exception as e:
            print(f"[Main] Audio listener note: {e}")

    # 5. Start Local Web Dashboard & API
    print(f"[Main] Launching Local Web Dashboard on port {config.DASHBOARD_PORT}...")
    dashboard_thread = threading.Thread(
        target=lambda: app.run(
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            debug=False,
            use_reloader=False,
            threaded=True
        ),
        daemon=True
    )
    dashboard_thread.start()

    db.log_event("SYSTEM_INIT", robot_action="STANDALONE_READY", details=f"Host: {pi_ip}:{config.DASHBOARD_PORT}")
    print(f"\n[Main] ALL SUBSYSTEMS ONLINE! Connect from your phone or browser at http://{pi_ip}:{config.DASHBOARD_PORT}\n")

    prev_time = time.time()
    fps = 0.0

    try:
        while running:
            # Check if monitoring is active
            if not shared_state["system_active"]:
                time.sleep(0.1)
                continue

            # Ingest ONE shared frame directly from local memory
            success, frame = camera_pipeline.read()
            if not success or frame is None:
                time.sleep(0.02)
                continue

            t_start = time.time()

            # Execute dual inference on the SAME shared frame in local memory
            bird_dets = bird_detector.detect(frame)
            tomato_dets = tomato_detector.detect(frame)

            t_infer = time.time()
            latency_ms = (t_infer - t_start) * 1000.0

            # Pass simultaneous detections to local in-process Decision Engine
            decision_engine.process_detections(bird_dets, tomato_dets)

            # Compute smoothed FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-6))
            prev_time = now

            # Render annotations according to selected overlay mode
            mode = shared_state.get("overlay_mode", "both")
            if mode != "raw":
                annotated = draw_detections(frame, bird_dets, tomato_dets, mode=mode)
            else:
                annotated = frame.copy()

            # Watermark with FPS, latency, and hardware source
            cam_src = camera_pipeline.active_source
            cv2.putText(
                annotated,
                f"Pi 5 FPS: {fps:.1f} | Latency: {latency_ms:.0f}ms | Source: {cam_src}",
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            # Update live feed for dashboard users (phones and desktop)
            update_annotated_frame(annotated, fps=fps, latency_ms=latency_ms)

            # Brief sleep to yield CPU gracefully on ARM64
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[Main] User interrupted via keyboard.")
    finally:
        print("[Main] Shutting down ScareX Pi 5 subsystems...")
        try:
            bird_audio_listener.stop()
        except Exception:
            pass
        camera_pipeline.stop()
        audio_player.stop()
        hardware.emergency_stop()
        db.log_event("SYSTEM_SHUTDOWN", robot_action="CLEAN_EXIT")
        print("[Main] System stopped cleanly.")

if __name__ == "__main__":
    main()
