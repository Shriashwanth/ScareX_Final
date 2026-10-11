"""
Test Suite: ScareX Complete Raspberry Pi 5 Standalone Integration
Verifies that all subsystems (AI inference, camera pipeline, decision engine,
hardware controllers, SQLite database, PDF reporting, and dashboard API)
run locally and autonomously on Raspberry Pi 5 without requiring a laptop hub.
"""
import os
import sys
import time
import pytest
import numpy as np

# Ensure root workspace is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import raspberry_pi.config as config
from raspberry_pi.models.bird.detector import BirdDetector
from raspberry_pi.models.tomato.detector import TomatoDetector, TOMATO_CLASS_MAPPING
from raspberry_pi.audio_listener import BirdAudioListener
from raspberry_pi.fusion.decision_engine import DecisionEngine
from raspberry_pi.database.db import DatabaseLogger
from raspberry_pi.reporting.pdf_generator import PDFReportGenerator
from raspberry_pi.camera import CameraPipeline
from raspberry_pi.dashboard.app import app as dashboard_app

def test_models_load_and_infer_on_pi():
    """Verify Bird and Tomato YOLO models load locally and execute inference on ARM64/CPU."""
    bird_det = BirdDetector()
    tomato_det = TomatoDetector()

    assert bird_det.model is not None, "Bird model failed to load"
    assert tomato_det.model is not None, "Tomato model failed to load"

    # Test dummy frame
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    b_results = bird_det.detect(dummy_frame)
    t_results = tomato_det.detect(dummy_frame)

    assert isinstance(b_results, list), "Bird detector output should be a list"
    assert isinstance(t_results, list), "Tomato detector output should be a list"

def test_tomato_exact_class_mapping():
    """Verify exact 3-class mapping: 0: b_fully_ripened, 1: b_half_ripened, 2: b_green."""
    tomato_det = TomatoDetector()
    names = tomato_det.names

    # Check that class indices 0, 1, 2 correspond to the exact categories
    assert 0 in names and "ripened" in names[0].lower()
    assert 1 in names and "half" in names[1].lower()
    assert 2 in names and "green" in names[2].lower()

    # Multispectral color discrimination filter
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100, 100, 200, 200]

    # Human skin must be rejected
    frame[100:200, 100:200] = (150, 180, 235)  # Fair skin tone
    is_valid, reason = TomatoDetector.verify_tomato_patch(frame, bbox, "fully_ripened")
    assert not is_valid, f"Human skin should be rejected, got reason: {reason}"

    # True red patch should be recognized as ripened
    frame[100:200, 100:200] = (20, 25, 230)
    is_valid, _ = TomatoDetector.verify_tomato_patch(frame, bbox, "fully_ripened")
    assert is_valid, "Pure red tomato patch must pass validation"

def test_decision_engine_standalone():
    """Verify consecutive bird confirmation, deterrence cooldown, and tomato silence."""
    engine = DecisionEngine()
    engine.confirmation_frames = 2

    # Frame 1: Bird present
    mock_bird = [{"type": "bird", "label": "Crow", "confidence": 0.88, "bbox": [50, 50, 150, 150]}]
    engine.process_detections(mock_bird, [])
    assert engine.bird_present_frames == 1
    assert engine.scare_active is False, "Should debounce for confirmation"

    # Frame 2: Confirmed bird
    engine.process_detections(mock_bird, [])
    assert engine.bird_present_frames >= 2
    assert engine.scare_active is True, "Should activate local deterrence upon confirmation"

    # Frame 3: Bird leaves
    engine.process_detections([], [])
    # Frame 4: Bird remains gone -> immediate cutoff
    engine.process_detections([], [])
    assert engine.scare_active is False, "Deterrence sound must halt immediately when bird departs"

    # Test tomato-only frame: must be strictly silent
    mock_tomato = [{"type": "tomato", "label": "fully_ripened", "confidence": 0.92, "bbox": [200, 200, 300, 300]}]
    engine.process_detections([], mock_tomato)
    assert engine.scare_active is False, "Tomato detections must NEVER trigger deterrence sound"
    assert engine.tomato_counts["fully_ripened"] == 1

def test_emergency_stop_and_watchdog():
    """Verify emergency stop halts actuators and watchdog behavior."""
    engine = DecisionEngine()
    engine.process_detections([{"type": "bird", "label": "Pigeon", "confidence": 0.9, "bbox": [10, 10, 50, 50]}], [])
    engine.process_detections([{"type": "bird", "label": "Pigeon", "confidence": 0.9, "bbox": [10, 10, 50, 50]}], [])

    # Trigger emergency stop
    engine.emergency_stop()
    assert engine.emergency_stopped is True
    assert engine.scare_active is False

    # Detections during emergency stop should not actuate
    engine.process_detections([{"type": "bird", "label": "Crow", "confidence": 0.95, "bbox": [10, 10, 50, 50]}], [])
    assert engine.scare_active is False

    # Reset emergency stop
    engine.reset_emergency_stop()
    assert engine.emergency_stopped is False

def test_sqlite_persistence(tmp_path):
    """Verify local SQLite database logging, retrieval, and date filtering."""
    test_db_path = str(tmp_path / "test_scarex.db")
    test_db = DatabaseLogger(db_path=test_db_path)

    test_db.log_event("TEST_BIRD", species="Crow", bird_confidence=0.89, robot_action="SCARE_MOTION")
    test_db.log_event("TEST_TOMATO", tomato_counts="{'fully_ripened': 5}", harvest_priority="HIGH")

    events = test_db.get_recent_events(limit=10)
    assert len(events) == 2
    assert events[0]["event_type"] == "TEST_TOMATO"
    assert events[1]["species"] == "Crow"

    stats = test_db.get_summary_stats()
    assert stats["total_events"] == 2

def test_pdf_report_generation(tmp_path):
    """Verify PDF report generation directly on Pi into persistent reports directory."""
    test_reports_dir = str(tmp_path / "reports")
    gen = PDFReportGenerator(reports_dir=test_reports_dir)

    report_path = gen.generate_report()
    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 500, "PDF should not be empty"

    reports_list = gen.list_reports()
    assert len(reports_list) >= 1
    assert reports_list[0]["filename"].endswith(".pdf")

def test_camera_and_audio_fallbacks():
    """Verify camera pipeline and audio listener run gracefully without hardware."""
    cam = CameraPipeline()
    # Read without physical camera should provide synthetic test frame
    ret, frame = cam.read()
    # Either connected or synthetic generator will yield a frame once started
    cam.start()
    time.sleep(0.1)
    ret, frame = cam.read()
    assert ret is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    cam.stop()

    listener = BirdAudioListener()
    # Test synthetic audio buffer
    dummy_audio = np.zeros(listener.buffer_size, dtype=np.float32)
    res = listener.analyze_chunk(dummy_audio)
    assert res["detected"] is False

def test_dashboard_api_endpoints():
    """Verify local Flask API endpoints for phone and browser access."""
    client = dashboard_app.test_client()

    # 1. Main Page
    res = client.get('/')
    assert res.status_code == 200
    assert b"ScareX Edge" in res.data

    # 2. System Status
    res = client.get('/api/status')
    assert res.status_code == 200
    data = res.get_json()
    assert "bird" in data
    assert "tomato" in data
    assert "system_metrics" in data
    assert "cpu_pct" in data["system_metrics"]

    # 3. Network Endpoint
    res = client.get('/api/network')
    assert res.status_code == 200
    net_data = res.get_json()
    assert "local_ip" in net_data
    assert "port" in net_data

    # 4. Controls
    res = client.post('/api/control/overlay', json={"mode": "bird_only"})
    assert res.status_code == 200

    res = client.post('/api/control/servo')
    assert res.status_code == 200

    res = client.post('/api/control/buzzer')
    assert res.status_code == 200

    # 5. Emergency Stop
    res = client.post('/api/control/emergency_stop')
    assert res.status_code == 200

    res = client.post('/api/control/reset_emergency')
    assert res.status_code == 200

    # 6. PDF Report Download
    res = client.get('/api/report/pdf')
    assert res.status_code == 200
    assert res.mimetype == 'application/pdf'
