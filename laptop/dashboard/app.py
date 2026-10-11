"""
Integrated Web Dashboard for ScareX System
Provides live stream with selectable detection overlays, real-time bird deterrence telemetry,
tomato maturity crop analysis, Pi hardware diagnostics, PDF reporting, and robot controls.
"""
import os
import time
import json
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file
from fpdf import FPDF

import laptop.config as config
from laptop.communication.pi_client import pi_client
from laptop.database.db import db
from laptop.fusion.decision_engine import decision_engine

app = Flask(__name__, template_folder="templates")

# Shared state between main inference worker and web dashboard
shared_state = {
    "annotated_frame": None,
    "system_active": True,
    "overlay_mode": "both",  # "both", "bird_only", "tomato_only", "raw"
    "camera_client": None
}

def set_camera_client(cam_client):
    shared_state["camera_client"] = cam_client

def update_annotated_frame(frame):
    shared_state["annotated_frame"] = frame

def generate_video_stream():
    """
    Generator yielding annotated live frames from the Pi camera to the web browser.
    """
    while True:
        frame = shared_state["annotated_frame"]
        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.04)  # ~25 FPS web delivery

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_system_status():
    fusion_status = decision_engine.get_status()
    pi_hw_status = pi_client.get_status()
    cam_status = shared_state["camera_client"].get_status() if shared_state["camera_client"] else {"connected": False}

    return jsonify({
        "system_active": shared_state["system_active"],
        "overlay_mode": shared_state["overlay_mode"],
        "camera": cam_status,
        "pi_node": {
            "connected": pi_client.is_connected,
            "telemetry": pi_hw_status
        },
        "bird": fusion_status["bird"],
        "tomato": fusion_status["tomato"],
        "metrics": fusion_status["system"]
    })

@app.route('/api/control/overlay', methods=['POST'])
def set_overlay():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "both").lower()
    if mode in ["both", "bird_only", "tomato_only", "raw"]:
        shared_state["overlay_mode"] = mode
        return jsonify({"success": True, "overlay_mode": mode})
    return jsonify({"success": False, "error": "Invalid mode"}), 400

@app.route('/api/control/start', methods=['POST'])
def start_system():
    shared_state["system_active"] = True
    db.log_event("SYSTEM_START", robot_action="MONITORING_RESUMED")
    return jsonify({"success": True, "system_active": True})

@app.route('/api/control/stop', methods=['POST'])
def stop_system():
    shared_state["system_active"] = False
    db.log_event("SYSTEM_STOP", robot_action="MONITORING_PAUSED")
    return jsonify({"success": True, "system_active": False})

@app.route('/api/control/emergency_stop', methods=['POST'])
def emergency_stop():
    res = pi_client.emergency_stop()
    shared_state["system_active"] = False
    db.log_event("EMERGENCY_STOP", robot_action="ALL_ACTUATORS_HALTED", details=json.dumps(res))
    return jsonify({"success": True, "pi_response": res})

@app.route('/api/control/servo', methods=['POST'])
def sweep_servo():
    res = pi_client.sweep_camera()
    db.log_event("MANUAL_SERVO_SWEEP", robot_action="CAMERA_ROTATED")
    return jsonify(res)

@app.route('/api/control/move', methods=['POST'])
def move_robot():
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "stop")
    res = pi_client.move(direction)
    return jsonify(res)

@app.route('/api/control/buzzer', methods=['POST'])
def test_buzzer():
    try:
        from laptop.buzzer import play_buzzer
    except ImportError:
        from buzzer import play_buzzer
    play_buzzer()
    db.log_event("MANUAL_BUZZER_TEST", robot_action="BUZZER_SOUND_ACTIVATED")
    return jsonify({"success": True, "message": "Buzzer sound activated"})

@app.route('/api/logs')
def get_recent_logs():
    logs = db.get_recent_events(limit=25)
    return jsonify(logs)

@app.route('/api/report/pdf')
def generate_pdf_report():
    status = decision_engine.get_status()
    tomato = status["tomato"]
    bird = status["bird"]
    recent_events = db.get_recent_events(limit=15)

    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, txt="ScareX Integrated Autonomous System", ln=True, align='C')
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 8, txt="Field Surveillance & Crop Maturity Intelligence Report", ln=True, align='C')
    pdf.ln(6)

    # Metadata
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, txt=f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, txt=f"Source Stream: Pi Camera ({config.CAMERA_STREAM_URL})", ln=True)
    pdf.ln(6)

    # Section 1: Tomato Crop Maturity
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, txt="1. Tomato Crop Ripeness Distribution", ln=True)
    pdf.set_font("Arial", size=11)
    
    max_counts = tomato["session_max_counts"]
    total = tomato["total_observed"]
    pdf.cell(0, 6, txt=f"Fully-Ripened: {max_counts['fully_ripened']} ({tomato['full_pct']}%)", ln=True)
    pdf.cell(0, 6, txt=f"Half-Ripened:  {max_counts['half_ripened']} ({tomato['half_pct']}%)", ln=True)
    pdf.cell(0, 6, txt=f"Green:         {max_counts['green']} ({tomato['green_pct']}%)", ln=True)
    pdf.cell(0, 6, txt=f"Total Observed: {total}", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=f"Recommended Harvesting Priority: {tomato['harvest_priority']}", ln=True)
    pdf.ln(4)

    # Section 2: Row-wise Analysis
    if tomato["row_analysis"]:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 6, txt="Row-wise Spatial Analysis:", ln=True)
        pdf.set_font("Arial", size=10)
        for r in tomato["row_analysis"]:
            row_str = f"Row {r['row_number']}: Total {r['total']} | Green: {r['green_pct']}% | Half: {r['half_pct']}% | Fully: {r['full_pct']}% -> Priority: {r['priority']}"
            pdf.cell(0, 6, txt=row_str, ln=True)
        pdf.ln(4)

    # Section 3: Bird Protection Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, txt="2. Bird Deterrence Surveillance", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, txt=f"Confirmation Threshold: {config.BIRD_CONFIRMATION_FRAMES} frames", ln=True)
    pdf.cell(0, 6, txt=f"Safety Cooldown: {config.BIRD_COOLDOWN_SECONDS} seconds", ln=True)
    pdf.ln(4)

    # Section 4: Recent Event Log
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 6, txt="Recent Security & Harvest Logs:", ln=True)
    pdf.set_font("Arial", size=9)
    for ev in recent_events[:8]:
        ev_str = f"[{ev['timestamp']}] {ev['event_type']} | Species: {ev['species']} | Conf: {ev['bird_confidence']*100:.0f}% | Action: {ev['robot_action']}"
        pdf.cell(0, 5, txt=ev_str, ln=True)

    report_path = os.path.join(config.BASE_DIR, "scarex_integrated_report.pdf")
    pdf.output(report_path)
    return send_file(os.path.abspath(report_path), as_attachment=True, download_name="scarex_report.pdf")

def run_dashboard(port=None):
    web_port = port or config.DASHBOARD_PORT
    print(f"[Dashboard] Starting Flask Web UI on http://0.0.0.0:{web_port}")
    app.run(host="0.0.0.0", port=web_port, debug=False, use_reloader=False)

if __name__ == "__main__":
    run_dashboard()
