"""
ScareX Standalone Raspberry Pi 5 - Local Backend & Web Dashboard API
Serves responsive dashboard on http://0.0.0.0:5000 for local desktop and mobile phone access.
Provides video streaming, hardware control, SQLite event logs, on-demand image analysis,
and direct local PDF reporting without requiring a laptop AI hub.
"""
import os
import sys
import time
import socket
import json
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
import psutil

try:
    import raspberry_pi.config as config
    from raspberry_pi.camera import camera_pipeline
    from raspberry_pi.hardware import hardware
    from raspberry_pi.audio import audio_player
    from raspberry_pi.database.db import db
    from raspberry_pi.fusion.decision_engine import decision_engine
    from raspberry_pi.reporting.pdf_generator import pdf_generator
except ImportError:
    try:
        import laptop.config as config
        from laptop.hardware import hardware
        from laptop.audio import audio_player
        from laptop.database.db import db
        from laptop.fusion.decision_engine import decision_engine
        from laptop.reporting.pdf_generator import pdf_generator
    except ImportError:
        import config

app = Flask(__name__, template_folder="templates")

# Shared state between main inference orchestrator and dashboard
shared_state = {
    "annotated_frame": None,
    "system_active": True,
    "overlay_mode": "both",
    "camera_client": camera_pipeline,
    "inference_fps": 0.0,
    "last_inference_latency_ms": 0.0
}

def set_camera_client(cam):
    shared_state["camera_client"] = cam

def update_annotated_frame(frame, fps=0.0, latency_ms=0.0):
    shared_state["annotated_frame"] = frame
    if fps > 0:
        shared_state["inference_fps"] = round(fps, 1)
    if latency_ms > 0:
        shared_state["last_inference_latency_ms"] = round(latency_ms, 1)

def generate_video_stream():
    """
    Generator yielding annotated live frames from local camera pipeline.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), getattr(config, "JPEG_QUALITY", 75)]
    while True:
        frame = shared_state["annotated_frame"]
        if frame is None and shared_state["camera_client"]:
            success, raw = shared_state["camera_client"].read()
            if success:
                frame = raw

        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame, encode_param)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.04)  # ~25 FPS

def get_local_ip():
    """
    Retrieves the actual LAN/Wi-Fi IP address of the Raspberry Pi.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_system_status():
    fusion_status = decision_engine.get_status()
    cam_status = shared_state["camera_client"].get_status() if shared_state["camera_client"] else {"connected": False}
    audio_stat = audio_player.get_status()

    # System resource metrics on Pi 5
    cpu_pct = psutil.cpu_percent(interval=None)
    ram_pct = psutil.virtual_memory().percent

    return jsonify({
        "system_active": shared_state["system_active"],
        "overlay_mode": shared_state["overlay_mode"],
        "camera": cam_status,
        "hardware": {
            "is_mock": hardware.is_mock,
            "distance_meters": hardware.get_distance(),
            "audio_playing": audio_stat.get("playing", False),
            "current_sound": audio_stat.get("current_sound", "None")
        },
        "bird": fusion_status["bird"],
        "tomato": fusion_status["tomato"],
        "metrics": fusion_status["system"],
        "system_metrics": {
            "cpu_pct": cpu_pct,
            "ram_pct": ram_pct,
            "inference_fps": shared_state["inference_fps"],
            "latency_ms": shared_state["last_inference_latency_ms"]
        }
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
    decision_engine.set_system_enabled(True)
    db.log_event("SYSTEM_START", robot_action="LOCAL_MONITORING_RESUMED")
    return jsonify({"success": True, "system_active": True})

@app.route('/api/control/stop', methods=['POST'])
def stop_system():
    shared_state["system_active"] = False
    decision_engine.set_system_enabled(False)
    db.log_event("SYSTEM_STOP", robot_action="LOCAL_MONITORING_PAUSED")
    return jsonify({"success": True, "system_active": False})

@app.route('/api/control/emergency_stop', methods=['POST'])
def emergency_stop():
    decision_engine.emergency_stop()
    shared_state["system_active"] = False
    return jsonify({"success": True, "message": "All actuators halted immediately on Raspberry Pi 5."})

@app.route('/api/control/reset_emergency', methods=['POST'])
def reset_emergency():
    decision_engine.reset_emergency_stop()
    shared_state["system_active"] = True
    return jsonify({"success": True, "message": "Emergency stop cleared."})

@app.route('/api/control/servo', methods=['POST'])
def sweep_servo():
    hardware.rotate_camera_servo()
    db.log_event("MANUAL_SERVO_SWEEP", robot_action="LOCAL_CAMERA_PAN_ROTATED")
    return jsonify({"success": True, "action": "SERVO_SWEEP"})

@app.route('/api/control/move', methods=['POST'])
def move_robot():
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "stop").lower()
    if direction == "forward":
        hardware.move_forward()
    elif direction == "backward":
        hardware.move_backward()
    elif direction == "turn_left":
        hardware.turn_left()
    elif direction == "turn_right":
        hardware.turn_right()
    else:
        hardware.stop_motors()
    return jsonify({"success": True, "action": f"MOVE_{direction.upper()}"})

@app.route('/api/control/buzzer', methods=['POST'])
def test_buzzer():
    try:
        if hasattr(hardware.buzzer, 'beep'):
            hardware.buzzer.beep(on_time=0.15, off_time=0.1, n=3)
        else:
            hardware.buzzer.on()
            time.sleep(0.3)
            hardware.buzzer.off()
    except Exception as e:
        print(f"[Dashboard] Buzzer test note: {e}")
    db.log_event("MANUAL_BUZZER_TEST", robot_action="LOCAL_PIEZO_BUZZER_ALERT")
    return jsonify({"success": True, "message": "Buzzer activated on Raspberry Pi 5"})

@app.route('/api/control/deter_manual', methods=['POST'])
def trigger_manual_scare():
    audio_res = audio_player.play_deterrent("default", loop=False)
    hardware.execute_scare_motion()
    db.log_event("MANUAL_SCARE_TRIGGER", robot_action="MANUAL_PREDATOR_DETERRENCE")
    return jsonify({"success": True, "audio": audio_res})

@app.route('/api/logs')
def get_recent_logs():
    limit = int(request.args.get("limit", 25))
    logs = db.get_recent_events(limit=limit)
    return jsonify(logs)

@app.route('/api/report/pdf')
def generate_pdf_report():
    start_date = request.args.get("start_date") or request.args.get("date")
    end_date = request.args.get("end_date")
    report_path = pdf_generator.generate_report(start_date=start_date, end_date=end_date)
    return send_file(os.path.abspath(report_path), as_attachment=True, download_name=os.path.basename(report_path))

@app.route('/api/reports')
def list_reports():
    reports = pdf_generator.list_reports()
    return jsonify({"success": True, "reports": reports})

@app.route('/api/reports/<filename>')
def download_specific_report(filename):
    safe_path = os.path.join(pdf_generator.reports_dir, os.path.basename(filename))
    if os.path.exists(safe_path):
        return send_file(os.path.abspath(safe_path), as_attachment=True, download_name=filename)
    return jsonify({"error": "Report not found"}), 404

@app.route('/api/network')
def get_network_info():
    ip = get_local_ip()
    hostname = socket.gethostname()
    port = getattr(config, "DASHBOARD_PORT", 5000)
    return jsonify({
        "local_ip": ip,
        "hostname": hostname,
        "port": port,
        "url": f"http://{ip}:{port}",
        "mdns_url": f"http://{hostname}.local:{port}"
    })

@app.route('/api/analyze/image', methods=['POST'])
def analyze_uploaded_image():
    """
    On-demand image inference endpoint running local YOLO models on Raspberry Pi.
    """
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Failed to decode image"}), 400

    try:
        from raspberry_pi.models.bird.detector import BirdDetector
        from raspberry_pi.models.tomato.detector import TomatoDetector
        b_det = BirdDetector()
        t_det = TomatoDetector()

        birds = b_det.detect(img)
        tomatoes = t_det.detect(img)
        row_stats = TomatoDetector.group_into_rows(tomatoes)

        counts = {"fully_ripened": 0, "half_ripened": 0, "green": 0}
        for t in tomatoes:
            lbl = t.get("label", "green")
            if lbl in counts:
                counts[lbl] += 1
            else:
                counts["green"] += 1

        total = sum(counts.values())
        ripe_pct = (counts["fully_ripened"] / total * 100) if total > 0 else 0
        prio = "HIGH" if ripe_pct >= 50 else ("MEDIUM" if ripe_pct >= 25 else "LOW")

        return jsonify({
            "success": True,
            "birds": birds,
            "tomatoes": tomatoes,
            "tomato_summary": counts,
            "total_tomatoes": total,
            "harvest_priority": prio,
            "row_stats": row_stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def run_dashboard(port=None):
    web_port = port or getattr(config, "DASHBOARD_PORT", 5000)
    host = getattr(config, "DASHBOARD_HOST", "0.0.0.0")
    print(f"[Dashboard] Starting ScareX Web Dashboard on http://{host}:{web_port}")
    app.run(host=host, port=web_port, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    run_dashboard()
