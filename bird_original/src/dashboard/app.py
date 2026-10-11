import os
import time
import cv2
import json
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file
from src.core.logger import logger
from src.logging.db_logger import db_logger
from src.core.config_manager import ConfigManager
from src.vision.tomato_analyzer import TomatoAnalyzer
from fpdf import FPDF

app = Flask(__name__)
config = ConfigManager()

analyzer = TomatoAnalyzer()
video_source = "dataset/videos/tomato_farm.mp4"

def generate_video_stream():
    while True:
        if analyzer.running:
            frame = analyzer.get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                save_session()
                time.sleep(1)
        else:
            time.sleep(0.1)

def save_session():
    duration = time.time() - analyzer.start_time
    total = sum(analyzer.session_stats.values())
    
    full_pct = (analyzer.session_stats['fully_ripened'] / total * 100) if total > 0 else 0
    priority = "LOW"
    if full_pct >= 50:
        priority = "HIGH"
    elif full_pct >= 25:
        priority = "MEDIUM"
        
    if total > 0:
        db_logger.log_event(
            event_type="MONITORING_SESSION_END",
            species="Tomato",
            confidence=1.0,
            alarm_played="None",
            system_status=f"Duration: {duration:.1f}s, Row Data: {json.dumps(analyzer.best_row_stats)}",
            detection_source="DASHBOARD",
            duration=duration,
            green_count=analyzer.session_stats['green'],
            half_ripened_count=analyzer.session_stats['half_ripened'],
            fully_ripened_count=analyzer.session_stats['fully_ripened'],
            harvest_priority=priority
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_webcam', methods=['POST'])
def start_webcam():
    success = analyzer.start(0)
    return jsonify({"success": success})

@app.route('/api/start_video', methods=['POST'])
def start_video():
    success = analyzer.start(video_source)
    return jsonify({"success": success})

@app.route('/api/stop', methods=['POST'])
def stop_analysis():
    analyzer.stop()
    save_session()
    return jsonify({"success": True})

@app.route('/api/status')
def system_status():
    stats = analyzer.session_stats
    total = sum(stats.values())
    
    green_pct = (stats['green'] / total * 100) if total > 0 else 0
    half_pct = (stats['half_ripened'] / total * 100) if total > 0 else 0
    full_pct = (stats['fully_ripened'] / total * 100) if total > 0 else 0
    
    priority = "LOW"
    if full_pct >= 50:
        priority = "HIGH"
    elif full_pct >= 25:
        priority = "MEDIUM"
        
    duration = time.time() - analyzer.start_time if analyzer.running else 0
        
    return jsonify({
        "running": analyzer.running,
        "total": total,
        "green": stats['green'],
        "half_ripened": stats['half_ripened'],
        "fully_ripened": stats['fully_ripened'],
        "green_pct": round(green_pct, 1),
        "half_pct": round(half_pct, 1),
        "full_pct": round(full_pct, 1),
        "priority": priority,
        "duration": round(duration, 1),
        "row_stats": analyzer.best_row_stats
    })

@app.route('/api/report/pdf')
def generate_pdf():
    stats = analyzer.session_stats
    total = sum(stats.values())
    
    green_pct = (stats['green'] / total * 100) if total > 0 else 0
    half_pct = (stats['half_ripened'] / total * 100) if total > 0 else 0
    full_pct = (stats['fully_ripened'] / total * 100) if total > 0 else 0
    
    priority = "LOW"
    if full_pct >= 50:
        priority = "HIGH"
    elif full_pct >= 25:
        priority = "MEDIUM"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="SCareX", ln=True, align='C')
    pdf.cell(200, 10, txt="AI Tomato Crop Monitoring Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt="Input source: Video/Webcam", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Overall Maturity:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Green | {stats['green']} | {green_pct:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Half-ripened | {stats['half_ripened']} | {half_pct:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Fully-ripened | {stats['fully_ripened']} | {full_pct:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Total | {total} | 100%", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Harvesting Priority: {priority}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Row-wise Analysis:", ln=True)
    pdf.set_font("Arial", size=10)
    for row in analyzer.best_row_stats:
        row_txt = f"Row {row['row_number']} | Total: {row['total']} | Green: {row['green_pct']}% | Half: {row['half_pct']}% | Fully: {row['full_pct']}% | Priority: {row['priority']}"
        pdf.cell(200, 8, txt=row_txt, ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="Observation: Percentages are calculated from tomatoes detected within the camera view. Hidden, occluded, or out-of-frame tomatoes may not be detected.")
    pdf.multi_cell(0, 10, txt="Disclaimer: Harvesting priority is a prototype decision-support indicator based on detected maturity distribution and is not a guaranteed agronomic prediction.")
    
    pdf_path = "scarex_report.pdf"
    pdf.output(pdf_path)
    return send_file(os.path.abspath(pdf_path), as_attachment=True)

import threading
def start_flask_app(fusion, vision, audio):
    port = config.get("flask_port", 5000)
    logger.info(f"Starting Flask Dashboard on port {port}...")
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False), daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
