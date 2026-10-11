"""
Raspberry Pi Camera Server
Provides high-performance MJPEG streaming on http://0.0.0.0:8000/video.
Designed for Raspberry Pi 5 with Picamera2 (with cv2.VideoCapture fallback for PC testing).
"""
import time
import threading
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np
import raspberry_pi.config as config

# Global thread-safe frame buffer
latest_jpeg = None
frame_lock = threading.Lock()
running = True

def capture_loop():
    global latest_jpeg, running
    picam2 = None
    cap = None
    use_picam2 = False

    # Attempt Picamera2 initialization
    try:
        from picam2 import Picamera2
        print("[CameraServer] Attempting Picamera2 initialization...")
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"}
        )
        picam2.configure(camera_config)
        picam2.start()
        use_picam2 = True
        print("[CameraServer] Picamera2 initialized successfully.")
    except Exception as e:
        print(f"[CameraServer] Picamera2 unavailable ({e}). Falling back to cv2.VideoCapture / Synthetic...")
        use_picam2 = False

    if not use_picam2:
        # Probe USB Cameras (V4L2 on Raspberry Pi 5 Linux, standard on Windows)
        target_idx = getattr(config, "USB_CAM_INDEX", 0)
        indices_to_try = [target_idx] + [i for i in [0, 1, 2, 4] if i != target_idx]
        
        for idx in indices_to_try:
            try:
                # On Linux/Pi, CAP_V4L2 provides fast, zero-hang camera binding
                c = cv2.VideoCapture(idx, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(idx)
                if not c.isOpened():
                    c = cv2.VideoCapture(idx)
                
                if c.isOpened():
                    # Set MJPG codec for high frame rate over USB 2.0/3.0 bus
                    try:
                        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    except Exception:
                        pass
                    c.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                    c.set(cv2.CAP_PROP_FPS, getattr(config, "CAMERA_FPS", 30))
                    
                    # Verify first frame read
                    ret_test, frame_test = c.read()
                    if ret_test and frame_test is not None:
                        cap = c
                        print(f"[CameraServer] USB Camera opened successfully on device index {idx} ({config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} @ 30 FPS).")
                        break
                    else:
                        c.release()
            except Exception as cam_err:
                pass

        if cap is None:
            print("[CameraServer] No physical USB camera detected. Using synthetic test generator.")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]
    frame_idx = 0

    while running:
        frame = None
        if use_picam2 and picam2:
            try:
                # Picamera2 returns RGB888 numpy array
                frame_rgb = picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[CameraServer] Picamera2 capture error: {e}")
                time.sleep(0.1)

        elif cap and cap.isOpened():
            ret, captured = cap.read()
            if ret:
                frame = captured
            else:
                time.sleep(0.05)

        if frame is None:
            # Generate synthetic test frame (moving pattern with timestamp)
            frame_idx += 1
            frame = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
            # Simulated sky and crop field
            frame[:240, :] = [200, 160, 100]  # Light blue/grey sky
            frame[240:, :] = [40, 140, 40]    # Green field
            
            # Draw moving mock tomato
            cx = int(200 + 150 * np.sin(frame_idx * 0.05))
            cy = int(340 + 30 * np.cos(frame_idx * 0.05))
            cv2.circle(frame, (cx, cy), 35, (0, 0, 220), -1)  # Red tomato
            cv2.putText(frame, "Mock Tomato", (cx - 40, cy - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw time and watermarks
            cv2.putText(frame, "Pi Camera Server (Synthetic Mode)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

        # Compress to JPEG
        ret, jpeg_buf = cv2.imencode('.jpg', frame, encode_param)
        if ret:
            with frame_lock:
                latest_jpeg = jpeg_buf.tobytes()

        # Enforce FPS limit to prevent saturating CPU/network
        time.sleep(1.0 / config.CAMERA_FPS)

    # Cleanup
    if picam2:
        try:
            picam2.stop()
        except:
            pass
    if cap:
        cap.release()

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/video', '/video_feed', '/stream.mjpg']:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()

            while running:
                with frame_lock:
                    frame_bytes = latest_jpeg

                if frame_bytes is not None:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b'\r\n')
                    except (ConnectionResetError, BrokenPipeError):
                        break
                time.sleep(0.03)  # ~30 fps delivery
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP access logs
        return

def start_camera_server(port=None):
    server_port = port or config.CAMERA_PORT
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    server = HTTPServer(('0.0.0.0', server_port), StreamHandler)
    print(f"[CameraServer] Streaming live Pi Camera at http://0.0.0.0:{server_port}/video")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        global running
        running = False
        server.server_close()

if __name__ == "__main__":
    start_camera_server()
