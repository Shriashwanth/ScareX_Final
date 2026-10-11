"""
ScareX Standalone Raspberry Pi 5 - Local Camera Capture Pipeline
Supports:
  1. Native Raspberry Pi Camera Module via Picamera2 (CSI ribbon cable on Pi 5)
  2. Standard USB Webcams via V4L2 / cv2.VideoCapture
  3. Synthetic test generator fallback for headless testing and development

Captures frames directly into local memory and shares atomic frame
between Bird and Tomato YOLO detectors without any network hops.
Includes auto-reconnection and fault tolerance.
"""
import time
import threading
import cv2
import numpy as np

try:
    import raspberry_pi.config as config
except ImportError:
    try:
        import laptop.config as config
    except ImportError:
        import config

class CameraPipeline:
    def __init__(self):
        self.width = getattr(config, "CAMERA_WIDTH", 640)
        self.height = getattr(config, "CAMERA_HEIGHT", 480)
        self.fps = getattr(config, "CAMERA_FPS", 30)
        self.source_pref = getattr(config, "CAMERA_SOURCE", "AUTO").upper()
        self.usb_index = getattr(config, "USB_CAM_INDEX", 0)
        self.jpeg_quality = getattr(config, "JPEG_QUALITY", 80)

        self.latest_frame = self._generate_synthetic_frame()
        self.frame_count = 0
        self.lock = threading.Lock()
        self.is_running = False
        self.thread = None

        self.picam2 = None
        self.cap = None
        self.active_source = "SYNTHETIC"
        self.connected = False
        self.last_frame_time = 0.0

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        print(f"[CameraPipeline] Started local camera capture worker (Preference: {self.source_pref}).")

    def _init_hardware_camera(self) -> bool:
        """
        Attempts to open native Picamera2 or USB camera based on availability and config.
        """
        self._release_cameras()

        # 1. Attempt Picamera2 if requested or AUTO
        if self.source_pref in ["AUTO", "PICAMERA2"]:
            try:
                try:
                    from picamera2 import Picamera2
                except ImportError:
                    from picam2 import Picamera2
                print("[CameraPipeline] Initializing Picamera2 (Pi 5 CSI interface)...")
                p = Picamera2()
                camera_config = p.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                p.configure(camera_config)
                p.start()
                self.picam2 = p
                self.active_source = "PICAMERA2"
                self.connected = True
                print("[CameraPipeline] Picamera2 successfully initialized.")
                return True
            except Exception as e:
                print(f"[CameraPipeline] Picamera2 unavailable ({e}). Checking USB webcam...")

        # 2. Attempt USB Camera
        if self.source_pref in ["AUTO", "USB"]:
            indices_to_try = [self.usb_index] + [i for i in [0, 1, 2, 4] if i != self.usb_index]
            for idx in indices_to_try:
                try:
                    # On Linux/Pi, CAP_V4L2 provides fast, stable camera binding
                    backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
                    c = cv2.VideoCapture(idx, backend)
                    if not c.isOpened():
                        c = cv2.VideoCapture(idx)

                    if c.isOpened():
                        try:
                            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        except Exception:
                            pass
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        c.set(cv2.CAP_PROP_FPS, self.fps)

                        ret, test_frame = c.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            self.cap = c
                            self.active_source = f"USB_CAM_{idx}"
                            self.connected = True
                            print(f"[CameraPipeline] USB Camera successfully bound on device {idx}.")
                            return True
                        else:
                            c.release()
                except Exception:
                    pass

        # 3. Fallback to synthetic
        print("[CameraPipeline] No physical camera found. Using internal synthetic generator.")
        self.active_source = "SYNTHETIC"
        self.connected = True
        return False

    def _release_cameras(self):
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Creates a dynamic 640x480 test frame with timestamps and simulated field environment.
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background field gradient (earth tones)
        frame[:, :] = (35, 60, 45)

        # Moving scan line / indicator
        t = time.time()
        cx = int((self.width / 2) + 180 * np.sin(t * 1.5))
        cy = int((self.height / 2) + 90 * np.cos(t * 1.5))
        cv2.circle(frame, (cx, cy), 18, (0, 200, 255), -1)

        # Draw status overlay
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"SCAREX PI 5 LOCAL CAMERA (SYNTHETIC) - {ts_str}",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "Connect Pi Camera (CSI) or USB Webcam for Live Feed",
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
        return frame

    def _capture_worker(self):
        self._init_hardware_camera()

        while self.is_running:
            frame = None
            now = time.time()

            # Read from Picamera2
            if self.picam2:
                try:
                    frame_rgb = self.picam2.capture_array()
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"[CameraPipeline] Picamera2 read failure: {e}. Attempting reconnect...")
                    time.sleep(1.0)
                    self._init_hardware_camera()

            # Read from OpenCV USB VideoCapture
            elif self.cap and self.cap.isOpened():
                try:
                    ret, raw = self.cap.read()
                    if ret and raw is not None:
                        frame = raw
                    else:
                        print("[CameraPipeline] USB camera read dropped frame. Reconnecting...")
                        time.sleep(0.5)
                        self._init_hardware_camera()
                except Exception as e:
                    print(f"[CameraPipeline] USB camera exception: {e}")
                    time.sleep(1.0)
                    self._init_hardware_camera()

            # Fallback to synthetic
            if frame is None:
                frame = self._generate_synthetic_frame()
                time.sleep(1.0 / max(10, self.fps))

            with self.lock:
                self.latest_frame = frame
                self.frame_count += 1
                self.last_frame_time = now

            time.sleep(0.005)

    def read(self) -> tuple:
        """
        Thread-safe extraction of the latest frame directly in memory.
        Returns: (success: bool, frame: np.ndarray)
        """
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return True, self._generate_synthetic_frame()

    def generate_mjpeg(self, quality=None):
        """
        Generator yielding multipart JPEG frames for the local web dashboard.
        """
        q = quality or self.jpeg_quality
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        while self.is_running:
            ret, frame = self.read()
            if ret and frame is not None:
                success, buffer = cv2.imencode('.jpg', frame, encode_params)
                if success:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)  # ~25 FPS delivery for web clients

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._release_cameras()
        self.connected = False
        print("[CameraPipeline] Camera pipeline stopped cleanly.")

    def get_status(self) -> dict:
        return {
            "connected": self.connected,
            "source": self.active_source,
            "fps_configured": self.fps,
            "width": self.width,
            "height": self.height,
            "frames_captured": self.frame_count,
            "last_frame_timestamp": round(self.last_frame_time, 2)
        }

# Global singleton
camera_pipeline = CameraPipeline()
