"""
Laptop Local Webcam Client (Development / Testing Mode)
Captures frames directly from the laptop webcam via OpenCV cv2.VideoCapture.
Provides the exact same thread-safe interface as PiCameraClient.
"""
import time
import threading
import cv2
import numpy as np

try:
    import laptop.config as config
except ImportError:
    import config

class LaptopCameraClient:
    def __init__(self, device_index=None):
        self.device_index = device_index if device_index is not None else config.WEBCAM_INDEX
        self.latest_frame = None
        self.is_connected = False
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        self.last_frame_time = 0.0
        self.cap = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[LaptopCameraClient] Local webcam ingestion started for device {self.device_index}")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected = False
        print("[LaptopCameraClient] Local webcam stopped.")

    def _capture_loop(self):
        # On Windows, DirectShow backend prevents freezing/hanging
        try:
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        except Exception:
            self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap or not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.device_index)

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.is_connected = True
            print(f"[LaptopCameraClient] Webcam opened successfully ({config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}).")
        else:
            print(f"[LaptopCameraClient] Warning: Unable to open webcam {self.device_index}.")
            self.is_connected = False

        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.last_frame_time = time.time()
                        self.is_connected = True
                else:
                    time.sleep(0.05)
            else:
                time.sleep(0.1)

        if self.cap:
            self.cap.release()
            self.cap = None

    def read(self):
        """
        Returns (success: bool, frame: np.ndarray)
        Thread-safe copy of the latest captured frame.
        """
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def get_status(self):
        now = time.time()
        active = self.is_connected and (now - self.last_frame_time < 3.0)
        return {
            "connected": active,
            "mode": "LAPTOP",
            "source": "LAPTOP WEBCAM",
            "device_index": self.device_index,
            "last_frame_age_sec": round(now - self.last_frame_time, 2) if self.last_frame_time > 0 else None
        }
