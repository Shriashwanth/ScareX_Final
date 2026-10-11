"""
Pi Camera Network Stream Client
Connects to the single Raspberry Pi camera stream at http://PI_IP:8000/video.
Consumes the stream in a background thread to always maintain the latest frame without lag.
Provides a unified read() method returning (ret, frame) for both models.
"""
import time
import threading
import urllib.request
import cv2
import numpy as np

try:
    import laptop.config as config
except ImportError:
    import config

class PiCameraClient:
    def __init__(self, stream_url=None):
        self.stream_url = stream_url or config.CAMERA_STREAM_URL
        self.latest_frame = None
        self.is_connected = False
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        self.last_frame_time = 0.0

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._ingestion_loop, daemon=True)
        self.thread.start()
        print(f"[PiCameraClient] Ingestion client started for: {self.stream_url}")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.is_connected = False
        print("[PiCameraClient] Client stopped.")

    def _ingestion_loop(self):
        """
        Connects to the MJPEG HTTP stream and continuously extracts JPEG frames.
        """
        while self.is_running:
            try:
                print(f"[PiCameraClient] Connecting to stream at {self.stream_url}...")
                req = urllib.request.Request(self.stream_url, headers={"User-Agent": "ScareX-Laptop-Client"})
                with urllib.request.urlopen(req, timeout=5.0) as stream:
                    self.is_connected = True
                    print("[PiCameraClient] Connected to Pi camera stream!")
                    
                    stream_bytes = b""
                    while self.is_running:
                        chunk = stream.read(4096)
                        if not chunk:
                            break
                        stream_bytes += chunk
                        
                        a = stream_bytes.find(b'\xff\xd8')  # JPEG start marker
                        b = stream_bytes.find(b'\xff\xd9')  # JPEG end marker
                        
                        if a != -1 and b != -1:
                            if b > a:
                                jpg_bytes = stream_bytes[a:b+2]
                                stream_bytes = stream_bytes[b+2:]
                                
                                # Decode JPEG to BGR OpenCV image
                                frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if frame is not None:
                                    with self.lock:
                                        self.latest_frame = frame
                                        self.last_frame_time = time.time()
                            else:
                                # Out of order marker recovery
                                stream_bytes = stream_bytes[a:]

            except Exception as e:
                self.is_connected = False
                # Connection dropped or server not yet started; retry after short backoff
                time.sleep(1.5)

    def read(self):
        """
        Returns (success: bool, frame: np.ndarray)
        Safe atomic copy of the single shared frame.
        """
        with self.lock:
            if self.latest_frame is None:
                return False, None
            # Return copy to guarantee thread safety across multiple models
            return True, self.latest_frame.copy()

    def get_status(self):
        now = time.time()
        # If no frame received within the last 3 seconds, consider disconnected
        active = self.is_connected and (now - self.last_frame_time < 3.0)
        return {
            "connected": active,
            "mode": "PI",
            "source": "PI CAMERA (STREAM)",
            "stream_url": self.stream_url,
            "last_frame_age_sec": round(now - self.last_frame_time, 2) if self.last_frame_time > 0 else None
        }
