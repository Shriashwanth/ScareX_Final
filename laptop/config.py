"""
ScareX Integrated System - Laptop Configuration
"""
import os

# Base directory for the laptop subsystem
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# Camera Mode Selection
# ==============================================================================
# Possible values:
#   "LAPTOP" -> Use laptop local webcam via OpenCV cv2.VideoCapture(0)
#               Purpose: Development and testing without Raspberry Pi hardware.
#   "PI"     -> Use Raspberry Pi 5 MJPEG camera stream from network
#               Purpose: Final field deployment with Raspberry Pi 5 and Pi Camera.
CAMERA_MODE = os.getenv("SCAREX_CAMERA_MODE", "LAPTOP")

# Webcam Device ID for LAPTOP mode
WEBCAM_INDEX = int(os.getenv("SCAREX_WEBCAM_INDEX", 0))

# ==============================================================================
# Raspberry Pi Network Configuration (Used when CAMERA_MODE = "PI" or for hardware)
# ==============================================================================
# Change PI_IP to the actual IP address of your Raspberry Pi 5 on your LAN/Wi-Fi
PI_IP = os.getenv("SCAREX_PI_IP", "127.0.0.1")  # Defaults to localhost for local testing
CAMERA_PORT = int(os.getenv("SCAREX_CAMERA_PORT", 8000))
COMMAND_PORT = int(os.getenv("SCAREX_COMMAND_PORT", 9000))
DASHBOARD_PORT = int(os.getenv("SCAREX_DASHBOARD_PORT", 5000))

# Video Stream URL from Pi Camera Server (used in PI mode)
CAMERA_STREAM_URL = f"http://{PI_IP}:{CAMERA_PORT}/video"
COMMAND_SERVER_URL = f"http://{PI_IP}:{COMMAND_PORT}/command"
STATUS_SERVER_URL = f"http://{PI_IP}:{COMMAND_PORT}/status"

# Camera Dimensions & Quality
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
JPEG_QUALITY = 80

# AI Model Paths (NCNN High-Speed Embedded Runtime, with PyTorch .pt fallback)
BIRD_NCNN_PATH = os.path.join(BASE_DIR, "models", "bird", "best_ncnn_model")
TOMATO_NCNN_PATH = os.path.join(BASE_DIR, "models", "tomato", "best_ncnn_model")

BIRD_MODEL_PATH = os.getenv("SCAREX_BIRD_MODEL", BIRD_NCNN_PATH if os.path.exists(BIRD_NCNN_PATH) else os.path.join(BASE_DIR, "models", "bird", "best.pt"))
TOMATO_MODEL_PATH = os.getenv("SCAREX_TOMATO_MODEL", TOMATO_NCNN_PATH if os.path.exists(TOMATO_NCNN_PATH) else os.path.join(BASE_DIR, "models", "tomato", "best.pt"))

# Confidence & Thresholds
BIRD_CONFIDENCE = float(os.getenv("SCAREX_BIRD_CONF", 0.45))
TOMATO_CONFIDENCE = float(os.getenv("SCAREX_TOMATO_CONF", 0.70))
IOU_THRESHOLD = 0.45

# Decision & Fusion Engine Settings
BIRD_CONFIRMATION_FRAMES = int(os.getenv("SCAREX_BIRD_CONFIRM_FRAMES", 3))
BIRD_COOLDOWN_SECONDS = float(os.getenv("SCAREX_BIRD_COOLDOWN", 10.0))
HEARTBEAT_INTERVAL_SECONDS = 1.0

# Database
DB_PATH = os.path.join(BASE_DIR, "database", "scarex_integrated.db")
