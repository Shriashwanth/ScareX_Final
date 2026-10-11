"""
ScareX Standalone Raspberry Pi 5 Configuration
Centralizes camera, AI inference models, decision engine, GPIO hardware,
SQLite database, audio deterrence, and web dashboard settings for edge autonomy.
"""
import os

# Root base directory for Raspberry Pi 5 subsystem
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# ==============================================================================
# Camera Pipeline Configuration (Raspberry Pi 5 Local Capture)
# ==============================================================================
# "AUTO" -> Probes native Picamera2 (CSI ribbon), then USB Webcam, then Synthetic fallback
# "PICAMERA2" -> Force Raspberry Pi Camera Module (libcamera / Picamera2)
# "USB" -> Force V4L2 USB camera (OpenCV)
# "SYNTHETIC" -> Test pattern generator for headless/dry-run mode
CAMERA_SOURCE = os.getenv("SCAREX_CAMERA_SOURCE", "AUTO").upper()
CAMERA_WIDTH = int(os.getenv("SCAREX_CAM_WIDTH", 640))
CAMERA_HEIGHT = int(os.getenv("SCAREX_CAM_HEIGHT", 480))
CAMERA_FPS = int(os.getenv("SCAREX_CAM_FPS", 30))
JPEG_QUALITY = int(os.getenv("SCAREX_JPEG_QUALITY", 80))
USB_CAM_INDEX = int(os.getenv("SCAREX_USB_CAM_INDEX", 0))

# Web Dashboard & Local Network Ports
DASHBOARD_PORT = int(os.getenv("SCAREX_DASHBOARD_PORT", 5000))
DASHBOARD_HOST = os.getenv("SCAREX_DASHBOARD_HOST", "0.0.0.0")

# Backward Compatibility Ports (for legacy clients / tests)
CAMERA_PORT = int(os.getenv("SCAREX_PI_CAMERA_PORT", 8000))
COMMAND_PORT = int(os.getenv("SCAREX_PI_COMMAND_PORT", 9000))

# ==============================================================================
# AI Inference Model Paths (NCNN ARM64 Optimized with PyTorch Fallback)
# ==============================================================================
MODELS_DIR = os.path.join(BASE_DIR, "models")
# Fallback to laptop/models if running in dev environment where models were stored
if not os.path.exists(MODELS_DIR) or not os.path.exists(os.path.join(MODELS_DIR, "bird")):
    ALT_MODELS_DIR = os.path.join(PROJECT_ROOT, "laptop", "models")
    if os.path.exists(ALT_MODELS_DIR):
        MODELS_DIR = ALT_MODELS_DIR

BIRD_NCNN_PATH = os.path.join(MODELS_DIR, "bird", "best_ncnn_model")
BIRD_PT_PATH = os.path.join(MODELS_DIR, "bird", "best.pt")
TOMATO_NCNN_PATH = os.path.join(MODELS_DIR, "tomato", "best_ncnn_model")
TOMATO_PT_PATH = os.path.join(MODELS_DIR, "tomato", "best.pt")
BIRD_AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, "bird_audio_model.joblib")

# Prefer NCNN for ARM64 CPU inference on Pi 5; automatically fall back to PyTorch .pt
BIRD_MODEL_PATH = os.getenv(
    "SCAREX_BIRD_MODEL",
    BIRD_NCNN_PATH if os.path.exists(BIRD_NCNN_PATH) else BIRD_PT_PATH
)
TOMATO_MODEL_PATH = os.getenv(
    "SCAREX_TOMATO_MODEL",
    TOMATO_NCNN_PATH if os.path.exists(TOMATO_NCNN_PATH) else TOMATO_PT_PATH
)

# Confidence & Discrimination Thresholds
BIRD_CONFIDENCE = float(os.getenv("SCAREX_BIRD_CONF", 0.45))
TOMATO_CONFIDENCE = float(os.getenv("SCAREX_TOMATO_CONF", 0.70))
IOU_THRESHOLD = 0.45

# Exact Tomato Maturity Class Mapping (Preserved from trained YOLO11 dataset)
TOMATO_CLASSES = {
    0: "b_fully_ripened",
    1: "b_half_ripened",
    2: "b_green"
}

# ==============================================================================
# Acoustic Bird Classifier Settings
# ==============================================================================
AUDIO_CLASSIFIER_SAMPLE_RATE = 22050
AUDIO_CLASSIFIER_BUFFER_SEC = 1.2
ENABLE_AUDIO_CLASSIFIER = os.getenv("SCAREX_ENABLE_AUDIO_CLASSIFIER", "true").lower() == "true"

# ==============================================================================
# Unified Decision & Fusion Engine Settings
# ==============================================================================
BIRD_CONFIRMATION_FRAMES = int(os.getenv("SCAREX_BIRD_CONFIRM_FRAMES", 2))
BIRD_COOLDOWN_SECONDS = float(os.getenv("SCAREX_BIRD_COOLDOWN", 10.0))
WATCHDOG_TIMEOUT = 3.0  # seconds

# ==============================================================================
# Hardware GPIO Pin Configuration (BCM numbering for Raspberry Pi 5)
# ==============================================================================
GPIO_PINS = {
    # DC Motors (H-Bridge L298N / TB6612)
    "motor_left_forward": 17,
    "motor_left_backward": 27,
    "motor_right_forward": 22,
    "motor_right_backward": 23,
    
    # Ultrasonic Distance Sensor (HC-SR04)
    "ultrasonic_trigger": 24,
    "ultrasonic_echo": 25,
    
    # Pan/Tilt or Camera Servo
    "servo_camera": 18,
    
    # Status Indicator LEDs & Buzzer
    "led_status": 5,
    "led_scare": 6,
    "buzzer": 12
}

# ==============================================================================
# Audio Deterrence Player Settings
# ==============================================================================
AUDIO_DIR = os.path.join(BASE_DIR, "assets", "audio")
DETERRENT_DIR = os.path.join(BASE_DIR, "assets", "deterrent_sounds")
if not os.path.exists(AUDIO_DIR) and os.path.exists(os.path.join(PROJECT_ROOT, "laptop", "assets")):
    AUDIO_DIR = os.path.join(PROJECT_ROOT, "laptop", "assets", "deterrent_sounds")

SPEAKER_VOLUME = 1.0  # 0.0 to 1.0
AUDIO_COOLDOWN_SECONDS = 5.0

# 5 Predator Bioacoustic Deterrence Sounds
DETERRENT_SOUNDS = [
    {"id": 1, "name": "Monster Roar", "file": "deterrent_1.wav", "orig": "audiopapkin-monster-roar-ps-oo6-306402.mp3"},
    {"id": 2, "name": "Monster Warrior Roar", "file": "deterrent_2.wav", "orig": "dffdv-monster-warrior-roar-195877.mp3"},
    {"id": 3, "name": "Epic Dragon Roar", "file": "deterrent_3.wav", "orig": "dragon-studio-epic-dragon-roar-364481.mp3"},
    {"id": 4, "name": "Spooky Wolf Howl", "file": "deterrent_4.wav", "orig": "dragon-studio-spooky-wolf-howl-410547.mp3"},
    {"id": 5, "name": "Beast Screaming", "file": "deterrent_5.wav", "orig": "freesound_community-beast-human-screaming-86831.mp3"}
]

# Species specific mapping + dynamic alternating mode
SPECIES_AUDIO_MAP = {
    "crow": "deterrent_3.wav",                 # Epic Dragon Roar
    "pigeon": "deterrent_4.wav",               # Wolf Howl
    "sparrow": "deterrent_1.wav",              # Monster Roar
    "common myna": "deterrent_2.wav",          # Warrior Roar
    "myna": "deterrent_2.wav",
    "rose ringed parakeet": "deterrent_5.wav", # Beast Scream
    "parakeet": "deterrent_5.wav",
    "parrot": "deterrent_5.wav",
    "peacock": "deterrent_2.wav",              # Warrior Roar
    "bird": "alternate",                       # Alternate across the 5 sounds
    "default": "alternate"
}

# ==============================================================================
# Persistent Storage & Reporting
# ==============================================================================
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DATABASE_DIR, "scarex_local.db")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
