# ScareX Integrated System: Unified Architecture & Deployment Guide

Welcome to the **ScareX Integrated Autonomous Surveillance System**. This project integrates multi-target AI computer vision (Bird Detection and Tomato Crop Maturity Analytics) with edge robotic actuation across a distributed network between a **Laptop** and a **Raspberry Pi 5 (8 GB RAM)**.

---

## 1. System Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               RASPBERRY PI 5 (Edge Hardware)            │
                    │                                                         │
                    │   Pi Camera (CSI)                                       │
                    │         │                                               │
                    │         ▼                                               │
                    │   camera_server.py (Picamera2 / MJPEG :8000/video)      │
                    │                                                         │
                    │   command_server.py (Flask REST API :9000/command)      │
                    │   ├── audio.py (Local speaker playback via pygame)      │
                    │   ├── hardware.py (gpiozero motor/servo controller)     │
                    │   └── Watchdog Safety (Halts motors on comms loss)      │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                                         LAN / Wi-Fi Network
                                                 │
                    ┌────────────────────────────▼────────────────────────────┐
                    │                     LAPTOP (AI Hub)                     │
                    │                                                         │
                    │   pi_camera_client.py (Single MJPEG Stream Ingestion)   │
                    │         │                                               │
                    │         ▼ (Single Atomic Shared Frame)                  │
                    │   ┌───────────────────────────┬─────────────────────┐   │
                    │   ▼                           ▼                     │   │
                    │   BirdDetector                TomatoDetector        │   │
                    │   (YOLO11 5-class)            (YOLO11 3-class)      │   │
                    │   [best.pt]                   [best.pt]             │   │
                    │   └─────────────┬─────────────┴─────────────────────┘   │
                    │                 ▼                                       │
                    │   fusion/decision_engine.py                             │
                    │   ├── Consecutive bird confirmation (e.g. 3 frames)     │
                    │   ├── Deterrence cooldown (10s)                         │
                    │   ├── Tomato maturity ratio & row-wise clustering       │
                    │   ├── database/db.py (Unified SQLite logging)           │
                    │   └── dashboard/app.py (Flask Dashboard on :5000)       │
                    │                 │                                       │
                    │                 ▼                                       │
                    │   communication/pi_client.py (Dispatches commands)      │
                    └─────────────────────────────────────────────────────────┘
```

---

## 2. Laptop Setup

The laptop handles all heavy YOLO neural network inference and hosts the unified monitoring dashboard.

1. **Prerequisites:**
   - Python 3.9+ (Python 3.10, 3.11, or 3.12 recommended)
   - CUDA-enabled GPU (optional, runs in CPU mode automatically if no GPU is present)

2. **Installation:**
   ```bash
   cd ScareX_Integrated
   pip install -r laptop/requirements.txt
   ```

---

## 3. Raspberry Pi 5 Setup

The Raspberry Pi 5 is strictly a lightweight edge capture and hardware actuation controller. **It does not run any YOLO models.**

1. **Prerequisites:**
   - Raspberry Pi OS (Bookworm, 64-bit)
   - Raspberry Pi Camera Module connected via CSI ribbon cable
   - Speaker connected via 3.5mm jack or USB / I2S amplifier
   - Motor driver (e.g. L298N) and pan/tilt servo connected to GPIO

2. **Installation on the Pi:**
   ```bash
   # Copy the raspberry_pi/ folder to your Pi:
   scp -r raspberry_pi/ pi@<PI_IP>:~/ScareX_Integrated/
   
   # SSH into the Pi:
   ssh pi@<PI_IP>
   cd ~/ScareX_Integrated
   
   # Install edge dependencies:
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-picamera2 python3-gpiozero python3-pygame
   pip install -r raspberry_pi/requirements.txt --break-system-packages
   ```

---

## 4. Python Dependencies

### Laptop (`laptop/requirements.txt`)
- `ultralytics>=8.3.0` (YOLO11 model runtime)
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `opencv-python>=4.8.0`
- `Flask>=3.0.0`
- `requests>=2.28.0`
- `fpdf>=1.7.2` (PDF reporting)
- `numpy>=1.24.0`

### Raspberry Pi (`raspberry_pi/requirements.txt`)
- `Flask>=3.0.0`
- `opencv-python-headless>=4.8.0`
- `numpy>=1.24.0`
- `pygame>=2.5.0` (Audio deterrent player)
- `gpiozero>=2.0` (Actuator and sensor abstraction)

---

## 5. Camera Setup

The system uses **one single Pi Camera**.
- **On the Pi:** `raspberry_pi/camera_server.py` opens the Pi Camera via native `Picamera2` at 640x480 @ 30 FPS and streams MJPEG on `http://0.0.0.0:8000/video`.
- **On the Laptop:** `laptop/camera/pi_camera_client.py` connects to this single URL, decodes the latest JPEG frame, and shares that exact frame in memory with both models simultaneously.

---

## 6. Model Weights & Paths

Both model weights are stored inside `laptop/models/`:

1. **Bird Model:**
   - Path: `laptop/models/bird/best.pt`
   - Architecture: Ultralytics YOLO11n fine-tuned
   - Classes (5): `0: 'Crow'`, `1: 'Common Myna'`, `2: 'Rose Ringed Parakeet'`, `3: 'Peacock'`, `4: 'Pigeon'`
2. **Tomato Model:**
   - Path: `laptop/models/tomato/best.pt`
   - Architecture: Ultralytics YOLO11n fine-tuned
   - Classes (3): `0: 'fully_ripened'`, `1: 'green'`, `2: 'half_ripened'`

---

## 7. Configuration

All parameters are cleanly centralized in two files:

### `laptop/config.py`
| Variable | Default | Purpose |
|---|---|---|
| `PI_IP` | `"127.0.0.1"` | IP address of Raspberry Pi on LAN/Wi-Fi |
| `CAMERA_PORT` | `8000` | Port for Pi camera MJPEG stream |
| `COMMAND_PORT` | `9000` | Port for Pi REST command server |
| `DASHBOARD_PORT` | `5000` | Web UI port on laptop |
| `BIRD_CONFIDENCE` | `0.60` | Minimum score to register bird detection |
| `TOMATO_CONFIDENCE` | `0.60` | Minimum score to register tomato detection |
| `BIRD_CONFIRMATION_FRAMES` | `3` | Consecutive detections before scaring |
| `BIRD_COOLDOWN_SECONDS` | `10.0` | Cooldown period between scare actions |

### `raspberry_pi/config.py`
| Variable | Default | Purpose |
|---|---|---|
| `CAMERA_PORT` | `8000` | Port for camera server |
| `COMMAND_PORT` | `9000` | Port for command server |
| `WATCHDOG_TIMEOUT` | `3.0` | Seconds without heartbeat before auto-stop |
| `GPIO_PINS` | Dict | BCM pin definitions for motors, servo, ultrasonic |

---

## 8. How to Start the System

### Step 1: Start Raspberry Pi
**On Raspberry Pi:**
```bash
./start_pi.sh
```
*(Or on Windows for local simulation: double click `START_PI.bat`)*

### Step 2: Configure IP on Laptop
In `laptop/config.py`, set:
```python
PI_IP = "192.168.1.105"  # Replace with your Pi's actual Wi-Fi/LAN IP
```
*(Or set environment variable `set SCAREX_PI_IP=192.168.1.105`)*

### Step 3: Start Laptop Hub
**On Laptop:**
Double click `START_LAPTOP.bat` or run:
```bash
python laptop/main.py
```

### Step 4: Open Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 9. Network Configuration

1. Both Laptop and Raspberry Pi must be connected to the same Wi-Fi router or Ethernet switch.
2. Find the Pi's IP address by running `hostname -I` on the Pi terminal.
3. Ensure ports `8000` (camera) and `9000` (command) are not blocked by firewall on the Pi.

---

## 10. Automated Testing Procedure

Run the test suite to verify all components end-to-end:

```bash
# 1. Test Bird & Tomato Detectors:
python tests/test_detectors.py

# 2. Test Stream & Dual Inference on Same Frame:
python tests/test_stream_and_inference.py

# 3. Test Command Server, Audio, Hardware & Watchdog:
python tests/test_command_and_safety.py
```

---

## 11. Safety & Emergency Stop

1. **Hardware Watchdog:** The laptop sends heartbeats every 1.0 second. If the network disconnects or laptop software freezes for longer than `3.0` seconds (`WATCHDOG_TIMEOUT`), the Pi immediately stops all motors and turns off actuators.
2. **Emergency Stop:** Clicking the red **EMERGENCY STOP** button on the web dashboard immediately dispatches `POST /command` with `{"command": "EMERGENCY_STOP"}` which stops motors and silences the speaker instantly.

---

## 12. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `Pi Camera: OFFLINE` on dashboard | Pi camera server not running or wrong IP | Verify `PI_IP` in `laptop/config.py` and ensure `camera_server.py` is running on the Pi. |
| `Raspberry Pi: DISCONNECTED` | Command server not running | Ensure port 9000 is open and `command_server.py` is active. |
| No sound playing on scare | Missing audio files | Confirm WAV files exist in `raspberry_pi/assets/audio/`. |
| Motors not moving on Pi | Mock mode active | Install `gpiozero` and verify pin mappings in `raspberry_pi/config.py`. |
