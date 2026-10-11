# ScareX – AI Autonomous Bird Protection Robot

ScareX is a modular, AI-powered desktop application designed to detect specific bird species via camera and microphone, and trigger an automated "scare sequence" using sound and hardware actions (e.g., rotating a camera, flapping wings).

## Features
- **Vision Module**: Uses YOLOv11 and OpenCV to detect birds (Crow, Common Myna, Rose Ringed Parakeet, Peacock, Pigeon).
- **Audio Module**: Uses `librosa`, `scikit-learn`, and `sounddevice` to recognize bird calls via MFCC features.
- **Decision Engine**: Combines vision and audio inputs to trigger a scare event if confidence thresholds are met.
- **Hardware Abstraction**: A mock layer (`src/hardware/hardware.py`) that can be easily replaced with Raspberry Pi GPIO code in the future.
- **GUI**: Built with `customtkinter` for a modern, dark-themed dashboard.

## Requirements
- Python 3.9+
- A working webcam and microphone.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your datasets in the following directories:
   - YOLOv11 Images: `dataset/images/BIRDS.v1i.yolov11/`
   - Bird Audio for Training: `dataset/audio/birds/<bird_name>/`
   - Scare Sounds: `dataset/audio/ScareCalls/`

## Usage

1. **Train Models**:
   Run the application, then use the **Train Vision Model** and **Train Audio Model** buttons on the dashboard to train the AI models. The models will be saved in the `models/` directory.

2. **Start the System**:
   Run the main script:
   ```bash
   python src/main.py
   ```
   Click the **Start System** button in the dashboard to begin monitoring.

## Raspberry Pi Deployment
This project is designed with migration in mind. When deploying to a Raspberry Pi:
1. Copy this entire project to the Pi.
2. Edit `src/hardware/hardware.py` and replace the mock `print` statements with actual `RPi.GPIO` or `gpiozero` logic to control servos and motors.
3. No other files need to be modified!
