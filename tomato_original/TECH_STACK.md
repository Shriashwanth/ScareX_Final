# ScareX Technology Stack: End-to-End Architecture

This document provides a detailed explanation of the technologies and libraries used to build the **ScareX** automated bird deterrent system.

---

## 1. Core Language: Python
* **What we use:** Python (Version 3.10+)
* **Why we use it:** Python is the industry standard for AI, machine learning, and computer vision. It has the richest ecosystem of libraries for hardware integration, audio processing, and deep learning, making it the perfect glue language for an AI-driven IoT system.
* **Where we use it:** The entire backend, frontend GUI, and AI inference pipeline are written in Python.

---

## 2. Vision System (Object Detection)
### Ultralytics YOLOv11 (`ultralytics`)
* **What we use:** YOLO (You Only Look Once) version 11 is a state-of-the-art, real-time object detection model.
* **Why we use it:** We need to process live video feeds from a webcam in real-time (30+ frames per second). Traditional image classification is too slow and doesn't provide bounding boxes. YOLO is lightweight enough to run on a laptop without a dedicated GPU while still being highly accurate.
* **Where we use it:** `src/vision/detector.py` and `src/vision/trainer.py`. It is responsible for taking individual frames from the webcam, detecting if a "Bird" is in the frame, and drawing the red bounding boxes.

### OpenCV (`cv2`)
* **What we use:** Open Source Computer Vision Library.
* **Why we use it:** YOLO does the *thinking*, but OpenCV does the *seeing*. We need OpenCV to connect to the laptop's physical webcam, capture frames, manipulate image arrays, and render the final video feed to the screen.
* **Where we use it:** `src/vision/detector.py`. It initializes the camera (`cv2.VideoCapture(0, cv2.CAP_DSHOW)`) and draws the confidence text and boxes onto the frames.

---

## 3. Audio System (Sound Recognition)
### SoundDevice (`sounddevice`)
* **What we use:** A Python library for capturing real-time audio streams.
* **Why we use it:** We need to constantly listen to the environment without freezing the rest of the application. `sounddevice` allows non-blocking, background recording of the microphone.
* **Where we use it:** `src/audio/recognizer.py` to continuously record 1-second chunks of audio from the laptop microphone.

### Librosa (`librosa`)
* **What we use:** A library for music and audio analysis.
* **Why we use it:** Raw audio waves are just a massive array of numbers. To feed audio into a machine learning model, we have to extract meaningful features (like Pitch, Mel-Frequency Cepstral Coefficients (MFCCs), and Spectral Centroids). `librosa` mathematically converts raw sound into these recognizable patterns.
* **Where we use it:** `src/audio/trainer.py` and `src/audio/recognizer.py` for feature extraction.

### Scikit-Learn (`sklearn` - Random Forest)
* **What we use:** A traditional machine learning library, specifically using the **Random Forest Classifier**.
* **Why we use it:** While we could use deep learning for audio, a Random Forest is incredibly fast to train (takes 1 second on a CPU) and requires very little data to achieve 90%+ accuracy for distinct sounds like bird caws.
* **Where we use it:** `src/audio/trainer.py` (to build the model) and `src/audio/recognizer.py` (to predict the bird species in real-time).

---

## 4. Hardware & Alarm System
### Pygame (`pygame`)
* **What we use:** A multimedia library (specifically `pygame.mixer`).
* **Why we use it:** Playing audio asynchronously in Python can be notoriously buggy. Pygame's mixer is highly optimized, reliable, and allows us to easily control volume, play, pause, and stop overlapping alarm sounds (`.wav` and `.mp3`) without blocking the camera feed.
* **Where we use it:** `src/audio/scare_player.py`.

### Hardware Controller (Mocked)
* **What we use:** Custom Python classes and `threading`.
* **Why we use it:** Real hardware (servos, lasers, speakers) requires GPIO pins (like on a Raspberry Pi). Since we are running on a Windows laptop, we use a software "Mock" to simulate these hardware actions (printing "Rotating camera", "Flapping wings" to the console).
* **Where we use it:** `src/hardware/hardware.py`.

---

## 5. User Interface (GUI)
### CustomTkinter (`customtkinter`)
* **What we use:** A modern, dark-mode wrapper around Python's standard `tkinter`.
* **Why we use it:** Standard `tkinter` looks like a program from Windows 95. `customtkinter` allows us to build a sleek, modern, dark-themed dashboard with rounded buttons and progress bars while keeping the app completely native (no web browser required).
* **Where we use it:** `src/gui/dashboard.py` and `src/main.py`.

---

## How It All Connects (The Decision Engine)
The **Decision Engine** (`src/core/decision_engine.py`) is the brain of the operation. It uses Python's `threading` library to run independently of the GUI. 

1. It constantly asks the **Vision System** (YOLO/OpenCV) "Do you see a bird?"
2. It constantly asks the **Audio System** (Librosa/Sklearn) "Do you hear a specific bird?"
3. If both answer yes, or one answers with high confidence, the Engine triggers the **Scare Player** (Pygame) and the **Hardware Controller** to scare the bird away!
