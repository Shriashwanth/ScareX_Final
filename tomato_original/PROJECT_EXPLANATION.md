# ScareX: AI-Powered Smart Bird Detection and Deterrent System

## Overview
**ScareX** is an autonomous, AI-powered agricultural robot designed to protect crops from harmful birds. It acts as a highly advanced, automated scarecrow that leverages computer vision and audio recognition to detect specific bird species and deploy targeted deterrents.

## The Problem It Solves
Farmers often lose significant portions of their crops (like paddy, sugarcane, and fruits) to various bird species. Traditional scarecrows are static and quickly ignored by birds. Modern acoustic deterrents (like propane cannons or continuous sirens) create noise pollution and become less effective over time as birds habituate to the constant, random noise.

## How ScareX Works
ScareX solves this by being **dynamic, targeted, and intelligent**. It operates completely offline on a low-cost, lightweight hardware platform (specifically designed for the Raspberry Pi 3).

Here is the step-by-step workflow of the system:

1. **Continuous Monitoring (Vision & Audio)**
   - **Camera:** The robot uses a lightweight YOLOv11 AI model (optimized via TensorFlow Lite) to continuously scan its environment for birds.
   - **Microphone:** Simultaneously, it listens for bird calls using a MobileNetV2 audio classification model that converts sounds into Mel Spectrograms for analysis.

2. **Sensor Fusion & Confirmation**
   - The `Fusion Engine` combines data from both the camera and the microphone to increase accuracy and reduce false alarms (e.g., ignoring a tractor noise but triggering on a crow's caw).
   - If the microphone hears a bird but the camera doesn't see it, the system can automatically command a servo motor to rotate the camera towards the sound source for visual confirmation.

3. **Targeted Deterrents**
   - Instead of playing a generic siren, ScareX identifies the *exact species* of the bird (e.g., Crow, Parakeet, Pigeon).
   - It then plays a specific predator call or alarm tailored to scare that specific bird (e.g., playing a Hawk call to scare a Sparrow, or an Eagle call to scare a Parakeet).
   - If the bird is harmless or beneficial (like farm poultry/hens), the system can be configured to ignore them or just play a gentle beep.

4. **Autonomous Navigation**
   - The robot is mounted on a motorized chassis. It uses an ultrasonic sensor (HC-SR04) to roam the agricultural field autonomously, stopping and turning to avoid obstacles like fences, trees, or crops.

5. **Logging and Dashboard**
   - Every detection and action is logged into a local SQLite database for analytics.
   - Farmers can connect to the robot's local Wi-Fi and open a web dashboard (built with Flask) on their phone or laptop to view a live video feed, real-time AI confidences, battery levels, and event logs.

## Why it is Built This Way
- **Offline & Edge AI:** Agricultural fields often lack internet access. All AI models in ScareX are heavily quantized (INT8) and use TensorFlow Lite so they can run directly on the Raspberry Pi 3's limited CPU without needing cloud servers.
- **Power Efficiency:** By using lightweight models and a Raspberry Pi, the system can run on a standard battery/power bank for extended periods.

## Summary
In short, ScareX is a smart robotic scarecrow that roams your farm, uses a camera and microphone to find birds, figures out exactly what kind of bird it is, and plays the perfect scary sound to chase it away—all while running completely offline.
