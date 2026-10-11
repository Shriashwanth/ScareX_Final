"""
Verification Test for Microphone Bird Audio Recognition & Deterrence
Validates the user requirement:
"if you hear this sound in microphone it should produce deterrence, if no sound of these sounds deterrence off"
"""
import sys
import os
import time
import numpy as np
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from laptop.audio.bird_audio_listener import bird_audio_listener
from laptop.fusion.decision_engine import DecisionEngine
from laptop.buzzer import is_playing, stop_deterrence

def test_acoustic_detection():
    print("=========================================================")
    print("Testing Microphone Bird Audio Recognition & Deterrence")
    print("=========================================================")

    # Ensure clean initial state
    stop_deterrence()
    engine = DecisionEngine()
    assert not is_playing(), "Audio should initially be silent"

    # Pick sample files from the user's Birds_Audio dataset
    audio_dir = os.path.join(BASE_DIR, "laptop", "assets", "birds_audio")
    sample_crow = os.path.join(audio_dir, "freesound_community-crow-67929.mp3")
    sample_sparrow = os.path.join(audio_dir, "freesound_community-032603_sparrow-65031.mp3")

    assert os.path.exists(sample_crow), f"Crow sample file missing: {sample_crow}"

    # -------------------------------------------------------------
    # Test 1: Simulating microphone hearing a Crow call from Birds_Audio
    # -------------------------------------------------------------
    print("\n[Test 1] Simulating microphone hearing Crow call from Birds_Audio...")
    y_crow, sr = librosa.load(sample_crow, sr=22050, duration=2.0)
    res_crow = bird_audio_listener.analyze_chunk(y_crow)
    print(f"  Classifier analysis result: {res_crow}")
    assert res_crow["detected"] is True, "Crow call from Birds_Audio MUST be recognized as bird"

    # Inject into bird_audio_listener state
    with bird_audio_listener.lock:
        bird_audio_listener.bird_detected = True
        bird_audio_listener.current_species = res_crow["species"]
        bird_audio_listener.confidence = res_crow["confidence"]

    # Process frame with camera seeing no bird
    engine.process_detections(bird_dets=[], tomato_dets=[])
    engine.process_detections(bird_dets=[], tomato_dets=[])

    status = engine.get_status()
    print(f"  Scare active: {status['bird']['scare_active']}")
    print(f"  Species: {status['bird']['species']} ({status['bird']['confidence']}%)")
    print(f"  Detection source: {status['bird']['source']}")
    print(f"  Audio playing: {is_playing()}")

    assert status['bird']['scare_active'] is True, "Deterrence MUST be active when bird sound is heard"
    assert is_playing() is True, "Deterrence sound MUST play when bird sound is heard"
    print("  -> PASSED: Hearing bird sound in microphone produces deterrence sound!")

    # -------------------------------------------------------------
    # Test 2: Simulating bird sound stopping (silence / room noise)
    # -------------------------------------------------------------
    print("\n[Test 2] Simulating bird sound stopping (silence / no bird audio)...")
    silence = np.zeros(22050, dtype=np.float32)
    res_silence = bird_audio_listener.analyze_chunk(silence)
    assert res_silence["detected"] is False, "Silence must be classified as no bird"

    # Inject silence into listener
    with bird_audio_listener.lock:
        bird_audio_listener.bird_detected = False
        bird_audio_listener.current_species = "None"
        bird_audio_listener.confidence = 0.0

    # Process frames with no bird seen or heard
    engine.process_detections(bird_dets=[], tomato_dets=[])
    engine.process_detections(bird_dets=[], tomato_dets=[])

    status = engine.get_status()
    print(f"  Scare active: {status['bird']['scare_active']}")
    print(f"  Species: {status['bird']['species']}")
    print(f"  Audio playing: {is_playing()}")

    assert status['bird']['scare_active'] is False, "Deterrence MUST be deactivated when sound stops"
    assert is_playing() is False, "Deterrence sound MUST be OFF when bird sound stops"
    print("  -> PASSED: When bird sound stops, deterrence sound turns OFF immediately!")

    # -------------------------------------------------------------
    # Test 3: Simulating Sparrow call from Birds_Audio
    # -------------------------------------------------------------
    if os.path.exists(sample_sparrow):
        print("\n[Test 3] Simulating microphone hearing Sparrow call from Birds_Audio...")
        y_sparrow, _ = librosa.load(sample_sparrow, sr=22050, duration=2.0)
        res_sparrow = bird_audio_listener.analyze_chunk(y_sparrow)
        print(f"  Classifier analysis result: {res_sparrow}")
        assert res_sparrow["detected"] is True, "Sparrow call MUST be recognized"

        with bird_audio_listener.lock:
            bird_audio_listener.bird_detected = True
            bird_audio_listener.current_species = res_sparrow["species"]
            bird_audio_listener.confidence = res_sparrow["confidence"]

        engine.process_detections(bird_dets=[], tomato_dets=[])
        assert is_playing() is True, "Sparrow audio produces deterrence sound"
        print("  -> PASSED: Sparrow audio triggered deterrence!")

        # Turn off
        with bird_audio_listener.lock:
            bird_audio_listener.bird_detected = False
        engine.process_detections([], [])
        engine.process_detections([], [])
        assert is_playing() is False, "Sound stops when sparrow stops"
        print("  -> PASSED: Deterrence sound turned OFF after sparrow stopped.")

    print("\n=========================================================")
    print("ALL MICROPHONE ACOUSTIC BIRD TESTS PASSED 100%!")
    print("=========================================================")

if __name__ == "__main__":
    test_acoustic_detection()
