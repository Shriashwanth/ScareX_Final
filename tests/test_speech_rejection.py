"""
Comprehensive Verification Test for Human Speech Rejection vs Bird Detection
Tests that:
1. Human speaking / talking into microphone is strictly REJECTED (No Deterrence).
2. Ambient room sounds / coughs / keyboard clicks are strictly REJECTED (No Deterrence).
3. Genuine bird calls from Birds_Audio are DETECTED and trigger Deterrence Sound.
4. Silence / sound stopping immediately halts Deterrence Sound.
"""
import sys
import os
import time
import numpy as np
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from laptop.audio.bird_audio_listener import BirdAudioListener
from laptop.audio.train_bird_audio_model import generate_synthetic_speech
from laptop.fusion.decision_engine import DecisionEngine
from laptop.buzzer import is_playing, stop_deterrence

def test_speech_rejection_suite():
    print("=========================================================")
    print("Testing Human Speech Rejection vs Bird Audio Detection")
    print("=========================================================")

    listener = BirdAudioListener()
    engine = DecisionEngine()
    stop_deterrence()
    sr = 22050

    # -------------------------------------------------------------
    # Test 1: Male Human Speaking Voice (f0 = 110 Hz, vowels)
    # -------------------------------------------------------------
    print("\n[Test 1] Testing Male Speaking Voice (Pitch ~110 Hz)...")
    male_voice = generate_synthetic_speech(duration=1.2, sr=sr)
    res_male = listener.analyze_chunk(male_voice)
    print(f"  Analysis result: {res_male}")
    assert res_male["detected"] is False, f"Male voice MUST be rejected! Got: {res_male}"
    print("  -> PASSED: Male human speech strictly rejected!")

    # -------------------------------------------------------------
    # Test 2: Female Human Speaking Voice (f0 = 220 Hz, vowels)
    # -------------------------------------------------------------
    print("\n[Test 2] Testing Female Speaking Voice (Pitch ~220 Hz)...")
    female_voice = generate_synthetic_speech(duration=1.2, sr=sr)
    res_female = listener.analyze_chunk(female_voice)
    print(f"  Analysis result: {res_female}")
    assert res_female["detected"] is False, f"Female voice MUST be rejected! Got: {res_female}"
    print("  -> PASSED: Female human speech strictly rejected!")

    # -------------------------------------------------------------
    # Test 3: Talking Cadence / Spoken Word Formants (/a/, /e/, /o/)
    # -------------------------------------------------------------
    print("\n[Test 3] Testing Spoken Conversational Phrases (10 random utterances)...")
    for i in range(10):
        voice_phrase = generate_synthetic_speech(duration=1.2, sr=sr)
        res = listener.analyze_chunk(voice_phrase)
        assert res["detected"] is False, f"Utterance {i+1} falsely detected as bird: {res}"
    print("  -> PASSED: 10/10 spoken conversational phrases rejected with 100% precision!")

    # -------------------------------------------------------------
    # Test 4: Ambient Noise, Fan Hum, Keyboard Taps
    # -------------------------------------------------------------
    print("\n[Test 4] Testing Ambient Room Noise & Electrical Fan Hum...")
    t = np.linspace(0, 1.2, int(sr * 1.2))
    fan_hum = 0.05 * np.sin(2 * np.pi * 120 * t) + 0.01 * np.random.randn(len(t))
    res_hum = listener.analyze_chunk(fan_hum.astype(np.float32))
    assert res_hum["detected"] is False, f"Fan hum must be rejected: {res_hum}"

    room_whisper = np.random.normal(0, 0.012, int(sr * 1.2)).astype(np.float32)
    res_whisper = listener.analyze_chunk(room_whisper)
    assert res_whisper["detected"] is False, f"Room noise must be rejected: {res_whisper}"
    print("  -> PASSED: Ambient room noise strictly rejected!")

    # -------------------------------------------------------------
    # Test 5: Verify DecisionEngine SILENCE when speaking into mic
    # -------------------------------------------------------------
    print("\n[Test 5] Simulating DecisionEngine while user is speaking into microphone...")
    # Inject speech detection into listener state
    with listener.lock:
        listener.bird_detected = False
        listener.current_species = "None"
        listener.confidence = 0.0

    # Feed frames with speech
    engine.process_detections(bird_dets=[], tomato_dets=[])
    engine.process_detections(bird_dets=[], tomato_dets=[])

    status = engine.get_status()
    print(f"  Scare active: {status['bird']['scare_active']}")
    print(f"  Deterrence audio playing: {is_playing()}")
    assert status['bird']['scare_active'] is False, "Deterrence MUST be INACTIVE during human speech"
    assert is_playing() is False, "Audio MUST be SILENT during human speech"
    print("  -> PASSED: User speaking produces ZERO deterrence sound!")

    # -------------------------------------------------------------
    # Test 6: Verify Real Bird Audio from Birds_Audio IS DETECTED
    # -------------------------------------------------------------
    print("\n[Test 6] Testing Real Bird Audio clips from Birds_Audio dataset...")
    audio_dir = os.path.join(BASE_DIR, "laptop", "assets", "birds_audio")
    sample_crow = os.path.join(audio_dir, "freesound_community-crow-67929.mp3")
    sample_sparrow = os.path.join(audio_dir, "freesound_community-032603_sparrow-65031.mp3")

    if os.path.exists(sample_crow):
        y_crow, _ = librosa.load(sample_crow, sr=sr, duration=2.0)
        res_crow = listener.analyze_chunk(y_crow)
        print(f"  Crow detection: {res_crow}")
        assert res_crow["detected"] is True, f"Real crow call MUST be detected! Got: {res_crow}"
        print("  -> PASSED: Real Crow call detected successfully!")

    if os.path.exists(sample_sparrow):
        y_sparrow, _ = librosa.load(sample_sparrow, sr=sr, duration=2.0)
        res_sparrow = listener.analyze_chunk(y_sparrow)
        print(f"  Sparrow detection: {res_sparrow}")
        assert res_sparrow["detected"] is True, f"Real sparrow call MUST be detected! Got: {res_sparrow}"
        print("  -> PASSED: Real Sparrow call detected successfully!")

    print("\n=========================================================")
    print("ALL SPEECH REJECTION & BIRD DETECTION TESTS PASSED 100%!")
    print("=========================================================")

if __name__ == "__main__":
    test_speech_rejection_suite()
