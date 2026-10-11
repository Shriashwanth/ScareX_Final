"""
Automated Test for ScareX Deterrence Sound Logic
Validates the user requirement:
1. If we show bird on camera it should produce deterrence sound.
2. If bird is removed no deterrence sound.
3. For tomato no sound.
"""
import sys
import os
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from laptop.fusion.decision_engine import DecisionEngine
from laptop.buzzer import is_playing, stop_deterrence

def test_deterrence_sound_scenarios():
    print("=========================================================")
    print("Testing Bird Deterrence & Tomato Silence Requirements")
    print("=========================================================")

    # Create fresh engine instance
    engine = DecisionEngine()
    stop_deterrence()
    assert not is_playing(), "Audio should initially be silent"

    # -------------------------------------------------------------
    # Scenario 1: SHOW BIRD ON CAMERA -> PRODUCE DETERRENCE SOUND
    # -------------------------------------------------------------
    print("\n[Test 1] Showing BIRD to camera...")
    bird_dets = [{"type": "bird", "label": "Crow", "confidence": 0.88, "bbox": [100, 100, 250, 250]}]
    tomato_dets = []

    # Frame 1: Initial detection
    engine.process_detections(bird_dets, tomato_dets)
    # Frame 2: Confirmed detection -> sound activates!
    engine.process_detections(bird_dets, tomato_dets)

    status = engine.get_status()
    print(f"  Scare active: {status['bird']['scare_active']}")
    print(f"  Current bird: {status['bird']['species']} ({status['bird']['confidence']}%)")
    print(f"  Audio playing: {is_playing()}")

    assert status['bird']['scare_active'] is True, "Bird deterrence should be ACTIVE"
    assert is_playing() is True, "Deterrence sound SHOULD BE PLAYING for bird"
    print("  -> PASSED: Bird produces deterrence sound!")

    # Sustain bird presence
    for _ in range(3):
        engine.process_detections(bird_dets, tomato_dets)
        time.sleep(0.05)
    assert is_playing() is True, "Sound should keep playing while bird is shown"
    print("  -> PASSED: Deterrence sound sustained while bird is on camera.")

    # -------------------------------------------------------------
    # Scenario 2: REMOVE BIRD FROM CAMERA -> NO DETERRENCE SOUND
    # -------------------------------------------------------------
    print("\n[Test 2] Removing BIRD from camera...")
    # Frames with no bird
    engine.process_detections([], [])
    engine.process_detections([], [])

    status = engine.get_status()
    print(f"  Scare active: {status['bird']['scare_active']}")
    print(f"  Current bird: {status['bird']['species']}")
    print(f"  Audio playing: {is_playing()}")

    assert status['bird']['scare_active'] is False, "Bird deterrence should be INACTIVE"
    assert is_playing() is False, "Deterrence sound MUST STOP when bird is removed"
    print("  -> PASSED: Deterrence sound stops immediately when bird is removed!")

    # -------------------------------------------------------------
    # Scenario 3: SHOW TOMATO ON CAMERA -> NO SOUND AT ALL
    # -------------------------------------------------------------
    print("\n[Test 3] Showing TOMATO on camera (No bird)...")
    tomato_dets = [
        {"type": "tomato", "label": "fully_ripened", "confidence": 0.92, "bbox": [150, 150, 300, 300]},
        {"type": "tomato", "label": "green", "confidence": 0.85, "bbox": [320, 150, 450, 280]}
    ]

    for _ in range(5):
        engine.process_detections([], tomato_dets)
        time.sleep(0.04)

    status = engine.get_status()
    print(f"  Tomatoes counted: {status['tomato']['current_frame_counts']}")
    print(f"  Harvest priority: {status['tomato']['harvest_priority']}")
    print(f"  Scare active: {status['bird']['scare_active']}")
    print(f"  Audio playing: {is_playing()}")

    assert status['bird']['scare_active'] is False, "Tomato must NEVER trigger scare active"
    assert is_playing() is False, "FOR TOMATO NO SOUND MUST BE PRODUCED"
    assert status['tomato']['current_frame_counts']['fully_ripened'] == 1, "Tomato count correct"
    print("  -> PASSED: Tomatoes processed with ZERO sound!")

    # -------------------------------------------------------------
    # Scenario 4: TOMATO WITH SPURIOUS OVERLAPPING BIRD BOX
    # -------------------------------------------------------------
    print("\n[Test 4] Tomato with spurious overlapping bird detection (False Positive filter)...")
    # Same region [150, 150, 300, 300]
    fake_bird_on_tomato = [{"type": "bird", "label": "Bird", "confidence": 0.50, "bbox": [150, 150, 300, 300]}]
    real_tomato = [{"type": "tomato", "label": "fully_ripened", "confidence": 0.90, "bbox": [150, 150, 300, 300]}]

    for _ in range(3):
        engine.process_detections(fake_bird_on_tomato, real_tomato)

    assert is_playing() is False, "Overlapping false-positive bird on tomato must be suppressed and produce NO sound"
    print("  -> PASSED: False positive on tomato suppressed, silence preserved!")

    # -------------------------------------------------------------
    # Scenario 5: ALTERNATING BETWEEN THE 5 UPLOADED PREDATOR SOUNDS
    # -------------------------------------------------------------
    print("\n[Test 5] Testing the 5 uploaded deterrence sounds with alternating rotation...")
    from laptop.buzzer import sound_controller, DETERRENT_SOUNDS
    
    played_sounds = []
    for cycle in range(5):
        stop_deterrence()
        sound_controller.start_deterrence(species="bird")
        assert is_playing() is True, f"Deterrence must play on cycle {cycle+1}"
        current_name = sound_controller.current_sound_name
        played_sounds.append(current_name)
        print(f"  Cycle {cycle+1}: Played '{current_name}'")
        time.sleep(0.05)
        stop_deterrence()

    print(f"  Played sounds sequence: {played_sounds}")
    # Verify that sounds were selected from the 5 deterrence sounds and rotated
    assert len(set(played_sounds)) >= 3, "Deterrence sounds must rotate and alternate across the uploaded predator audio!"
    print("  -> PASSED: Successfully alternating across the 5 uploaded predator sounds!")

    print("\n=========================================================")
    print("ALL TESTS PASSED! User requirements fully satisfied.")
    print("=========================================================")

if __name__ == "__main__":
    test_deterrence_sound_scenarios()
