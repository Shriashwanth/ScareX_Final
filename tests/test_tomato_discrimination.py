"""
Test Suite: Tomato Discrimination and False Positive Rejection
Verifies that human faces, skin, walls, furniture, and room objects are strictly rejected,
while genuine tomatoes (fully ripened, half ripened, green) are accurately validated.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from raspberry_pi.models.tomato.detector import TomatoDetector
except ImportError:
    from laptop.models.tomato.detector import TomatoDetector

def test_reject_human_faces_and_skin():
    """Human skin across all Fitzpatrick scales must NEVER pass as any tomato."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100, 100, 200, 200]
    
    skin_tones = [
        ("Pale Skin (Fitzpatrick I)", (180, 200, 250)),
        ("Fair Skin (Fitzpatrick II)", (150, 180, 235)),
        ("Medium Skin (Fitzpatrick III)", (110, 150, 210)),
        ("Olive Skin (Fitzpatrick IV)", (90, 130, 180)),
        ("Brown Skin (Fitzpatrick V)", (50, 80, 130)),
        ("Dark Brown Skin (Fitzpatrick VI)", (30, 45, 75))
    ]
    
    for name, bgr in skin_tones:
        frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = bgr
        for label in ["fully_ripened", "half_ripened", "green"]:
            is_valid, reason = TomatoDetector.verify_tomato_patch(frame, bbox, label)
            assert not is_valid, f"False positive leak! Skin '{name}' passed as '{label}' (Reason: {reason})"

def test_reject_walls_and_indoor_background():
    """Walls, ceilings, desks, and neutral backgrounds must be rejected."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [50, 50, 220, 220]
    
    backgrounds = [
        ("White Wall", (240, 240, 240)),
        ("Cream / Off-White Wall", (190, 210, 220)),
        ("Beige Wall", (160, 190, 205)),
        ("Grey Concrete / Drywall", (140, 140, 140)),
        ("Wooden Desk / Table", (40, 70, 120)),
        ("Black Chair / Dark Shadow", (20, 20, 20))
    ]
    
    for name, bgr in backgrounds:
        frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = bgr
        for label in ["fully_ripened", "half_ripened", "green"]:
            is_valid, reason = TomatoDetector.verify_tomato_patch(frame, bbox, label)
            assert not is_valid, f"False positive leak! Background '{name}' passed as '{label}' (Reason: {reason})"

def test_reject_distorted_aspect_ratios():
    """Non-spherical/non-oval shapes (doorways, arms, ceiling panels) must be rejected."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Deep red ripe tomato color, but elongated shape
    red = (25, 30, 210)
    
    # Very tall box (w=40, h=250 -> aspect=0.16)
    tall_bbox = [100, 50, 140, 300]
    frame[tall_bbox[1]:tall_bbox[3], tall_bbox[0]:tall_bbox[2]] = red
    valid, reason = TomatoDetector.verify_tomato_patch(frame, tall_bbox, "fully_ripened")
    assert not valid and "aspect_ratio" in reason
    
    # Very wide box (w=300, h=40 -> aspect=7.5)
    wide_bbox = [50, 100, 350, 140]
    frame[wide_bbox[1]:wide_bbox[3], wide_bbox[0]:wide_bbox[2]] = red
    valid, reason = TomatoDetector.verify_tomato_patch(frame, wide_bbox, "fully_ripened")
    assert not valid and "aspect_ratio" in reason

def test_reject_oversized_regions():
    """A bounding box spanning the entire wall/frame area must be rejected."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    huge_bbox = [10, 10, 630, 470]  # ~95% of frame
    frame[huge_bbox[1]:huge_bbox[3], huge_bbox[0]:huge_bbox[2]] = (25, 30, 210)
    valid, reason = TomatoDetector.verify_tomato_patch(frame, huge_bbox, "fully_ripened")
    assert not valid and "too_large" in reason

def test_accept_genuine_tomatoes():
    """Genuine tomatoes with natural color profiles must pass verification."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [150, 150, 270, 270]  # 120x120 aspect 1.0
    
    # 1. Vibrant Fully Ripened Red Tomato
    frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = (20, 25, 215)
    valid, reason = TomatoDetector.verify_tomato_patch(frame, bbox, "fully_ripened")
    assert valid, f"Valid ripe tomato was rejected! Reason: {reason}"
    
    # 2. Vine Deep Red Tomato
    frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = (15, 20, 180)
    valid, reason = TomatoDetector.verify_tomato_patch(frame, bbox, "fully_ripened")
    assert valid, f"Valid deep red tomato was rejected! Reason: {reason}"
    
    # 3. Half Ripened (Turning / Orange) Tomato
    frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = (25, 95, 220)
    valid, reason = TomatoDetector.verify_tomato_patch(frame, bbox, "half_ripened")
    assert valid, f"Valid half-ripened tomato was rejected! Reason: {reason}"
    
    # 4. Green Unripe Tomato
    frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = (40, 180, 55)
    valid, reason = TomatoDetector.verify_tomato_patch(frame, bbox, "green")
    assert valid, f"Valid green tomato was rejected! Reason: {reason}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
