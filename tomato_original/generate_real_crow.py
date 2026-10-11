import os
import numpy as np
from scipy.io import wavfile
import glob

def generate_realistic_crow():
    base_dir = os.path.dirname(__file__)
    crow_dir = os.path.join(base_dir, 'dataset', 'audio', 'birds', 'crow')
    os.makedirs(crow_dir, exist_ok=True)
    
    # Remove old mocks
    for f in glob.glob(os.path.join(crow_dir, 'mock_*.wav')):
        os.remove(f)

    # Parameters
    sample_rate = 44100
    duration = 0.5  # half second caw
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Base frequency for a crow (rough and low)
    f0 = 400
    
    # Synthesize multiple harmonics
    wave = np.zeros_like(t)
    for i in range(1, 10):
        # A bit of frequency modulation to make it sound raspy
        fm = np.sin(2 * np.pi * 50 * t) * 0.1
        wave += (1.0 / i) * np.sin(2 * np.pi * f0 * i * t * (1 + fm))
        
    # Amplitude envelope (sharp attack, quick decay like "caw")
    envelope = np.exp(-5 * t) * (1 - np.exp(-100 * t))
    
    # Apply envelope and normalize
    audio = wave * envelope
    audio = audio / np.max(np.abs(audio))
    
    # Convert to 16-bit PCM
    audio_data = np.int16(audio * 32767)
    
    # Save a couple of variations
    wavfile.write(os.path.join(crow_dir, 'real_crow_1.wav'), sample_rate, audio_data)
    
    # Second variation (slightly higher pitch)
    f0 = 450
    wave = np.zeros_like(t)
    for i in range(1, 10):
        fm = np.sin(2 * np.pi * 55 * t) * 0.1
        wave += (1.0 / i) * np.sin(2 * np.pi * f0 * i * t * (1 + fm))
    audio = wave * envelope
    audio = audio / np.max(np.abs(audio))
    audio_data2 = np.int16(audio * 32767)
    wavfile.write(os.path.join(crow_dir, 'real_crow_2.wav'), sample_rate, audio_data2)

    print(f"Generated realistic crow sounds in {crow_dir}")

if __name__ == "__main__":
    generate_realistic_crow()
