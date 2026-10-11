import os
import numpy as np
from scipy.io import wavfile
import glob

def generate_bird_audio(bird_name, generate_func, count=2):
    base_dir = os.path.dirname(__file__)
    bird_dir = os.path.join(base_dir, 'dataset', 'audio', bird_name)
    os.makedirs(bird_dir, exist_ok=True)
    
    # Remove old mocks
    for f in glob.glob(os.path.join(bird_dir, '*.wav')):
        os.remove(f)

    sample_rate = 44100
    for i in range(count):
        audio = generate_func(sample_rate, variation=i)
        # Normalize to 16-bit PCM
        audio = audio / np.max(np.abs(audio))
        audio_data = np.int16(audio * 32767)
        wavfile.write(os.path.join(bird_dir, f'synth_{bird_name}_{i}.wav'), sample_rate, audio_data)
        
    print(f"Generated synthetic {bird_name} sounds in {bird_dir}")

def generate_crow(sample_rate, variation=0):
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    f0 = 400 + (variation * 50)
    wave = np.zeros_like(t)
    for i in range(1, 10):
        fm = np.sin(2 * np.pi * 50 * t) * 0.1
        wave += (1.0 / i) * np.sin(2 * np.pi * f0 * i * t * (1 + fm))
    envelope = np.exp(-5 * t) * (1 - np.exp(-100 * t))
    return wave * envelope

def generate_peacock(sample_rate, variation=0):
    duration = 0.8
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Pitch sweep (honk)
    f0 = 800 + (variation * 100)
    f1 = 1200
    freq = np.linspace(f0, f1, len(t))
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    envelope = np.sin(np.pi * t / duration) ** 2
    return wave * envelope

def generate_dove(sample_rate, variation=0):
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Low frequency cooing with amplitude modulation
    f0 = 150 + (variation * 20)
    wave = np.sin(2 * np.pi * f0 * t)
    # Cooing rhythm
    envelope = 0.5 * (1 - np.cos(2 * np.pi * 3 * t)) * (np.sin(np.pi * t / duration))
    return wave * envelope

def generate_parrot(sample_rate, variation=0):
    duration = 0.4
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # High pitch screech with fast FM
    f0 = 2000 + (variation * 200)
    fm = np.sin(2 * np.pi * 200 * t) * 0.5
    wave = np.sin(2 * np.pi * f0 * t * (1 + fm))
    envelope = np.exp(-3 * t) * (1 - np.exp(-50 * t))
    return wave * envelope

def generate_myna(sample_rate, variation=0):
    duration = 0.6
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Varied chatter: frequency hopping
    freq = 1000 + 500 * np.sin(2 * np.pi * 10 * t) + (variation * 100)
    wave = np.sin(2 * np.pi * freq * t)
    # Bursts of amplitude
    envelope = np.sin(2 * np.pi * 8 * t) ** 4
    # Smooth overall
    envelope *= np.sin(np.pi * t / duration)
    return wave * envelope

if __name__ == "__main__":
    generate_bird_audio('Crow', generate_crow)
    generate_bird_audio('Peacock', generate_peacock)
    generate_bird_audio('Dove', generate_dove)
    generate_bird_audio('Parakeet', generate_parrot)
    generate_bird_audio('Common Myna', generate_myna)
    print("All bird datasets generated successfully!")
