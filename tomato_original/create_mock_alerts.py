import numpy as np
from scipy.io.wavfile import write
import os

def create_synthetic_wav(filename, frequency=440, duration=1.0):
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a simple waveform (e.g., triangle or sine wave)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add a little noise
    wave += np.random.normal(0, 0.05, len(t))
    
    # Convert to 16-bit PCM
    wave_16bit = np.int16(wave * 32767)
    write(filename, sample_rate, wave_16bit)
    print(f"Created {filename}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, 'assets', 'audio')
    
    os.makedirs(audio_dir, exist_ok=True)
    
    sounds = {
        "hawk.wav": 800,
        "eagle.wav": 1000,
        "falcon.wav": 1200,
        "siren.wav": 600,
        "loud_siren.wav": 500,
        "alarm.wav": 700,
        "beep.wav": 440
    }
    
    for filename, freq in sounds.items():
        filepath = os.path.join(audio_dir, filename)
        create_synthetic_wav(filepath, frequency=freq)
        
    print("Mock alarm sound files generated successfully.")
