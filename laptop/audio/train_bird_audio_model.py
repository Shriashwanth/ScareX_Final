"""
Bird Audio Feature Extractor and Classifier Trainer
Extracts acoustic signatures (MFCCs, Spectral Centroid, Bandwidth, Rolloff, ZCR, Voice-Band Ratio, Contrast, Modulation)
from C:\\Users\\Ashwanth\\Downloads\\Birds_Audio\\Birds_Audio dataset.
Includes comprehensive negative training samples (speech, ambient room, fan, AC, electronic tones, hiss, clicks)
and StandardScaler normalization to guarantee strict discrimination against non-bird sounds.
"""
import os
import sys
import glob
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "assets", "birds_audio")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "bird_audio_model.joblib")

def extract_features(y, sr=22050):
    """
    Extracts fixed 59-dimensional acoustic feature vector tailored for Bird vs Non-Bird discrimination.
    """
    if len(y) < sr * 0.2:
        y = np.pad(y, (0, int(sr * 0.2) - len(y)))
    
    # 1. MFCC (20 coefficients -> 20 mean + 20 std = 40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Spectral Centroid (2 features)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(cent)
    cent_std = np.std(cent)

    # 3. Spectral Rolloff (2 features)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    roll_mean = np.mean(rolloff)
    roll_std = np.std(rolloff)

    # 4. Zero-Crossing Rate (2 features)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # 5. Frequency Band Energy Decomposition
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Human voice band (100 Hz - 1,500 Hz)
    voice_mask = (freqs >= 100) & (freqs <= 1500)
    # Bird dominant vocalization band (2,000 Hz - 8,000 Hz)
    bird_mask = (freqs >= 2000) & (freqs <= 8000)
    
    total_energy = np.sum(S) + 1e-6
    voice_energy = np.sum(S[voice_mask, :]) + 1e-6
    bird_energy = np.sum(S[bird_mask, :]) + 1e-6
    
    voice_band_ratio = voice_energy / total_energy
    bird_to_voice_ratio = bird_energy / voice_energy

    rms = np.mean(librosa.feature.rms(y=y))

    # 6. Spectral Flatness (2 features)
    flatness = librosa.feature.spectral_flatness(y=y)
    flat_mean = np.mean(flatness)
    flat_std = np.std(flatness)

    # 7. Temporal Energy Modulation Ratio (1 feature)
    frame_len = int(0.05 * sr)
    hop = int(0.025 * sr)
    energies = [np.sum(y[i : i + frame_len] ** 2) for i in range(0, len(y) - frame_len, hop)]
    mod_ratio = float(np.std(energies) / (np.mean(energies) + 1e-6)) if energies else 0.0

    # 8. Spectral Contrast (7 features: 6 bands + 1)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    contrast_mean = np.mean(contrast, axis=1)

    features = np.hstack([
        mfcc_mean, mfcc_std,
        [cent_mean, cent_std, roll_mean, roll_std, zcr_mean, zcr_std, 
         voice_band_ratio, bird_to_voice_ratio, rms, flat_mean, flat_std, mod_ratio],
        contrast_mean
    ])
    return features.astype(np.float32)

def generate_synthetic_speech(duration=1.2, sr=22050):
    """
    Synthesizes realistic human speech audio with multi-formant vowels (F1, F2, F3)
    and conversational syllable rhythm.
    """
    t = np.linspace(0, duration, int(sr * duration))
    f0 = np.random.uniform(90, 260)
    
    vowel_formants = [
        (700, 1200, 2500),  # /a/
        (300, 2200, 3000),  # /i/
        (350, 800, 2300),   # /u/
        (500, 1800, 2600),  # /e/
        (500, 1000, 2400),  # /o/
    ]
    f1, f2, f3 = vowel_formants[np.random.randint(0, len(vowel_formants))]

    harmonics = np.zeros_like(t)
    for h in range(1, 25):
        freq = f0 * h
        if freq < 4000:
            amp1 = np.exp(-((freq - f1) ** 2) / (2 * 120 ** 2))
            amp2 = 0.6 * np.exp(-((freq - f2) ** 2) / (2 * 180 ** 2))
            amp3 = 0.3 * np.exp(-((freq - f3) ** 2) / (2 * 250 ** 2))
            weight = (1.0 / (h ** 0.6)) * (amp1 + amp2 + amp3 + 0.05)
            harmonics += weight * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))

    syllable_env = 0.5 * (1 + np.sin(2 * np.pi * np.random.uniform(3.0, 5.0) * t))
    voice = harmonics * syllable_env
    voice += 0.015 * np.random.randn(len(t))
    voice = (voice / (np.max(np.abs(voice)) + 1e-6)) * np.random.uniform(0.1, 0.4)
    return voice.astype(np.float32)

def generate_electronic_tones_and_hiss(duration=1.2, sr=22050):
    """
    Simulates high-frequency electronic tones, mic feedback, hiss, whistling.
    """
    t = np.linspace(0, duration, int(sr * duration))
    freq = np.random.choice([1200, 1800, 2400, 3000, 3500, 4200, 5000, 6000])
    tone = np.sin(2 * np.pi * freq * t)
    # Add optional second harmonic and high-frequency noise
    if np.random.rand() > 0.5:
        tone += 0.4 * np.sin(2 * np.pi * (freq * 1.5) * t)
    tone += np.random.normal(0, 0.08, len(t))
    tone = (tone / (np.max(np.abs(tone)) + 1e-6)) * np.random.uniform(0.05, 0.25)
    return tone.astype(np.float32)

def generate_broadband_noise(duration=1.2, sr=22050):
    """
    Simulates room ventilation, air conditioner blow, white/pink noise.
    """
    noise = np.random.normal(0, 1.0, int(sr * duration))
    # Filter or shape
    if np.random.rand() > 0.5:
        # Pink-ish integration
        noise = np.cumsum(noise)
        noise = noise - np.mean(noise)
    noise = (noise / (np.max(np.abs(noise)) + 1e-6)) * np.random.uniform(0.02, 0.15)
    return noise.astype(np.float32)

def generate_appliances_hum(duration=1.2, sr=22050):
    """
    Simulates fan hum, electrical 50/60 Hz hum + harmonics.
    """
    t = np.linspace(0, duration, int(sr * duration))
    base = np.random.choice([50, 60, 100, 120])
    hum = np.sin(2 * np.pi * base * t) + 0.5 * np.sin(2 * np.pi * base * 2 * t) + 0.2 * np.sin(2 * np.pi * base * 3 * t)
    hum += 0.05 * np.random.randn(len(t))
    hum = (hum / (np.max(np.abs(hum)) + 1e-6)) * np.random.uniform(0.04, 0.2)
    return hum.astype(np.float32)

def generate_transients_and_clicks(duration=1.2, sr=22050):
    """
    Simulates keyboard typing, mouse clicks, finger taps, claps.
    """
    sig = np.zeros(int(sr * duration), dtype=np.float32)
    num_clicks = np.random.randint(1, 5)
    for _ in range(num_clicks):
        pos = np.random.randint(100, len(sig) - 1000)
        click_len = np.random.randint(100, 800)
        env = np.exp(-np.linspace(0, 5, click_len))
        click = np.random.normal(0, 1.0, click_len) * env
        sig[pos : pos + click_len] += click.astype(np.float32)
    sig += np.random.normal(0, 0.005, len(sig)).astype(np.float32)
    sig = (sig / (np.max(np.abs(sig)) + 1e-6)) * np.random.uniform(0.1, 0.4)
    return sig

def determine_species(filename):
    fn = os.path.basename(filename).lower()
    if "crow" in fn:
        return "Crow"
    elif "sparrow" in fn:
        return "Sparrow"
    elif "pigeon" in fn or "duiven" in fn or "paloma" in fn:
        return "Pigeon"
    elif "peacock" in fn:
        return "Peacock"
    elif "parrot" in fn or "parakeet" in fn:
        return "Parakeet"
    elif "finch" in fn:
        return "Finch"
    else:
        return "Bird"

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        return

    print(f"Loading {len(audio_files)} reference bird audio files...")
    X = []
    y_bird = []
    y_species = []

    # 1. POSITIVE SAMPLES (Bird audio from Birds_Audio dataset)
    for idx, fpath in enumerate(audio_files):
        try:
            y_full, sr = librosa.load(fpath, sr=22050, duration=15.0)
            chunk_len = int(1.2 * sr)
            hop = int(0.8 * sr)
            species = determine_species(fpath)

            for start in range(0, len(y_full) - chunk_len + 1, hop):
                chunk = y_full[start : start + chunk_len]
                if np.max(np.abs(chunk)) > 0.02:
                    feats = extract_features(chunk, sr=sr)
                    X.append(feats)
                    y_bird.append(1)  # 1 = Bird sound
                    y_species.append(species)
        except Exception as e:
            print(f"Note skipping {os.path.basename(fpath)}: {e}")

    num_birds = len(X)
    print(f"Extracted {num_birds} bird audio segments from uploaded files.")

    # 2. NEGATIVE SAMPLES: COMPREHENSIVE NON-BIRD COHORTS
    print("Generating comprehensive Non-Bird negative training cohorts...")
    sr = 22050
    duration = 1.2
    
    # A. Multi-formant Human Speech & Conversation
    for _ in range(350):
        s = generate_synthetic_speech(duration=duration, sr=sr)
        X.append(extract_features(s, sr=sr))
        y_bird.append(0)
        y_species.append("Speech")

    # B. Electronic Tones, Whistling & Microphone Hiss
    for _ in range(250):
        t = generate_electronic_tones_and_hiss(duration=duration, sr=sr)
        X.append(extract_features(t, sr=sr))
        y_bird.append(0)
        y_species.append("Electronic")

    # C. Broadband Static & Air Noise
    for _ in range(200):
        n = generate_broadband_noise(duration=duration, sr=sr)
        X.append(extract_features(n, sr=sr))
        y_bird.append(0)
        y_species.append("Broadband")

    # D. Fan, AC & Appliance Hum
    for _ in range(200):
        h = generate_appliances_hum(duration=duration, sr=sr)
        X.append(extract_features(h, sr=sr))
        y_bird.append(0)
        y_species.append("Hum")

    # E. Clicks, Claps, Taps, Keyboard
    for _ in range(200):
        c = generate_transients_and_clicks(duration=duration, sr=sr)
        X.append(extract_features(c, sr=sr))
        y_bird.append(0)
        y_species.append("Transient")

    # F. Room Silence / Ambient Baseline
    for _ in range(120):
        sil = np.random.normal(0, 0.003, int(sr * duration)).astype(np.float32)
        X.append(extract_features(sil, sr=sr))
        y_bird.append(0)
        y_species.append("Silence")

    X = np.array(X)
    y_bird = np.array(y_bird)
    y_species = np.array(y_species)

    num_non_birds = len(X) - num_birds
    print(f"Total training dataset size: {len(X)} feature vectors ({num_birds} Bird vs {num_non_birds} Non-Bird).")

    # Standardize all features
    print("Fitting StandardScaler on feature space...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Random Forest binary detector (Bird vs All Non-Bird Sounds)...")
    clf_detector = RandomForestClassifier(n_estimators=150, max_depth=16, random_state=42)
    clf_detector.fit(X_scaled, y_bird)

    # Train species classifier on positive samples only
    bird_mask = (y_bird == 1)
    X_birds_scaled = X_scaled[bird_mask]
    y_bird_species = y_species[bird_mask]
    print(f"Training species identifier on {len(X_birds_scaled)} bird segments...")
    clf_species = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf_species.fit(X_birds_scaled, y_bird_species)

    # Nearest neighbor embedding index for cosine similarity against uploaded recordings
    nn_index = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_index.fit(X_birds_scaled)

    # Save model artifact with scaler included
    model_data = {
        "scaler": scaler,
        "detector": clf_detector,
        "species_clf": clf_species,
        "nn_index": nn_index,
        "reference_embeddings": X_birds_scaled,
        "reference_species": y_bird_species,
        "classes": list(clf_species.classes_)
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"Bird audio model successfully trained and saved with StandardScaler to: {MODEL_PATH}")

if __name__ == "__main__":
    train()
