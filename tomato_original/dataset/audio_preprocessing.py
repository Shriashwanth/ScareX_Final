import os
import imageio_ffmpeg
# Make FFmpeg available to librosa for decoding mp3s
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

import numpy as np
import librosa
import glob

def normalize_feature(feature_matrix):
    """Normalize a feature matrix to range [0, 1]."""
    min_val = np.min(feature_matrix)
    max_val = np.max(feature_matrix)
    if max_val - min_val == 0:
        return np.zeros_like(feature_matrix)
    return (feature_matrix - min_val) / (max_val - min_val)

def extract_features(audio_path, sr=16000):
    """
    Extract all requested audio features for research/training.
    All features are normalized.
    """
    try:
        # Load audio (mono, 16kHz)
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        
        # 1. Mel Spectrogram
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spect, ref=np.max)
        mel_norm = normalize_feature(mel_db)
        
        # 2. MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_norm = normalize_feature(mfcc)
        
        # 3. Spectral Contrast
        stft = np.abs(librosa.stft(y))
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        spectral_contrast_norm = normalize_feature(spectral_contrast)
        
        # 4. Chroma (Pitch class profile)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_norm = normalize_feature(chroma)
        
        # 5. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_norm = normalize_feature(zcr)
        
        # 6. RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_norm = normalize_feature(rms)
        
        # 7. Tonnetz (Tonal centroid features)
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        tonnetz_norm = normalize_feature(tonnetz)
        
        return {
            "mel": mel_norm,
            "mfcc": mfcc_norm,
            "spectral_contrast": spectral_contrast_norm,
            "chroma": chroma_norm,
            "zcr": zcr_norm,
            "rms": rms_norm,
            "tonnetz": tonnetz_norm
        }
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def process_directory(input_dir, output_dir):
    """
    Process all .wav files in a directory and save their features.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    audio_files.extend(glob.glob(os.path.join(input_dir, "**/*.mp3"), recursive=True))
    
    if not audio_files:
        print(f"No .wav or .mp3 files found in {input_dir}")
        return
        
    for audio_file in audio_files:
        print(f"Processing {audio_file}...")
        features = extract_features(audio_file)
        if features:
            # Save features as .npz
            rel_path = os.path.relpath(audio_file, input_dir)
            out_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + "_features.npz")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            
            np.savez_compressed(out_file, **features)
            
    print(f"Feature extraction complete. Saved to {output_dir}")

if __name__ == "__main__":
    # Example usage targeting the audio training dataset directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ScareX_Dataset")
    input_audio = os.path.join(base_dir, "audio", "train")
    output_features = os.path.join(base_dir, "audio_features", "train")
    
    # process_directory(input_audio, output_features)
    print("Audio preprocessing script ready. Uncomment process_directory() to run on actual data.")
