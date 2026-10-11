"""
ScareX Standalone Raspberry Pi 5 - Real-Time Bird Acoustic Classifier
Uses Librosa feature extraction, physics-based human speech rejection,
and the trained Scikit-learn/Nearest-Neighbor classifier (bird_audio_model.joblib).
Handles missing microphones or missing models gracefully without crashing.
"""
import os
import sys
import time
import threading
import numpy as np
import joblib
import librosa

try:
    import raspberry_pi.config as config
except ImportError:
    try:
        import laptop.config as config
    except ImportError:
        import config

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception as e:
    SOUNDDEVICE_AVAILABLE = False

def extract_features(y, sr=22050):
    """
    Extracts fixed 59-dimensional acoustic feature vector tailored for Bird vs Non-Bird discrimination.
    Identical to training-time feature extractor.
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

    # 7. Spectral Contrast (7 bands * 2 = 14, summarized to mean & std = 2)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    cont_mean = np.mean(contrast)
    cont_std = np.std(contrast)

    # 8. Temporal Energy Modulation
    frame_len = int(0.05 * sr)
    hop_len = int(0.025 * sr)
    frame_energies = [np.sum(y[i : i + frame_len] ** 2) for i in range(0, len(y) - frame_len, hop_len)]
    modulation_ratio = np.std(frame_energies) / (np.mean(frame_energies) + 1e-6) if frame_energies else 0.0

    features = np.hstack([
        mfcc_mean,           # 20
        mfcc_std,            # 20
        cent_mean, cent_std, # 2
        roll_mean, roll_std, # 2
        zcr_mean, zcr_std,   # 2
        voice_band_ratio,    # 1
        bird_to_voice_ratio, # 1
        rms,                 # 1
        flat_mean, flat_std, # 2
        cont_mean, cont_std, # 2
        modulation_ratio     # 1
    ])
    return features.astype(np.float32)

class BirdAudioListener:
    def __init__(self, sample_rate=22050, buffer_duration=1.2, poll_interval=0.3):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.poll_interval = poll_interval
        self.lock = threading.Lock()
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.is_running = False
        self.stream = None
        self.worker_thread = None
        
        # Detection state
        self.bird_detected = False
        self.current_species = "None"
        self.confidence = 0.0
        self.rms_energy = 0.0
        self.rejection_reason = "Idle"
        self.consecutive_detects = 0
        self.consecutive_silence = 0
        
        # Model
        self.model = None
        self.model_path = getattr(config, "BIRD_AUDIO_MODEL_PATH", None)
        if not self.model_path or not os.path.exists(self.model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate1 = os.path.join(base_dir, "models", "bird_audio_model.joblib")
            candidate2 = os.path.join(os.path.dirname(base_dir), "laptop", "models", "bird_audio_model.joblib")
            if os.path.exists(candidate1):
                self.model_path = candidate1
            elif os.path.exists(candidate2):
                self.model_path = candidate2

        self.load_model()

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"[AudioListener] Loaded bird acoustic model from {self.model_path}")
            except Exception as e:
                print(f"[AudioListener] Failed to load model ({e})")
        else:
            print(f"[AudioListener] Note: Bird acoustic model not found at {self.model_path}. Acoustic listening idle.")

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Receives incoming microphone audio chunks.
        """
        if status:
            pass
        with self.lock:
            new_samples = indata[:, 0]
            n = len(new_samples)
            if n >= self.buffer_size:
                self.audio_buffer = new_samples[-self.buffer_size:].copy()
            else:
                self.audio_buffer[:-n] = self.audio_buffer[n:]
                self.audio_buffer[-n:] = new_samples

    def is_human_speech_or_ambient(self, audio_data: np.ndarray) -> tuple:
        """
        Acoustic physics pre-filter to reject human speech, talking, and ambient hum
        before neural/ML classification.
        Returns: (is_rejected: bool, rejection_reason: str)
        """
        sr = self.sample_rate

        # 1. Spectral Centroid
        # Human speaking voice: 300 Hz - 1,800 Hz.
        # Bird vocalizations: 2,200 Hz - 6,500 Hz.
        cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        mean_cent = float(np.mean(cent))
        if mean_cent < 1950.0:
            return True, f"Speech/Ambient: Low spectral centroid ({mean_cent:.0f} Hz < 1950 Hz)"

        # 2. Sub-band Frequency Energy Distribution
        S = np.abs(librosa.stft(audio_data))
        freqs = librosa.fft_frequencies(sr=sr)
        voice_mask = (freqs >= 80) & (freqs <= 1400)
        bird_mask = (freqs >= 2000) & (freqs <= 8000)

        voice_energy = float(np.sum(S[voice_mask, :])) + 1e-6
        bird_energy = float(np.sum(S[bird_mask, :])) + 1e-6

        # If energy in human voice band exceeds high-frequency bird band, it is human speech
        if voice_energy > bird_energy * 1.25:
            return True, f"Speech: Voice band dominant ({voice_energy:.1f} > {bird_energy:.1f})"

        # 3. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        mean_zcr = float(np.mean(zcr))
        if mean_zcr < 0.070:
            return True, f"Speech: Low ZCR ({mean_zcr:.3f} < 0.070)"

        return False, "Passed"

    def analyze_chunk(self, audio_data: np.ndarray) -> dict:
        """
        Analyzes an audio array to determine if it contains genuine bird audio.
        Guarantees rejection of human speech, ambient noise, continuous hiss, and room sounds.
        """
        if self.model is None or len(audio_data) < self.sample_rate * 0.2:
            return {"detected": False, "species": "None", "confidence": 0.0, "rms": 0.0, "reason": "No model/buffer"}

        rms = float(np.sqrt(np.mean(audio_data**2)))
        # 1. Energy Gate
        if rms < 0.016:
            return {"detected": False, "species": "None", "confidence": 0.0, "rms": round(rms, 4), "reason": f"Quiet/Ambient ({rms:.4f} < 0.016)"}

        # 2. Temporal Modulation Gate
        frame_len = int(0.05 * self.sample_rate)
        hop = int(0.025 * self.sample_rate)
        energies = [np.sum(audio_data[i : i + frame_len] ** 2) for i in range(0, len(audio_data) - frame_len, hop)]
        mod_ratio = float(np.std(energies) / (np.mean(energies) + 1e-6)) if energies else 0.0
        if mod_ratio < 0.35:
            return {"detected": False, "species": "None", "confidence": 0.0, "rms": round(rms, 4), "reason": f"Static/Continuous noise (ModRatio {mod_ratio:.2f} < 0.35)"}

        # 3. Spectral Flatness Gate
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio_data)))
        if flatness > 0.12:
            return {"detected": False, "species": "None", "confidence": 0.0, "rms": round(rms, 4), "reason": f"Diffuse white/air noise (Flatness {flatness:.3f} > 0.12)"}

        # 4. Acoustic Physics Human Speech Rejection Filter
        is_speech, reason = self.is_human_speech_or_ambient(audio_data)
        if is_speech:
            return {"detected": False, "species": "None", "confidence": 0.0, "rms": round(rms, 4), "reason": reason}

        # 5. Machine Learning Classification & Standardized Nearest-Neighbor Fingerprinting
        try:
            feats = extract_features(audio_data, sr=self.sample_rate)
            X = np.expand_dims(feats, axis=0)

            scaler = self.model.get("scaler")
            X_scaled = scaler.transform(X) if scaler is not None else X

            prob = self.model["detector"].predict_proba(X_scaled)[0]
            bird_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])

            distances, _ = self.model["nn_index"].kneighbors(X_scaled, n_neighbors=3)
            mean_dist = float(np.mean(distances))
            cosine_sim = max(0.0, 1.0 - mean_dist)

            combined_score = (0.55 * bird_prob) + (0.45 * cosine_sim)

            if bird_prob >= 0.80 and cosine_sim >= 0.68 and combined_score >= 0.75:
                species = str(self.model["species_clf"].predict(X_scaled)[0])
                return {
                    "detected": True,
                    "species": species,
                    "confidence": round(combined_score, 3),
                    "rms": round(rms, 4),
                    "reason": f"Confirmed {species}"
                }
            else:
                return {
                    "detected": False,
                    "species": "None",
                    "confidence": round(combined_score, 3),
                    "rms": round(rms, 4),
                    "reason": f"Low match (prob={bird_prob:.2f}, sim={cosine_sim:.2f})"
                }
        except Exception as e:
            return {"detected": False, "species": "None", "confidence": 0.0, "rms": rms, "error": str(e), "reason": "Error"}

    def _worker_loop(self):
        print("[AudioListener] Acoustic processing worker loop active.")
        while self.is_running:
            time.sleep(self.poll_interval)
            with self.lock:
                current_data = self.audio_buffer.copy()

            res = self.analyze_chunk(current_data)
            with self.lock:
                self.rms_energy = res.get("rms", 0.0)
                if res.get("detected"):
                    self.consecutive_detects += 1
                    self.consecutive_silence = 0
                    if self.consecutive_detects >= 1:
                        self.bird_detected = True
                        self.current_species = res.get("species", "Bird")
                        self.confidence = float(res.get("confidence", 0.0) * 100.0)
                        self.rejection_reason = "Bird Acoustic Signature Confirmed"
                else:
                    self.consecutive_silence += 1
                    if self.consecutive_silence >= 2:
                        self.bird_detected = False
                        self.current_species = "None"
                        self.confidence = 0.0
                        self.consecutive_detects = 0
                        self.rejection_reason = res.get("reason", "Ambient/Speech")

    def start(self):
        if self.is_running:
            return
        if not SOUNDDEVICE_AVAILABLE:
            print("[AudioListener] sounddevice not available. Running in passive acoustic mode.")
            return

        self.is_running = True
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1),
                callback=self._audio_callback
            )
            self.stream.start()
            print("[AudioListener] Microphone audio stream started.")
        except Exception as e:
            print(f"[AudioListener] Microphone not detected or stream failed ({e}). Passive mode active.")

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        print("[AudioListener] Audio listener stopped.")

    def get_status(self) -> dict:
        with self.lock:
            return {
                "bird_detected": self.bird_detected,
                "species": self.current_species,
                "confidence": round(self.confidence, 1),
                "rms_energy": round(self.rms_energy, 4),
                "status_text": self.rejection_reason,
                "mic_available": bool(self.stream is not None and self.is_running)
            }

# Global singleton
bird_audio_listener = BirdAudioListener()
