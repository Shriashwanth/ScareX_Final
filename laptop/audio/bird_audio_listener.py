"""
Real-Time Microphone Bird Audio Recognition Listener for ScareX
Continuously listens via microphone using sounddevice.
Extracts spectral features and matches against the Birds_Audio trained model.
Includes acoustic physics speech rejection filters (spectral centroid, energy ratio, ZCR)
to strictly prevent human speaking, talking, coughing, or room noise from triggering false alarms.
"""
import os
import sys
import time
import threading
import numpy as np
import joblib
import librosa

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception as e:
    SOUNDDEVICE_AVAILABLE = False
    print(f"[AudioListener] sounddevice not available ({e}).")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
for p in [ROOT_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from laptop.audio.train_bird_audio_model import extract_features
except ImportError:
    from train_bird_audio_model import extract_features

MODEL_PATH = os.path.join(BASE_DIR, "models", "bird_audio_model.joblib")

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
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                print(f"[AudioListener] Loaded bird acoustic model from {MODEL_PATH}")
            except Exception as e:
                print(f"[AudioListener] Failed to load model ({e})")
        else:
            print(f"[AudioListener] Model not found at {MODEL_PATH}. Run train_bird_audio_model.py first.")

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
        # Bird vocalizations (Crow, Sparrow, Parrot, Peacock): 2,200 Hz - 6,500 Hz.
        cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        mean_cent = float(np.mean(cent))
        if mean_cent < 1950.0:
            return True, f"Speech/Ambient: Low spectral centroid ({mean_cent:.0f} Hz < 1950 Hz)"

        # 2. Sub-band Frequency Energy Distribution
        # Human speech formants dominate below 1,400 Hz.
        # Bird chirps dominate between 2,000 Hz and 8,000 Hz.
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
        # Voiced speech has low ZCR (< 0.075); bird calls have high ZCR (> 0.10)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        mean_zcr = float(np.mean(zcr))
        if mean_zcr < 0.070:
            return True, f"Speech: Low ZCR ({mean_zcr:.3f} < 0.070)"

        return False, "Passed"

    def analyze_chunk(self, audio_data: np.ndarray) -> dict:
        """
        Analyzes an audio array to determine if it contains genuine bird audio from the uploaded dataset.
        Guarantees rejection of human speech, ambient noise, continuous hiss, fans, and indoor sounds.
        """
        if self.model is None or len(audio_data) < self.sample_rate * 0.2:
            return {"detected": False, "species": "None", "confidence": 0.0, "rms": 0.0, "reason": "No model/buffer"}

        rms = float(np.sqrt(np.mean(audio_data**2)))
        # 1. Energy Gate: reject silence and very low-level background baseline (ambient breathing/hum)
        if rms < 0.016:
            return {"detected": False, "species": "None", "confidence": 0.0, "rms": round(rms, 4), "reason": f"Quiet/Ambient ({rms:.4f} < 0.016)"}

        # 2. Temporal Modulation Gate: reject static/continuous noise (AC hum, whistling, constant hiss)
        # Bird vocalizations are bursty with high frame-to-frame variance (ModRatio >= 0.35)
        frame_len = int(0.05 * self.sample_rate)
        hop = int(0.025 * self.sample_rate)
        energies = [np.sum(audio_data[i : i + frame_len] ** 2) for i in range(0, len(audio_data) - frame_len, hop)]
        mod_ratio = float(np.std(energies) / (np.mean(energies) + 1e-6)) if energies else 0.0
        if mod_ratio < 0.35:
            return {"detected": False, "species": "None", "confidence": 0.0, "rms": round(rms, 4), "reason": f"Static/Continuous noise (ModRatio {mod_ratio:.2f} < 0.35)"}

        # 3. Spectral Flatness Gate: reject broadband diffuse noise (white noise, air rush, fan blow)
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

            # Standardize features using the model's fitted StandardScaler
            scaler = self.model.get("scaler")
            X_scaled = scaler.transform(X) if scaler is not None else X

            # Binary Bird vs Non-Bird detector prediction
            prob = self.model["detector"].predict_proba(X_scaled)[0]
            bird_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])

            # Nearest-Neighbor cosine similarity against uploaded Birds_Audio reference set
            distances, _ = self.model["nn_index"].kneighbors(X_scaled, n_neighbors=3)
            mean_dist = float(np.mean(distances))
            cosine_sim = max(0.0, 1.0 - mean_dist)

            # Combined score
            combined_score = (0.55 * bird_prob) + (0.45 * cosine_sim)

            # Strict thresholding: must pass both detector probability and reference template similarity
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

    def _listen_loop(self):
        """
        Background loop polling audio buffer periodically with temporal debounce.
        """
        while self.is_running:
            with self.lock:
                chunk = self.audio_buffer.copy()

            res = self.analyze_chunk(chunk)
            with self.lock:
                self.rms_energy = res["rms"]
                self.rejection_reason = res.get("reason", "OK")
                
                if res["detected"]:
                    self.consecutive_detects += 1
                    self.consecutive_silence = 0
                    # Require 2 consecutive windows (~0.6s of sustained bird sound)
                    if self.consecutive_detects >= 2:
                        self.bird_detected = True
                        self.current_species = res["species"]
                        self.confidence = res["confidence"]
                else:
                    self.consecutive_silence += 1
                    self.consecutive_detects = 0
                    # Immediate cutoff when sound stops (2 silence cycles ~0.6s)
                    if self.consecutive_silence >= 2:
                        self.bird_detected = False
                        self.current_species = "None"
                        self.confidence = 0.0

            time.sleep(self.poll_interval)

    def start(self):
        """
        Starts live microphone streaming and background analysis.
        """
        if self.is_running:
            return
        self.is_running = True

        if SOUNDDEVICE_AVAILABLE:
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    callback=self._audio_callback,
                    blocksize=int(self.sample_rate * 0.1)
                )
                self.stream.start()
                print("[AudioListener] Microphone stream started successfully with Speech Rejection Filters active.")
            except Exception as e:
                print(f"[AudioListener] Error opening microphone stream ({e}).")
        else:
            print("[AudioListener] Running in listener mode without live microphone hardware.")

        self.worker_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """
        Stops the microphone stream and worker thread.
        """
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        print("[AudioListener] Microphone stream stopped.")

    def get_status(self) -> dict:
        """
        Returns snapshot of current acoustic detection status.
        """
        with self.lock:
            return {
                "bird_detected": self.bird_detected,
                "species": self.current_species,
                "confidence": round(self.confidence * 100, 1),
                "rms_energy": round(self.rms_energy, 4),
                "status_reason": self.rejection_reason
            }

# Singleton audio listener instance
bird_audio_listener = BirdAudioListener()

if __name__ == "__main__":
    print("Testing BirdAudioListener speech rejection and bird detection...")
    listener = BirdAudioListener()
    sample_file = os.path.join(BASE_DIR, "assets", "birds_audio", "freesound_community-crow-67929.mp3")
    if os.path.exists(sample_file):
        y, sr = librosa.load(sample_file, sr=22050, duration=2.0)
        res = listener.analyze_chunk(y)
        print(f"Sample test result for real crow audio: {res}")
    else:
        print("Sample file not found for test.")
