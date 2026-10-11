"""
Raspberry Pi Audio Player
Plays species-specific bird deterrent audio files locally on the Pi speaker.
Supports continuous deterrence looping while a bird is detected on camera,
and instant stopping when the bird is removed.
"""
import os
import time
import threading
import raspberry_pi.config as config

try:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    AUDIO_AVAILABLE = True
except Exception as e:
    print(f"[Audio] Pygame mixer init note ({e}). Audio will be logged if audio device is absent.")
    AUDIO_AVAILABLE = False

class AudioPlayer:
    def __init__(self):
        self.audio_dir = config.AUDIO_DIR
        self.volume = config.SPEAKER_VOLUME
        self.last_play_time = 0.0
        self.current_sound = None
        self.is_looping = False
        self.sound_index = 0
        self.lock = threading.Lock()

        if AUDIO_AVAILABLE:
            try:
                pygame.mixer.music.set_volume(self.volume)
            except:
                pass

    def play_deterrent(self, species: str, loop: bool = True) -> dict:
        """
        Plays the targeted deterrent audio for the specified bird species in a loop.
        Rotates across the 5 uploaded predator sounds when in alternating mode.
        Returns status dict: {"success": bool, "file_played": str, "message": str}
        """
        now = time.time()
        with self.lock:
            clean_species = str(species).strip().lower()
            mapping = config.SPECIES_AUDIO_MAP.get(clean_species, "alternate")

            if mapping == "alternate" or clean_species in ["bird", "default", ""]:
                sound_entry = config.DETERRENT_SOUNDS[self.sound_index % len(config.DETERRENT_SOUNDS)]
                self.sound_index += 1
                filename = sound_entry["file"]
            else:
                filename = mapping

            file_path = os.path.join(self.audio_dir, filename)
            if not os.path.exists(file_path) and hasattr(config, "DETERRENT_DIR"):
                file_path = os.path.join(config.DETERRENT_DIR, filename)

            # Fallback to default.wav or buzzer.wav if specific file is missing
            if not os.path.exists(file_path):
                file_path = os.path.join(self.audio_dir, "default.wav")
            if not os.path.exists(file_path):
                file_path = os.path.join(self.audio_dir, "buzzer.wav")

            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "file_played": None,
                    "message": f"Audio file not found: {filename}"
                }

            basename = os.path.basename(file_path)

            # If already looping this exact sound, keep playing smoothly
            if AUDIO_AVAILABLE and self.is_looping and self.current_sound == basename:
                try:
                    if pygame.mixer.music.get_busy():
                        return {
                            "success": True,
                            "file_played": basename,
                            "message": f"Already looping {basename} for {species}"
                        }
                except Exception:
                    pass

            print(f"[Audio] Playing deterrent for '{species}': {basename} (loop={loop})")
            self.last_play_time = now
            self.current_sound = basename
            self.is_looping = loop

            if AUDIO_AVAILABLE:
                try:
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.play(loops=-1 if loop else 0)
                except Exception as e:
                    print(f"[Audio] Playback error: {e}")

            return {
                "success": True,
                "file_played": basename,
                "message": f"Playing {basename} for {species}"
            }

    def stop(self):
        """
        Immediately stops all audio playback when bird is removed or safe state requested.
        """
        with self.lock:
            if AUDIO_AVAILABLE:
                try:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                except Exception:
                    pass
            self.is_looping = False
            self.current_sound = None
            print("[Audio] Deterrence playback stopped.")

    def get_status(self):
        busy = False
        if AUDIO_AVAILABLE:
            try:
                busy = pygame.mixer.music.get_busy()
            except:
                busy = False
        return {
            "playing": busy,
            "current_sound": self.current_sound if busy else "None",
            "last_played": self.current_sound
        }

audio_player = AudioPlayer()
