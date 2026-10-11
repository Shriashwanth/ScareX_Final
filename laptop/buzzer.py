"""
Laptop Deterrence Audio Alert Module for ScareX System
Manages continuous deterrence sound when a bird is present on camera,
and provides instantaneous stopping when the bird is removed.
For tomatoes or clear views, sound remains completely silent.
Supports Pygame Mixer audio with native Windows winsound fallback.
"""
import os
import time
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "assets", "audio")
DETERRENT_DIR = os.path.join(BASE_DIR, "assets", "deterrent_sounds")
BUZZER_WAV = os.path.join(AUDIO_DIR, "deterrent_1.wav")

# The 5 Uploaded Bioacoustic Predator Deterrence Sounds
DETERRENT_SOUNDS = [
    {"id": 1, "name": "Monster Roar", "file": "deterrent_1.wav", "orig": "audiopapkin-monster-roar-ps-oo6-306402.mp3"},
    {"id": 2, "name": "Monster Warrior Roar", "file": "deterrent_2.wav", "orig": "dffdv-monster-warrior-roar-195877.mp3"},
    {"id": 3, "name": "Epic Dragon Roar", "file": "deterrent_3.wav", "orig": "dragon-studio-epic-dragon-roar-364481.mp3"},
    {"id": 4, "name": "Spooky Wolf Howl", "file": "deterrent_4.wav", "orig": "dragon-studio-spooky-wolf-howl-410547.mp3"},
    {"id": 5, "name": "Beast Screaming", "file": "deterrent_5.wav", "orig": "freesound_community-beast-human-screaming-86831.mp3"}
]

# Species specific mapping + dynamic alternating mode
SPECIES_AUDIO_MAP = {
    "crow": "deterrent_3.wav",                 # Epic Dragon Roar
    "pigeon": "deterrent_4.wav",               # Wolf Howl
    "sparrow": "deterrent_1.wav",              # Monster Roar
    "common myna": "deterrent_2.wav",          # Warrior Roar
    "myna": "deterrent_2.wav",
    "rose ringed parakeet": "deterrent_5.wav", # Beast Scream
    "parakeet": "deterrent_5.wav",
    "parrot": "deterrent_5.wav",
    "peacock": "deterrent_2.wav",              # Warrior Roar
    "bird": "alternate",                       # Alternate across the 5 sounds
    "default": "alternate"
}

try:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception as e:
    PYGAME_AVAILABLE = False
    print(f"[AudioAlert] Note: pygame.mixer unavailable ({e}), using system audio fallback.")

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

class DeterrenceSoundController:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_active = False
        self.current_sound_name = None
        self.current_species = None
        self.active_channel = None
        self.loaded_sounds = {}
        self.sound_index = 0
        self.backend = "pygame" if PYGAME_AVAILABLE else ("winsound" if WINSOUND_AVAILABLE else "none")
        print(f"[AudioAlert] Controller initialized using backend: {self.backend} with 5 predator deterrence sounds.")

    def _get_audio_path(self, species: str) -> str:
        clean = str(species).strip().lower()
        mapping = SPECIES_AUDIO_MAP.get(clean, "alternate")
        
        # When "alternate" or general bird is triggered, rotate to the next alternative sound
        if mapping == "alternate" or clean in ["bird", "default", ""]:
            sound_entry = DETERRENT_SOUNDS[self.sound_index % len(DETERRENT_SOUNDS)]
            self.sound_index += 1
            filename = sound_entry["file"]
        else:
            filename = mapping

        path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(path):
            path = os.path.join(DETERRENT_DIR, filename)
        if not os.path.exists(path):
            path = BUZZER_WAV
        return path

    def start_deterrence(self, species: str = "bird"):
        """
        Activates deterrence sound in a continuous loop while the bird is shown.
        If already playing for this species, continues seamlessly without interruption.
        """
        with self.lock:
            if self.is_active and self.current_species == species:
                # Already playing deterrence sound for this species
                return

            sound_path = self._get_audio_path(species)
            sound_name = os.path.basename(sound_path) if sound_path else "buzzer"
            self.current_sound_name = sound_name
            self.current_species = species
            self.is_active = True

            print(f"[AudioAlert] >>> DETERRENCE SOUND ACTIVATED: Playing '{sound_name}' for '{species}' (LOOPING) <<<")

            if PYGAME_AVAILABLE and sound_path and os.path.exists(sound_path):
                try:
                    if sound_path not in self.loaded_sounds:
                        self.loaded_sounds[sound_path] = pygame.mixer.Sound(sound_path)
                    sound_obj = self.loaded_sounds[sound_path]
                    if self.active_channel and self.active_channel.get_busy():
                        self.active_channel.stop()
                    self.active_channel = sound_obj.play(loops=-1)
                    return
                except Exception as e:
                    print(f"[AudioAlert] Pygame playback error ({e}), falling back to winsound.")

            # Fallback to winsound on Windows
            if WINSOUND_AVAILABLE:
                def _win_play():
                    try:
                        if sound_path and os.path.exists(sound_path):
                            winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                        else:
                            # Periodic beep while active
                            while self.is_active:
                                winsound.Beep(1200, 300)
                                time.sleep(0.1)
                    except Exception as err:
                        print(f"[AudioAlert] Winsound error: {err}")

                threading.Thread(target=_win_play, daemon=True).start()

    def stop_deterrence(self):
        """
        Instantly halts deterrence sound when the bird is removed.
        Guarantees complete silence for tomatoes or clear camera views.
        """
        with self.lock:
            if not self.is_active:
                return

            print(f"[AudioAlert] >>> DETERRENCE SOUND STOPPED (Bird Removed / No Bird) <<<")
            self.is_active = False
            self.current_sound_name = None
            self.current_species = None

            # Stop Pygame channel
            if PYGAME_AVAILABLE:
                try:
                    if self.active_channel:
                        self.active_channel.stop()
                    pygame.mixer.stop()
                except Exception as e:
                    pass

            # Stop Windows winsound
            if WINSOUND_AVAILABLE:
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass

    def play_once(self, sound_path=None):
        """
        Plays a single pulse alert for manual dashboard testing.
        """
        def _run():
            target_path = sound_path or BUZZER_WAV
            if PYGAME_AVAILABLE and os.path.exists(target_path):
                try:
                    snd = pygame.mixer.Sound(target_path)
                    snd.play()
                    return
                except Exception:
                    pass
            if WINSOUND_AVAILABLE:
                try:
                    if os.path.exists(target_path):
                        winsound.PlaySound(target_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    else:
                        winsound.Beep(1200, 400)
                except Exception:
                    pass
        threading.Thread(target=_run, daemon=True).start()

    def is_playing(self) -> bool:
        with self.lock:
            return self.is_active

    def get_status(self) -> dict:
        with self.lock:
            return {
                "active": self.is_active,
                "current_sound": self.current_sound_name if self.is_active else "None",
                "species": self.current_species if self.is_active else "None"
            }

# Singleton controller instance
audio_alert = DeterrenceSoundController()
sound_controller = audio_alert

# Convenience functional interface
def start_deterrence(species: str = "bird"):
    audio_alert.start_deterrence(species)

def stop_deterrence():
    audio_alert.stop_deterrence()

def play_buzzer():
    audio_alert.play_once()

def is_playing() -> bool:
    return audio_alert.is_playing()

if __name__ == "__main__":
    print("Testing start_deterrence('bird')...")
    start_deterrence("bird")
    time.sleep(1.5)
    print("Testing stop_deterrence()...")
    stop_deterrence()
    time.sleep(0.5)
    print("Deterrence sound test complete.")
