"""
Decision & Fusion Engine for ScareX Integrated System
Combines simultaneous Bird (Vision + Microphone Audio) and Tomato Crop Maturity Detections.
- When a BIRD is detected (on CAMERA or heard via MICROPHONE from Birds_Audio):
  produces continuous targeted deterrence sound.
- When the BIRD is removed / bird sound stops:
  immediately stops deterrence sound.
- For TOMATOES: strictly silent (no deterrence sound), computes agronomic maturity analytics.
"""
import time
import threading
import json
import laptop.config as config
from laptop.communication.pi_client import pi_client
from laptop.database.db import db
from laptop.models.tomato.detector import TomatoDetector

try:
    from laptop.buzzer import start_deterrence, stop_deterrence, is_playing, play_buzzer
except ImportError:
    from buzzer import start_deterrence, stop_deterrence, is_playing, play_buzzer

try:
    from laptop.audio.bird_audio_listener import bird_audio_listener
except ImportError:
    try:
        from audio.bird_audio_listener import bird_audio_listener
    except ImportError:
        bird_audio_listener = None

def compute_box_iou(box1, box2):
    """
    Computes Intersection over Union between two [x1, y1, x2, y2] bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

class DecisionEngine:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Bird detection confirmation & state tracking
        self.bird_present_frames = 0
        self.bird_absent_frames = 0
        self.last_scare_time = 0.0
        self.current_bird = "No Bird Detected"
        self.current_bird_conf = 0.0
        self.detection_source = "None"
        self.scare_active = False
        
        # Tomato monitoring state (never triggers sound)
        self.tomato_counts = {"fully_ripened": 0, "half_ripened": 0, "green": 0}
        self.max_session_tomatoes = {"fully_ripened": 0, "half_ripened": 0, "green": 0}
        self.harvest_priority = "STANDBY"
        self.row_analysis = []
        self.total_frames_processed = 0
        self.last_tomato_sync_time = 0.0

    def process_detections(self, bird_dets: list, tomato_dets: list):
        """
        Processes multi-modal detections (Camera Vision + Microphone Audio + Tomato Crop Monitoring).
        - Bird detected via Camera OR Microphone -> start / sustain continuous deterrence sound.
        - Bird removed / sound stops -> immediately stop deterrence sound.
        - Tomato alone -> completely silent, update crop maturity analytics.
        """
        now = time.time()
        with self.lock:
            self.total_frames_processed += 1

            # -----------------------------------------------------------------
            # 1. TOMATO SUPPRESSION FILTER (Guarantees NO sound for tomatoes)
            # -----------------------------------------------------------------
            valid_birds = []
            for b in (bird_dets or []):
                b_box = b.get("bbox", [0, 0, 0, 0])
                is_tomato_overlap = False
                for t in (tomato_dets or []):
                    t_box = t.get("bbox", [0, 0, 0, 0])
                    iou = compute_box_iou(b_box, t_box)
                    if iou > 0.30:
                        # Overlaps significantly with tomato -> treat as tomato, reject bird
                        is_tomato_overlap = True
                        break
                if not is_tomato_overlap:
                    valid_birds.append(b)

            # -----------------------------------------------------------------
            # 2. MICROPHONE ACOUSTIC DETECTION CHECK
            # -----------------------------------------------------------------
            audio_status = bird_audio_listener.get_status() if bird_audio_listener else {"bird_detected": False}
            heard_bird = audio_status.get("bird_detected", False)
            audio_species = audio_status.get("species", "Bird")
            audio_conf = (audio_status.get("confidence", 0.0)) / 100.0

            # Multi-modal presence: camera vision OR microphone acoustic detection
            bird_present = bool(valid_birds or heard_bird)

            # -----------------------------------------------------------------
            # 3. BIRD PROCESSING (Safety & Sound Deterrence)
            # -----------------------------------------------------------------
            if bird_present:
                if valid_birds:
                    best_bird = max(valid_birds, key=lambda d: d["confidence"])
                    self.current_bird = best_bird["label"]
                    self.current_bird_conf = best_bird["confidence"]
                    self.detection_source = "Camera + Mic" if heard_bird else "Camera Vision"
                else:
                    self.current_bird = audio_species
                    self.current_bird_conf = audio_conf
                    self.detection_source = "Microphone (Acoustic)"

                self.bird_present_frames += 1
                self.bird_absent_frames = 0

                # Fast debounce to confirm bird presence
                if self.bird_present_frames >= 2 or heard_bird:
                    if not self.scare_active:
                        print(f"[Fusion] BIRD DETECTED via {self.detection_source}: '{self.current_bird}' (Conf: {self.current_bird_conf*100:.1f}%) -> ACTIVATING DETERRENCE SOUND")
                        self.scare_active = True
                        self.last_scare_time = now

                        # 1. Start continuous audio alert on Laptop
                        start_deterrence(self.current_bird)

                        # 2. Dispatch non-blocking command to Raspberry Pi edge node
                        threading.Thread(
                            target=self._start_pi_scare,
                            args=(self.current_bird, self.current_bird_conf),
                            daemon=True
                        ).start()

                        # 3. Log activation to database
                        db.log_event(
                            event_type="BIRD_DETERRENCE_START",
                            species=self.current_bird,
                            bird_confidence=self.current_bird_conf,
                            robot_action=f"DETERRENCE_ACTIVE_{self.detection_source.upper()}"
                        )
                    else:
                        # Bird is still present / heard -> ensure deterrence sound is looping
                        if not is_playing():
                            start_deterrence(self.current_bird)

            else:
                # -------------------------------------------------------------
                # 4. NO BIRD / BIRD REMOVED / SOUND STOPPED (Immediate cutoff)
                # -------------------------------------------------------------
                self.bird_absent_frames += 1
                self.bird_present_frames = 0

                # When bird is absent/silent for 2 frames, turn off sound immediately
                if self.bird_absent_frames >= 2:
                    self.current_bird = "No Bird Detected"
                    self.current_bird_conf = 0.0
                    self.detection_source = "None"

                    if self.scare_active:
                        print("[Fusion] BIRD REMOVED / AUDIO SILENT: Stopping deterrence sound immediately.")
                        self.scare_active = False

                        # 1. Stop audio alert on Laptop immediately
                        stop_deterrence()

                        # 2. Dispatch stop command to Raspberry Pi
                        threading.Thread(
                            target=self._stop_pi_scare,
                            daemon=True
                        ).start()

                        # 3. Log event to database
                        db.log_event(
                            event_type="BIRD_REMOVED",
                            species="None",
                            bird_confidence=0.0,
                            robot_action="DETERRENCE_SOUND_STOPPED"
                        )
                    else:
                        # Ensure sound is completely silent
                        if is_playing():
                            stop_deterrence()

            # -----------------------------------------------------------------
            # 5. TOMATO PROCESSING (Crop Monitoring & Analytics - STRICTLY SILENT)
            # -----------------------------------------------------------------
            frame_counts = {"fully_ripened": 0, "half_ripened": 0, "green": 0}
            for t in (tomato_dets or []):
                lbl = t["label"]
                if lbl in frame_counts:
                    frame_counts[lbl] += 1
                else:
                    frame_counts["green"] += 1

            self.tomato_counts = frame_counts

            # Update session max-observed counts
            for k in ["fully_ripened", "half_ripened", "green"]:
                if frame_counts[k] > self.max_session_tomatoes[k]:
                    self.max_session_tomatoes[k] = frame_counts[k]

            # Compute row-wise clustering if tomatoes present
            if tomato_dets:
                self.row_analysis = TomatoDetector.group_into_rows(tomato_dets)
                
                total_tomatoes = sum(self.max_session_tomatoes.values())
                if total_tomatoes > 0:
                    ripe_ratio = self.max_session_tomatoes["fully_ripened"] / total_tomatoes
                    if ripe_ratio >= 0.50:
                        self.harvest_priority = "HIGH"
                    elif ripe_ratio >= 0.25:
                        self.harvest_priority = "MEDIUM"
                    else:
                        self.harvest_priority = "LOW"

            # Periodically sync tomato telemetry to Pi (every 5 seconds)
            if now - self.last_tomato_sync_time > 5.0 and tomato_dets:
                self.last_tomato_sync_time = now
                threading.Thread(
                    target=pi_client.send_tomato_update,
                    args=(tomato_dets,),
                    daemon=True
                ).start()

    def _start_pi_scare(self, species: str, confidence: float):
        """
        Dispatches start deterrence command to Raspberry Pi node.
        """
        try:
            res = pi_client.trigger_bird_deterrence(species, confidence)
            print(f"[PiClient] Deterrence active response: {res.get('action', 'OK')}")
        except Exception as e:
            print(f"[PiClient] Deterrence dispatch note: {e}")

    def _stop_pi_scare(self):
        """
        Dispatches stop deterrence command to Raspberry Pi node.
        """
        try:
            res = pi_client.stop_bird_deterrence()
            print(f"[PiClient] Deterrence stopped response: {res.get('action', 'OK')}")
        except Exception as e:
            print(f"[PiClient] Deterrence stop note: {e}")

    def get_status(self) -> dict:
        """
        Returns snapshot of current state for the unified dashboard.
        """
        with self.lock:
            total_max = sum(self.max_session_tomatoes.values())
            
            green_pct = round((self.max_session_tomatoes["green"] / total_max * 100), 1) if total_max > 0 else 0
            half_pct = round((self.max_session_tomatoes["half_ripened"] / total_max * 100), 1) if total_max > 0 else 0
            full_pct = round((self.max_session_tomatoes["fully_ripened"] / total_max * 100), 1) if total_max > 0 else 0

            audio_stat = bird_audio_listener.get_status() if bird_audio_listener else {
                "bird_detected": False, "species": "None", "confidence": 0.0, "rms_energy": 0.0
            }

            return {
                "bird": {
                    "species": self.current_bird,
                    "confidence": round(self.current_bird_conf * 100, 1),
                    "source": self.detection_source,
                    "consecutive_frames": self.bird_present_frames,
                    "confirmation_target": 2,
                    "scare_active": self.scare_active,
                    "cooldown_remaining_sec": 0.0,
                    "acoustic_listener": audio_stat
                },
                "tomato": {
                    "current_frame_counts": self.tomato_counts,
                    "session_max_counts": self.max_session_tomatoes,
                    "total_observed": total_max,
                    "green_pct": green_pct,
                    "half_pct": half_pct,
                    "full_pct": full_pct,
                    "harvest_priority": self.harvest_priority,
                    "row_analysis": self.row_analysis
                },
                "system": {
                    "frames_processed": self.total_frames_processed
                }
            }

decision_engine = DecisionEngine()
