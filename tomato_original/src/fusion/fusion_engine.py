import threading
import time
from src.core.logger import logger
from src.logging.db_logger import db_logger
from src.core.config_manager import ConfigManager
from src.hardware.hardware import hardware_controller
from src.alerts.alert_manager import alert_manager

class FusionEngine:
    def __init__(self, vision_module, audio_module):
        self.vision = vision_module
        self.audio = audio_module
        self.config = ConfigManager()
        
        self.is_running = False
        self.thread = None
        self.last_scare_time = 0
        self.scare_active = False
        self.consecutive_detections = 0

    def _engine_loop(self):
        logger.info("Fusion engine started.")
        while self.is_running:
            v_bird, v_conf = self.vision.get_latest_detection()
            a_bird, a_conf = self.audio.get_latest_detection()

            current_time = time.time()
            deterrent_cooldown = self.config.get('deterrent_cooldown', 10.0)
            required_consecutive = self.config.get('required_consecutive_detections', 3)
            
            # Check camera vision threshold
            v_valid = (v_bird != "No Bird Detected" and v_conf >= self.config.get('camera_confidence', 0.6))

            if not self.scare_active and (current_time - self.last_scare_time) > deterrent_cooldown:
                if v_valid:
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0

                if self.consecutive_detections >= required_consecutive:
                    logger.info(f"Camera visually detected bird: {v_bird} ({v_conf*100:.1f}% confidence). Triggering deterrence sequence...")
                    self.trigger_scare(v_bird, v_conf, "CAMERA")
                    self.consecutive_detections = 0
            else:
                self.consecutive_detections = 0

            time.sleep(0.5)

    def trigger_scare(self, bird_name, confidence, source="UNKNOWN"):
        self.scare_active = True
        self.last_scare_time = time.time()
        scare_duration = self.config.get('scare_duration_sec', 10.0)
        
        # Play alarm
        alarm_played = alert_manager.play_alert(bird_name)
        
        # Hardware action (e.g. flap wings)
        threading.Thread(target=hardware_controller.execute_scare_sequence).start()
        
        # Log to DB
        db_logger.log_event("SCARE_ACTIVATED", species=bird_name, confidence=confidence, alarm_played=alarm_played, system_status="SCARE", detection_source=source)

        # Schedule deactivate
        threading.Timer(scare_duration, self.deactivate_scare).start()

    def deactivate_scare(self):
        alert_manager.stop()
        hardware_controller.stop()
        self.scare_active = False
        logger.info("Scare mode deactivated. System ready.")

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._engine_loop, daemon=True)
            self.thread.start()
            db_logger.log_event("SYSTEM_START", system_status="RUNNING")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.deactivate_scare()
        logger.info("Fusion engine stopped.")
        db_logger.log_event("SYSTEM_STOP", system_status="STOPPED")
