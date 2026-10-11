"""
Laptop to Raspberry Pi Communication Client
Dispatches HTTP commands to http://PI_IP:9000/command.
Runs an automatic heartbeat loop to keep the Pi watchdog healthy.
Fetches hardware telemetry from http://PI_IP:9000/status.
"""
import time
import threading
import requests
try:
    import laptop.config as config
except ImportError:
    import config

class PiClient:
    def __init__(self, command_url=None, status_url=None):
        self.command_url = command_url or config.COMMAND_SERVER_URL
        self.status_url = status_url or config.STATUS_SERVER_URL
        self.session = requests.Session()
        self.is_connected = False
        self.last_status = {}
        self.heartbeat_running = False
        self.heartbeat_thread = None

    def start_heartbeat(self):
        if self.heartbeat_running:
            return
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def stop_heartbeat(self):
        self.heartbeat_running = False

    def _heartbeat_loop(self):
        while self.heartbeat_running:
            self.send_heartbeat()
            time.sleep(config.HEARTBEAT_INTERVAL_SECONDS)

    def send_command(self, payload: dict, timeout=2.0) -> dict:
        """
        Sends a JSON command payload to the Pi command server.
        """
        try:
            resp = self.session.post(self.command_url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                self.is_connected = True
                return resp.json()
            else:
                return {"success": False, "status_code": resp.status_code, "error": resp.text}
        except Exception as e:
            self.is_connected = False
            return {"success": False, "error": str(e)}

    def send_heartbeat(self) -> bool:
        res = self.send_command({"type": "heartbeat", "command": "HEARTBEAT"}, timeout=1.5)
        self.is_connected = res.get("success", False)
        return self.is_connected

    def trigger_bird_deterrence(self, species: str, confidence: float):
        """
        Commands the Pi to trigger audio and scare motion for a verified bird.
        """
        payload = {
            "type": "bird",
            "command": "DETER_BIRD",
            "species": species,
            "confidence": round(float(confidence), 3)
        }
        return self.send_command(payload)

    def stop_bird_deterrence(self):
        """
        Commands the Pi to immediately stop deterrence audio and scare motion when bird is removed.
        """
        payload = {
            "type": "bird",
            "command": "STOP_DETERRENCE"
        }
        return self.send_command(payload)

    def send_tomato_update(self, detections: list):
        """
        Sends tomato telemetry to the Pi.
        """
        payload = {
            "type": "tomato",
            "command": "TOMATO_UPDATE",
            "detections": detections
        }
        return self.send_command(payload)

    def emergency_stop(self):
        """
        Immediately sends emergency stop to halt all hardware.
        """
        payload = {
            "type": "system",
            "command": "EMERGENCY_STOP"
        }
        return self.send_command(payload)

    def move(self, direction: str):
        payload = {
            "type": "motion",
            "command": "MOVE",
            "direction": direction
        }
        return self.send_command(payload)

    def sweep_camera(self):
        payload = {
            "type": "servo",
            "command": "ROTATE_CAMERA"
        }
        return self.send_command(payload)

    def get_status(self) -> dict:
        """
        Fetches current status of the Pi hardware node.
        """
        try:
            resp = self.session.get(self.status_url, timeout=2.0)
            if resp.status_code == 200:
                self.is_connected = True
                self.last_status = resp.json()
                return self.last_status
            else:
                self.is_connected = False
                return {"status": "error", "code": resp.status_code}
        except Exception as e:
            self.is_connected = False
            return {"status": "offline", "error": str(e)}

# Singleton client instance
pi_client = PiClient()
