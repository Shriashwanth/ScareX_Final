"""
Raspberry Pi Command Server with Communication Watchdog
Receives control commands from the laptop decision engine on POST http://0.0.0.0:9000/command.
Hosts hardware and audio controllers.
Includes an active watchdog that forces a hardware safe-state if communication drops.
"""
import time
import threading
from flask import Flask, request, jsonify
import raspberry_pi.config as config
from raspberry_pi.hardware import hardware
from raspberry_pi.audio import audio_player

app = Flask(__name__)

# State & Watchdog tracking
last_contact_time = time.time()
emergency_stopped = False
state_lock = threading.Lock()
tomato_telemetry = []

def watchdog_loop():
    """
    Monitors time elapsed since last message/heartbeat from laptop.
    Forces emergency stop if connection is lost.
    """
    global emergency_stopped
    print(f"[Watchdog] Active. Timeout set to {config.WATCHDOG_TIMEOUT}s.")
    while True:
        time.sleep(0.5)
        now = time.time()
        with state_lock:
            elapsed = now - last_contact_time
            if elapsed > config.WATCHDOG_TIMEOUT and not emergency_stopped:
                print(f"[Watchdog] ALERT: No contact from laptop for {elapsed:.1f}s! Entering SAFE STATE.")
                hardware.emergency_stop()
                audio_player.stop()
                emergency_stopped = True

# Start watchdog in background
watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
watchdog_thread.start()

@app.route('/command', methods=['POST'])
def handle_command():
    global last_contact_time, emergency_stopped, tomato_telemetry
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid or missing JSON payload"}), 400

    now = time.time()
    with state_lock:
        last_contact_time = now
        # Reset watchdog trigger on new valid message unless explicit EMERGENCY_STOP requested
        if emergency_stopped and data.get("command") != "EMERGENCY_STOP":
            print("[CommandServer] Laptop communication restored. Exiting watchdog safe state.")
            emergency_stopped = False

    cmd_type = data.get("type", "").lower()
    cmd = data.get("command", "").upper()

    # 1. EMERGENCY STOP
    if cmd == "EMERGENCY_STOP":
        with state_lock:
            emergency_stopped = True
        hardware.emergency_stop()
        audio_player.stop()
        print("[CommandServer] User requested EMERGENCY STOP.")
        return jsonify({"success": True, "message": "Emergency stop executed."})

    # 2. HEARTBEAT
    if cmd_type == "heartbeat" or cmd == "HEARTBEAT":
        return jsonify({
            "success": True,
            "status": "alive",
            "distance": hardware.get_distance()
        })

    # 3. BIRD DETERRENCE COMMANDS
    if cmd in ["DETER_BIRD", "START_DETERRENCE"]:
        species = data.get("species", "default")
        conf = data.get("confidence", 0.0)
        print(f"[CommandServer] Deterrence triggered for {species} ({conf*100:.1f}%)")
        
        # Audio deterrence (looping while bird is present)
        audio_res = audio_player.play_deterrent(species, loop=True)
        
        # Physical hardware scare motion (wings/movement)
        hardware.execute_scare_motion()
        
        return jsonify({
            "success": True,
            "action": "DETERRENCE_ACTIVE",
            "audio": audio_res
        })

    if cmd in ["STOP_DETERRENCE", "STOP_BIRD"]:
        print("[CommandServer] Bird removed -> Stopping deterrence audio and hardware motion.")
        audio_player.stop()
        hardware.stop_scare_motion()
        return jsonify({
            "success": True,
            "action": "DETERRENCE_STOPPED"
        })

    # 4. MOTION COMMANDS
    if cmd == "MOVE":
        direction = data.get("direction", "stop").lower()
        if direction == "forward":
            hardware.move_forward()
        elif direction == "backward":
            hardware.move_backward()
        elif direction == "turn_left":
            hardware.turn_left()
        elif direction == "turn_right":
            hardware.turn_right()
        else:
            hardware.stop_motors()
        return jsonify({"success": True, "action": f"MOVE_{direction.upper()}"})

    # 5. SERVO COMMANDS
    if cmd in ["ROTATE_CAMERA", "SWEEP_SERVO"]:
        hardware.rotate_camera_servo()
        return jsonify({"success": True, "action": "SERVO_SWEEP"})

    # 6. TOMATO MONITORING DATA UPDATE
    if cmd == "TOMATO_UPDATE":
        detections = data.get("detections", [])
        with state_lock:
            tomato_telemetry = detections
        return jsonify({"success": True, "received_detections": len(detections)})

    return jsonify({"success": False, "error": f"Unknown command: {cmd}"}), 400

@app.route('/status', methods=['GET'])
def get_status():
    with state_lock:
        now = time.time()
        elapsed = now - last_contact_time
        watchdog_ok = elapsed <= config.WATCHDOG_TIMEOUT
        
    return jsonify({
        "status": "online",
        "emergency_stop": emergency_stopped,
        "watchdog_ok": watchdog_ok,
        "seconds_since_contact": round(elapsed, 2),
        "distance_meters": hardware.get_distance(),
        "audio": audio_player.get_status(),
        "is_mock_hardware": hardware.is_mock
    })

def start_command_server(port=None):
    server_port = port or config.COMMAND_PORT
    print(f"[CommandServer] Listening for laptop commands at http://0.0.0.0:{server_port}/command")
    app.run(host='0.0.0.0', port=server_port, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    start_command_server()
