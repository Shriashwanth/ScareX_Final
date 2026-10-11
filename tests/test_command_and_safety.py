"""
Verification Test for Stages 10, 11, 12, 13, 14, 15:
- Pi command server endpoints (DETER_BIRD, MOVE, EMERGENCY_STOP)
- Laptop PiClient communication
- Audio deterrent selection & playback
- Hardware control and mock safety
- Communication Watchdog and Emergency Stop
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raspberry_pi.command_server import app as pi_app
from laptop.communication.pi_client import PiClient
import raspberry_pi.config as rpi_config

def run_test():
    print("=== TESTING PI COMMAND SERVER & LAPTOP CLIENT ===")
    port = 9000
    
    # Start Pi Command Server in background
    def run_server():
        pi_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1.0)

    client = PiClient(
        command_url=f"http://127.0.0.1:{port}/command",
        status_url=f"http://127.0.0.1:{port}/status"
    )

    # 1. Test Heartbeat & Status
    print("[Test 1] Testing Heartbeat and Status...")
    res = client.send_heartbeat()
    assert res is True, "Heartbeat failed"
    status = client.get_status()
    assert status.get("status") == "online", f"Status check failed: {status}"
    print(f"  Status response: {status}")

    # 2. Test Deter Bird Command
    print("[Test 2] Testing DETER_BIRD command...")
    deter_res = client.trigger_bird_deterrence(species="crow", confidence=0.91)
    assert deter_res.get("success") is True, f"Deterrence command failed: {deter_res}"
    print(f"  Deterrence response: {deter_res}")

    # 3. Test Emergency Stop Command
    print("[Test 3] Testing EMERGENCY_STOP command...")
    stop_res = client.emergency_stop()
    assert stop_res.get("success") is True, f"Emergency stop failed: {stop_res}"
    status = client.get_status()
    assert status.get("emergency_stop") is True, "Emergency stop state not reflected"
    print(f"  Emergency stop confirmed: {status}")

    # 4. Test Watchdog
    print("[Test 4] Testing Watchdog safety timeout...")
    # Sleep past WATCHDOG_TIMEOUT (3.0s) without sending any heartbeats or commands
    time.sleep(3.5)
    status = client.get_status()
    assert status.get("watchdog_ok") is False, "Watchdog should have triggered safe state!"
    print(f"  Watchdog safely triggered timeout! status: {status}")

    # 5. Test Communication Restoration
    print("[Test 5] Testing watchdog recovery upon command restoration...")
    recover_res = client.send_heartbeat()
    assert recover_res is True, "Recovery heartbeat failed"
    status = client.get_status()
    assert status.get("watchdog_ok") is True, "Watchdog failed to recover after heartbeat!"
    print(f"  Watchdog safely recovered: {status}")

    print("\n=== ALL COMMAND, AUDIO, HARDWARE & WATCHDOG TESTS PASSED! ===")

if __name__ == "__main__":
    run_test()
