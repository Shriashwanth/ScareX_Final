"""
Raspberry Pi Hardware Abstraction Layer
Manages DC Motors, Camera Pan Servo, HC-SR04 Ultrasonic Distance Sensor, and LEDs.
Uses gpiozero when available, with automatic Mock hardware fallback for local development.
"""
import time
import threading
import raspberry_pi.config as config

try:
    from gpiozero import Motor, DistanceSensor, Servo, LED, Buzzer
    GPIO_AVAILABLE = True
except (ImportError, Exception) as e:
    GPIO_AVAILABLE = False

class MockBuzzer:
    def __init__(self, name="Buzzer"):
        self.name = name
    def on(self): print(f"[MOCK-HW] {self.name} -> ON (Buzzing)")
    def off(self): print(f"[MOCK-HW] {self.name} -> OFF")
    def beep(self, on_time=0.15, off_time=0.1, n=3):
        print(f"[MOCK-HW] {self.name} -> BEEP BEEP BEEP (Buzzer Sound Alert)")

class MockMotor:
    def __init__(self, name="Motor"):
        self.name = name
    def forward(self): print(f"[MOCK-HW] {self.name} -> FORWARD")
    def backward(self): print(f"[MOCK-HW] {self.name} -> BACKWARD")
    def stop(self): print(f"[MOCK-HW] {self.name} -> STOP")

class MockServo:
    def __init__(self, name="Servo"):
        self.name = name
    def min(self): print(f"[MOCK-HW] {self.name} -> MIN")
    def mid(self): print(f"[MOCK-HW] {self.name} -> MID")
    def max(self): print(f"[MOCK-HW] {self.name} -> MAX")
    def detach(self): pass

class MockDistanceSensor:
    @property
    def distance(self):
        # Returns simulated 1.25 meters
        return 1.25

class MockLED:
    def __init__(self, name="LED"):
        self.name = name
    def on(self): pass
    def off(self): pass
    def blink(self, on_time=0.5, off_time=0.5): pass

class HardwareController:
    def __init__(self):
        pins = config.GPIO_PINS
        self.is_mock = not GPIO_AVAILABLE
        self.lock = threading.RLock()
        self.is_scaring = False

        if GPIO_AVAILABLE:
            try:
                self.motor_left = Motor(
                    forward=pins["motor_left_forward"],
                    backward=pins["motor_left_backward"]
                )
                self.motor_right = Motor(
                    forward=pins["motor_right_forward"],
                    backward=pins["motor_right_backward"]
                )
                self.ultrasonic = DistanceSensor(
                    echo=pins["ultrasonic_echo"],
                    trigger=pins["ultrasonic_trigger"]
                )
                self.camera_servo = Servo(pins["servo_camera"])
                self.led_status = LED(pins["led_status"])
                self.led_scare = LED(pins["led_scare"])
                self.buzzer = Buzzer(pins.get("buzzer", 12))
                print("[Hardware] Initialized via gpiozero.")
            except Exception as e:
                print(f"[Hardware] Error initializing gpiozero ({e}). Using Mock hardware.")
                self._init_mock()
        else:
            print("[Hardware] gpiozero not available. Using Mock hardware.")
            self._init_mock()

    def _init_mock(self):
        self.is_mock = True
        self.motor_left = MockMotor("LeftMotor")
        self.motor_right = MockMotor("RightMotor")
        self.ultrasonic = MockDistanceSensor()
        self.camera_servo = MockServo("CameraServo")
        self.led_status = MockLED("StatusLED")
        self.led_scare = MockLED("ScareLED")
        self.buzzer = MockBuzzer("HardwareBuzzer")

    def move_forward(self):
        with self.lock:
            self.motor_left.forward()
            self.motor_right.forward()

    def move_backward(self):
        with self.lock:
            self.motor_left.backward()
            self.motor_right.backward()

    def turn_left(self):
        with self.lock:
            self.motor_left.backward()
            self.motor_right.forward()

    def turn_right(self):
        with self.lock:
            self.motor_left.forward()
            self.motor_right.backward()

    def stop_motors(self):
        with self.lock:
            self.motor_left.stop()
            self.motor_right.stop()

    def rotate_camera_servo(self):
        """
        Sweeps the camera pan servo to scan surroundings.
        """
        def _sweep():
            with self.lock:
                print("[Hardware] Sweeping camera servo...")
                self.camera_servo.max()
            time.sleep(0.6)
            with self.lock:
                self.camera_servo.min()
            time.sleep(0.6)
            with self.lock:
                self.camera_servo.mid()
                if hasattr(self.camera_servo, 'detach'):
                    self.camera_servo.detach()
        threading.Thread(target=_sweep, daemon=True).start()

    def execute_scare_motion(self):
        """
        Executes an assertive scare sequence (turns/flaps/LED flash) to frighten birds away.
        """
        if self.is_scaring:
            return
        self.is_scaring = True

        def _action():
            try:
                self.led_scare.on()
                if hasattr(self.buzzer, 'beep'):
                    self.buzzer.beep(on_time=0.15, off_time=0.1, n=3)
                else:
                    self.buzzer.on()
                # Rapid alternating turns simulates wing flapping / body shaking
                self.turn_left()
                time.sleep(0.4)
                self.turn_right()
                time.sleep(0.4)
                self.turn_left()
                time.sleep(0.3)
                self.stop_motors()
            finally:
                self.buzzer.off()
                self.led_scare.off()
                self.is_scaring = False

        threading.Thread(target=_action, daemon=True).start()

    def stop_scare_motion(self):
        """
        Immediately stops any active scare motion, turns off buzzer and scare LED.
        """
        with self.lock:
            self.is_scaring = False
            self.stop_motors()
            try:
                self.buzzer.off()
            except Exception:
                pass
            try:
                self.led_scare.off()
            except Exception:
                pass

    def get_distance(self) -> float:
        """
        Returns obstacle distance in meters from HC-SR04 ultrasonic sensor.
        """
        try:
            return round(float(self.ultrasonic.distance), 2)
        except Exception:
            return 1.0

    def emergency_stop(self):
        """
        Immediately halts all movement and forces actuators into safe state.
        """
        with self.lock:
            print("[Hardware] !!! EMERGENCY STOP TRIGGERED !!!")
            self.motor_left.stop()
            self.motor_right.stop()
            if hasattr(self.camera_servo, 'detach'):
                self.camera_servo.detach()
            self.buzzer.off()
            self.led_scare.off()
            self.is_scaring = False

# Global singleton
hardware = HardwareController()
