import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# ====================== CONFIG ====================== #
SERIAL_PORT = "/dev/ttyACM0"   # Change if needed
BAUD_RATE = 115200

CAM_INDEX = 0
SEND_INTERVAL = 0.05           # seconds (20 Hz)
HAND_LOST_TIMEOUT = 0.5        # seconds before failsafe triggers

# If MediaPipe handedness feels reversed after mirror flip, change to "Left"
TARGET_HAND_LABEL = "Right"

# Braccio servo limits (adjust carefully after testing)
M1_MIN, M1_MAX = 0, 180       # Base
M2_MIN, M2_MAX = 15, 165      # Shoulder
M3_MIN, M3_MAX = 15, 165      # Elbow
M4_MIN, M4_MAX = 0, 180       # Wrist vertical
M5_MIN, M5_MAX = 0, 180       # Wrist rotation
M6_MIN, M6_MAX = 15, 73       # Gripper (your current safe range)

# Fixed joints for now (you can later make these dynamic)
M2_FIXED = 90
M4_FIXED = 90
M5_FIXED = 90

# Smoothing (0..1): lower = smoother but slower
ALPHA_M1 = 0.18
ALPHA_M3 = 0.18
ALPHA_M6 = 0.22

# Max step per SEND_INTERVAL (deg). Prevents sudden jumps.
MAX_STEP_M1 = 4
MAX_STEP_M3 = 4
MAX_STEP_M6 = 3

# Failsafe behavior: "HOLD" (freeze current pose) or "NEUTRAL"
FAILSAFE_MODE = "HOLD"

# Neutral pose if FAILSAFE_MODE == "NEUTRAL"
NEUTRAL_M1 = 90
NEUTRAL_M2 = 90
NEUTRAL_M3 = 90
NEUTRAL_M4 = 90
NEUTRAL_M5 = 90
NEUTRAL_M6 = 25

# Optional serial ACK reading (only if Arduino sends "OK")
READ_ACK = False

# ==================================================== #


# ---------------- Utility Functions ---------------- #
def clamp(v, lo, hi):
    return int(np.clip(v, lo, hi))

def smooth(old, new, alpha):
    return old * (1.0 - alpha) + new * alpha

def rate_limit(current, target, max_step):
    """Limit change per update to avoid jerky motion."""
    if target > current + max_step:
        return current + max_step
    if target < current - max_step:
        return current - max_step
    return target

def map_x_to_m1(x, frame_width):
    # mirror-friendly mapping (frame is flipped)
    angle = 180.0 - (x / frame_width) * 180.0
    return clamp(angle, M1_MIN, M1_MAX)

def map_y_to_m3(y, frame_height):
    # camera y=0 top -> invert so moving hand up feels like "arm up"
    angle = 180.0 - (y / frame_height) * 180.0
    return clamp(angle, M3_MIN, M3_MAX)

def map_pinch_to_m6(thumb_tip, index_tip, wrist, middle_mcp):
    """
    Use thumb-index distance normalized by hand size.
    More robust and intuitive than using finger angle at the wrist.
    """
    pinch_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
    hand_ref = np.linalg.norm(np.array(wrist) - np.array(middle_mcp)) + 1e-6

    norm = pinch_dist / hand_ref  # normalized pinch distance

    # Calibrate these after testing:
    # norm ~0.25 => fingers close together (pinch)
    # norm ~0.95 => fingers far apart (open)
    PINCH_MIN = 0.25
    PINCH_MAX = 0.95

    norm = (norm - PINCH_MIN) / (PINCH_MAX - PINCH_MIN)
    norm = float(np.clip(norm, 0.0, 1.0))

    # Decide direction:
    # If pinch close should CLOSE gripper, usually map smaller distance -> "closed" angle.
    # Depending on your gripper mechanics, you may need to invert this line.
    # Current mapping: open fingers -> larger M6 value
    m6 = M6_MIN + norm * (M6_MAX - M6_MIN)

    return clamp(m6, M6_MIN, M6_MAX)

def build_packet(m1, m2, m3, m4, m5, m6):
    return f"CMD,{m1},{m2},{m3},{m4},{m5},{m6}\n"

def send_pose(ser, m1, m2, m3, m4, m5, m6):
    packet = build_packet(m1, m2, m3, m4, m5, m6)
    ser.write(packet.encode())
    if READ_ACK:
        try:
            ack = ser.readline().decode(errors="ignore").strip()
            if ack:
                print("ACK:", ack)
        except Exception:
            pass


# ---------------- Serial Setup ---------------- #
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    time.sleep(1.5)  # allow Arduino reset
    arduino.reset_input_buffer()
    print(f"[INFO] Serial connected on {SERIAL_PORT} @ {BAUD_RATE}")
except Exception as e:
    raise RuntimeError(f"Could not open serial port {SERIAL_PORT}: {e}")


# ---------------- Mediapipe Setup ---------------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils


# ---------------- Camera Setup ---------------- #
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")


# ---------------- State Variables ---------------- #
last_send_time = 0.0
last_hand_seen_time = 0.0
failsafe_active = False

# Smoothed values (floats)
smooth_m1 = 90.0
smooth_m3 = 90.0
smooth_m6 = 25.0

# Last sent values (ints) for rate limiting
servo_m1 = 90
servo_m2 = M2_FIXED
servo_m3 = 90
servo_m4 = M4_FIXED
servo_m5 = M5_FIXED
servo_m6 = 25

print("[INFO] Press ESC to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Camera frame read failed.")
            break

        # Mirror view for more natural teleop feel
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        selected_hand = None
        selected_lm = None
        detected_label = None

        # ---- Select only target hand (Right by default) ---- #
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label  # "Left" or "Right"
                if label == TARGET_HAND_LABEL:
                    selected_hand = hand_lm
                    selected_lm = hand_lm.landmark
                    detected_label = label
                    break

            # If only one hand is seen and label mismatch happens due to camera/mirror settings,
            # you can temporarily allow fallback:
            if selected_hand is None and len(results.multi_hand_landmarks) == 1:
                # Comment this block out if you want strict right-hand-only behavior
                selected_hand = results.multi_hand_landmarks[0]
                selected_lm = selected_hand.landmark
                detected_label = results.multi_handedness[0].classification[0].label

        hand_detected = selected_hand is not None

        if hand_detected:
            last_hand_seen_time = time.time()
            failsafe_active = False

            def pt(i):
                return (selected_lm[i].x * w, selected_lm[i].y * h)

            wrist = pt(0)
            thumb_tip = pt(4)
            index_tip = pt(8)
            middle_mcp = pt(9)

            # 1) Map hand -> target servos
            mapped_m1 = map_x_to_m1(wrist[0], w)
            mapped_m3 = map_y_to_m3(wrist[1], h)
            mapped_m6 = map_pinch_to_m6(thumb_tip, index_tip, wrist, middle_mcp)

            # 2) Smooth targets (float)
            smooth_m1 = smooth(smooth_m1, mapped_m1, ALPHA_M1)
            smooth_m3 = smooth(smooth_m3, mapped_m3, ALPHA_M3)
            smooth_m6 = smooth(smooth_m6, mapped_m6, ALPHA_M6)

            # 3) Rate-limit before sending (int)
            target_m1 = clamp(round(smooth_m1), M1_MIN, M1_MAX)
            target_m3 = clamp(round(smooth_m3), M3_MIN, M3_MAX)
            target_m6 = clamp(round(smooth_m6), M6_MIN, M6_MAX)

            servo_m1 = clamp(rate_limit(servo_m1, target_m1, MAX_STEP_M1), M1_MIN, M1_MAX)
            servo_m3 = clamp(rate_limit(servo_m3, target_m3, MAX_STEP_M3), M3_MIN, M3_MAX)
            servo_m6 = clamp(rate_limit(servo_m6, target_m6, MAX_STEP_M6), M6_MIN, M6_MAX)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, selected_hand, mp_hands.HAND_CONNECTIONS)

            # Debug info
            cv2.putText(frame, f"Hand: {detected_label}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"M1:{servo_m1} M3:{servo_m3} M6:{servo_m6}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        else:
            # No hand detected -> check failsafe timeout
            elapsed = time.time() - last_hand_seen_time
            cv2.putText(frame, "No hand detected", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if elapsed > HAND_LOST_TIMEOUT:
                failsafe_active = True
                cv2.putText(frame, "FAILSAFE ACTIVE", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if FAILSAFE_MODE == "NEUTRAL":
                    # Smoothly move toward neutral (still rate-limited)
                    servo_m1 = clamp(rate_limit(servo_m1, NEUTRAL_M1, MAX_STEP_M1), M1_MIN, M1_MAX)
                    servo_m2 = NEUTRAL_M2
                    servo_m3 = clamp(rate_limit(servo_m3, NEUTRAL_M3, MAX_STEP_M3), M3_MIN, M3_MAX)
                    servo_m4 = NEUTRAL_M4
                    servo_m5 = NEUTRAL_M5
                    servo_m6 = clamp(rate_limit(servo_m6, NEUTRAL_M6, MAX_STEP_M6), M6_MIN, M6_MAX)
                else:
                    # HOLD mode = keep last pose
                    servo_m2 = M2_FIXED
                    servo_m4 = M4_FIXED
                    servo_m5 = M5_FIXED

        # Ensure fixed joints are always set (unless NEUTRAL mode overwrote them)
        if not (failsafe_active and FAILSAFE_MODE == "NEUTRAL"):
            servo_m2 = M2_FIXED
            servo_m4 = M4_FIXED
            servo_m5 = M5_FIXED

        # ---- Send serial at fixed rate ---- #
        now = time.time()
        if now - last_send_time >= SEND_INTERVAL:
            try:
                send_pose(
                    arduino,
                    servo_m1, servo_m2, servo_m3, servo_m4, servo_m5, servo_m6
                )
                print(f"M1:{servo_m1} | M2:{servo_m2} | M3:{servo_m3} | "
                      f"M4:{servo_m4} | M5:{servo_m5} | M6:{servo_m6} "
                      f"{'[FAILSAFE]' if failsafe_active else ''}")
            except Exception as e:
                print("[SERIAL ERROR]", e)

            last_send_time = now

        cv2.imshow("Braccio Control v2", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

finally:
    cap.release()
    try:
        arduino.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("[INFO] Clean exit.")