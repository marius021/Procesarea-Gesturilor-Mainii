import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import math

# ---------------- Serial Setup ---------------- #
SERIAL_PORT = "/dev/ttyACM0"     # change if needed
BAUD_RATE   = 115200
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)

time.sleep(1)                 # allow Arduino to reboot
arduino.reset_input_buffer()  # clear garbage


start = time.time()

# ---------------- Mediapipe Setup ---------------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- Utility Functions ---------------- #
def angle_3pts(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def map_fingers_to_m6(finger_angle):
    # clamp input angle
    finger_angle = max(15, min(53, finger_angle))

    # normalize: 15° → 0.0, 53° → 1.0
    norm = (finger_angle - 15) / (53 - 15)

    # inverse normalization (so 53° gives small, 15° gives large)
    norm = 1.0 - norm

    # scale to servo range 15..73
    m6 = 15 + norm * (73 - 15)

    return int(m6)

def limit_M6(angle):
    return int(np.clip(angle, 15, 73))

def smooth(old, new, alpha=0.25):
    return old * (1 - alpha) + new * alpha

cap = cv2.VideoCapture(0)
last_send_time = 0
smooth_m6 = 20  # initial value

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        lm = hand.landmark
        def pt(i): return (lm[i].x * w, lm[i].y * h)

        wrist     = pt(0)
        thumb_tip = pt(4)
        index_tip = pt(8)

        # Compute angle at wrist between thumb and index
        finger_angle = angle_3pts(thumb_tip, wrist, index_tip)

        # Apply inverse mapping you asked for
        mapped_m6 = map_fingers_to_m6(finger_angle)

        # Smooth & limit
        smooth_m6 = smooth(smooth_m6, mapped_m6)
        servo_m6 = limit_M6(smooth_m6)

        # Send every 0.1 seconds
        now = time.time()
        if now - last_send_time >= 0.1:
            packet = f"CMD,90,90,90,90,90,{servo_m6}\n"
            arduino.write(packet.encode())
            print(f"Finger angle={finger_angle:.1f}° -> M6={servo_m6}")
            last_send_time = now

        # Draw hand
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gripper Control (M6)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
