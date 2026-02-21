import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import math

# ---------------- Serial Setup ---------------- #
SERIAL_PORT = "/dev/ttyACM0"          # change if needed
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

def map_x_to_m1(x, frame_width):
    """ Maps the X coordinate (left/right) to the Base Servo (M1) """
    # Map X (0 to width) to Servo Angle (0 to 180)
    # Note: 180.0 - ... inverts the axis so moving right on camera moves arm right.
    angle = 180.0 - (x / frame_width) * 180.0 
    return int(np.clip(angle, 0, 180))

def map_y_to_m3(y, frame_height):
    """ Maps the Y coordinate (up/down) to the Elbow Servo (M3) """
    # Map Y (0 to height) to Servo Angle (15 to 165 to prevent hardware collision)
    # Camera Y is 0 at the top, so we invert it for intuitive up/down arm movement.
    angle = 180.0 - (y / frame_height) * 180.0
    return int(np.clip(angle, 15, 165))

def limit_M6(angle):
    return int(np.clip(angle, 15, 73))

def smooth(old, new, alpha=0.25):
    return old * (1 - alpha) + new * alpha

# ---------------- Main Loop ---------------- #
cap = cv2.VideoCapture(0)
last_send_time = 0

# Initial smoothed values (90 is usually center for Braccio)
smooth_m1 = 90.0 
smooth_m3 = 90.0 
smooth_m6 = 20.0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: Flip the frame horizontally for a "mirror" effect 
    # which usually feels more natural for controlling a robot.
    frame = cv2.flip(frame, 1)

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

        # --- 1. Calculate Gripper (M6) ---
        finger_angle = angle_3pts(thumb_tip, wrist, index_tip)
        mapped_m6 = map_fingers_to_m6(finger_angle)

        # --- 2. Calculate Base (M1) and Elbow (M3) ---
        mapped_m1 = map_x_to_m1(wrist[0], w)
        mapped_m3 = map_y_to_m3(wrist[1], h)

        # --- 3. Smooth & Limit All Values ---
        smooth_m1 = smooth(smooth_m1, mapped_m1, alpha=0.20)
        smooth_m3 = smooth(smooth_m3, mapped_m3, alpha=0.20)
        smooth_m6 = smooth(smooth_m6, mapped_m6, alpha=0.25)

        servo_m1 = int(smooth_m1)
        servo_m3 = int(smooth_m3)
        servo_m6 = limit_M6(smooth_m6)

        # --- 4. Send to Arduino ---
        now = time.time()
        if now - last_send_time >= 0.1:
            # Replaced the first '90' with M1, and third '90' with M3
            packet = f"CMD,{servo_m1},90,{servo_m3},90,90,{servo_m6}\n"
            arduino.write(packet.encode())
            print(f"M1(Base):{servo_m1}° | M3(Elbow):{servo_m3}° | M6(Grip):{servo_m6}°")
            last_send_time = now

        # Draw hand landmarks
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Braccio Control", frame)
    if cv2.waitKey(1) & 0xFF == 27: # Press ESC to quit
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()