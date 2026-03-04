import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import math

# ---------------- Serial Setup ---------------- #
SERIAL_PORT = "/dev/ttyACM0"          # change if needed
BAUD_RATE   = 115200

# NOTE: If your Arduino is not plugged in yet, leave these commented out to test the camera first!
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
    finger_angle = max(15, min(53, finger_angle))
    norm = (finger_angle - 15) / (53 - 15)
    norm = 1.0 - norm
    m6 = 15 + norm * (73 - 15)
    return int(m6)

def map_x_to_m1(x, frame_width):
    angle = 180.0 - (x / frame_width) * 180.0 
    return int(np.clip(angle, 0, 180))

def map_y_to_m3(y, frame_height):
    angle = 180.0 - (y / frame_height) * 180.0
    return int(np.clip(angle, 15, 165))

def limit_M6(angle):
    return int(np.clip(angle, 15, 73))

def smooth(old, new, alpha=0.25):
    return old * (1 - alpha) + new * alpha

def calculate_angle_2d(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle_rad = math.atan2(delta_y, delta_x)
    return np.degrees(angle_rad)

def map_value(value, in_min, in_max, out_min=0, out_max=180):
    mapped = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return int(np.clip(mapped, out_min, out_max))

# ---------------- Main Loop ---------------- #
cap = cv2.VideoCapture(0)
last_send_time = 0

# Starting positions
smooth_m1 = 90.0 
smooth_m2 = 90.0  
smooth_m3 = 90.0 
smooth_m4 = 90.0  
smooth_m5 = 90.0  
smooth_m6 = 20.0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark
        
        def pt(i): return (lm[i].x * w, lm[i].y * h)

        wrist      = pt(0)
        thumb_tip  = pt(4)
        index_tip  = pt(8)
        
        index_mcp  = pt(5)
        middle_mcp = pt(9)
        pinky_mcp  = pt(17)

        # 1. Gripper
        finger_angle = angle_3pts(thumb_tip, wrist, index_tip)
        mapped_m6 = map_fingers_to_m6(finger_angle)

        # 2. Base and Elbow
        mapped_m1 = map_x_to_m1(wrist[0], w)
        mapped_m3 = map_y_to_m3(wrist[1], h)

        # 3. Wrist Roll and Pitch
        raw_roll_angle = calculate_angle_2d(index_mcp, pinky_mcp)
        mapped_m5 = map_value(raw_roll_angle, -90, 90, 0, 180)

        raw_pitch_angle = calculate_angle_2d(wrist, middle_mcp)
        mapped_m4 = map_value(raw_pitch_angle, -135, -45, 0, 180)

        # 4. Shoulder (Pseudo-Depth)
        hand_size_pixels = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])
        mapped_m2 = map_value(hand_size_pixels, 50, 250, 45, 135)

        # 5. Smoothing
        smooth_m1 = smooth(smooth_m1, mapped_m1, alpha=0.20)
        smooth_m2 = smooth(smooth_m2, mapped_m2, alpha=0.20)
        smooth_m3 = smooth(smooth_m3, mapped_m3, alpha=0.20)
        smooth_m4 = smooth(smooth_m4, mapped_m4, alpha=0.20)
        smooth_m5 = smooth(smooth_m5, mapped_m5, alpha=0.20)
        smooth_m6 = smooth(smooth_m6, mapped_m6, alpha=0.25)

        servo_m1 = int(smooth_m1)
        servo_m2 = int(smooth_m2)
        servo_m3 = int(smooth_m3)
        servo_m4 = int(smooth_m4)
        servo_m5 = int(smooth_m5)
        servo_m6 = limit_M6(smooth_m6)

        # 6. Serial Comm
        now = time.time()
        if now - last_send_time >= 0.1:
            packet = f"CMD,{servo_m1},{servo_m2},{servo_m3},{servo_m4},{servo_m5},{servo_m6}\n"
            arduino.write(packet.encode())
            print(f"M1:{servo_m1} | M2:{servo_m2} | M3:{servo_m3} | M4:{servo_m4} | M5:{servo_m5} | M6:{servo_m6}")
            last_send_time = now

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Braccio Control", frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
arduino.close() 
cv2.destroyAllWindows()