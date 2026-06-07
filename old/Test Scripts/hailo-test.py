import cv2
import numpy as np
import serial
import time
import math
from hailo_platform import VDevice, HEF, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType

# ---------------- Serial Setup ---------------- #
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 115200
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
time.sleep(1)
arduino.reset_input_buffer()

# ---------------- Utility Functions ---------------- #
def angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def map_fingers_to_m6(finger_angle):
    finger_angle = max(15, min(53, finger_angle))
    norm = 1.0 - ((finger_angle - 15) / (53 - 15))
    return int(15 + norm * (73 - 15))

def calculate_angle_2d(p1, p2):
    return np.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def map_value(value, in_min, in_max, out_min=0, out_max=180):
    mapped = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return int(np.clip(mapped, out_min, out_max))

def smooth(old, new, alpha=0.20):
    return old * (1 - alpha) + new * alpha

# ---------------- NPU Setup & Main Loop ---------------- #
# Point this to the .hef file you downloaded!
HEF_PATH = "hand_landmark_lite.hef" 
hef = HEF(HEF_PATH)

cap = cv2.VideoCapture(0)
last_send_time = 0

# Initial Servo Values
sm1, sm2, sm3, sm4, sm5, sm6 = 90.0, 90.0, 90.0, 90.0, 90.0, 20.0

print("Connecting to Hailo-8 NPU...")
with VDevice() as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=target.create_interface())
    network_group = target.configure(hef, configure_params)[0]
    
    # Configure input as UINT8 (standard pixels) and output as FLOAT32 (raw math)
    input_vparams = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    output_vparams = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    with network_group.create_vstreams(input_vparams, output_vparams) as vstreams:
        input_height, input_width = hef.get_input_vstream_infos()[0].shape[1:3]
        print(f"NPU Ready. Expected Input: {input_width}x{input_height}")

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # 1. PRE-PROCESS: Resize to fit NPU
            resized = cv2.resize(frame, (input_width, input_height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb, axis=0) # Shape becomes (1, H, W, 3)

            # 2. INFERENCE: Send to Hailo-8 and instantly get result
            vstreams.input[0].send(input_data)
            raw_output = vstreams.output[0].recv() 

            # 3. POST-PROCESS: Decode the Tensor
            # Convert flat array of 63 numbers into 21 rows of (X, Y, Z)
            try:
                landmarks = raw_output[0].reshape(21, 3) 
                
                # Helper to grab coordinates and scale them to your webcam size
                def pt(i): return np.array([landmarks[i][0] * w, landmarks[i][1] * h])
                
                wrist      = pt(0)
                thumb_tip  = pt(4)
                index_mcp  = pt(5)
                index_tip  = pt(8)
                middle_mcp = pt(9)
                pinky_mcp  = pt(17)

                # --- Execute Robot Math ---
                f_angle = angle_3pts(thumb_tip, wrist, index_tip)
                m6 = map_fingers_to_m6(f_angle)
                
                m1 = map_value(wrist[0], 0, w, 180, 0) # X to Base (Inverted)
                m3 = map_value(wrist[1], 0, h, 165, 15) # Y to Elbow (Inverted)
                m5 = map_value(calculate_angle_2d(index_mcp, pinky_mcp), -90, 90, 0, 180) # Roll
                m4 = map_value(calculate_angle_2d(wrist, middle_mcp), -135, -45, 0, 180) # Pitch
                
                hand_size = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])
                m2 = map_value(hand_size, 50, 250, 45, 135) # Pseudo-Depth

                # Smooth
                sm1 = smooth(sm1, m1); sm2 = smooth(sm2, m2); sm3 = smooth(sm3, m3)
                sm4 = smooth(sm4, m4); sm5 = smooth(sm5, m5); sm6 = smooth(sm6, m6, 0.25)

                # Send to Arduino at 10Hz
                now = time.time()
                if now - last_send_time >= 0.1:
                    packet = f"CMD,{int(sm1)},{int(sm2)},{int(sm3)},{int(sm4)},{int(sm5)},{int(sm6)}\n"
                    arduino.write(packet.encode())
                    print(packet.strip())
                    last_send_time = now

                # (Optional) Draw dots on the frame to verify tracking
                for point in landmarks:
                    cv2.circle(frame, (int(point[0]*w), int(point[1]*h)), 4, (0, 255, 0), -1)

            except Exception as e:
                pass # If no hand is detected, tensor reshape might fail. Just pass to next frame.

            cv2.imshow("Hailo Braccio Teleop", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
arduino.close()
cv2.destroyAllWindows()