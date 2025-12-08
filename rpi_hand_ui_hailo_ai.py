import cv2
import numpy as np
import time
import socket
import mediapipe as mp

from logger import DataLogger

from hailo_platform import (
    HEF, VDevice,
    HailoStreamInterface, ConfigureParams,
    InputVStreams, OutputVStreams,
    InputVStreamParams, OutputVStreamParams,
    FormatType
)

####################################################
# CONFIG
####################################################

hailo_enabled = True                 # folosim Hailo dacă init reușește
MAX_COMMANDS = 100
CSV_NAME = "session_ai.csv"

TCP_IP = "192.168.88.230"
TCP_PORT = 5005

ROI_X1, ROI_Y1 = 100, 100
ROI_X2, ROI_Y2 = 400, 400

HAILO_SCORE_THRESHOLD = 0.3          # prag după normalizare [0..1]

####################################################
# TCP CLIENT
####################################################

class CommandSender:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connect()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1)
            self.sock.connect((self.ip, self.port))
            print("[TCP] Connected.")
        except:
            print("[TCP] Failed to connect.")
            self.sock = None

    def send(self, command: str) -> bool:
        if self.sock is None:
            self.connect()
            return False

        try:
            self.sock.sendall((command + "\n").encode("utf-8"))
            return True
        except:
            print("[TCP] Lost connection — reconnecting.")
            self.sock = None
            return False

####################################################
# HAILO INIT – EXACT CA PROBE, DAR ÎNTR-O FUNCȚIE
####################################################

def init_hailo_full():
    print("[Hailo] Initializing (probe-style)...")

    hef = HEF("hand_landmark_lite.hef")
    print("[Hailo] HEF loaded.")

    device = VDevice()
    print("[Hailo] VDevice created.")

    cfg = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe
    )
    network_groups = device.configure(hef, cfg)
    network_group = network_groups[0]
    print("[Hailo] Network configured.")

    net_params = network_group.create_params()

    in_params = InputVStreamParams.make(
        network_group, format_type=FormatType.UINT8
    )
    out_params = OutputVStreamParams.make(
        network_group, format_type=FormatType.UINT8
    )

    print("[Hailo] VStream params created.")

    return device, hef, network_group, net_params, in_params, out_params

####################################################
# HAILO RUN – folosește un singur output vstream
####################################################

def run_hailo_single(input_stream, output_stream, roi_bgr):
    # ROI -> 224x224 RGB uint8 cu batch dimension (1,224,224,3)
    img = cv2.resize(roi_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.uint8)

    t0 = time.time()
    input_stream.send(img)
    raw = output_stream.recv()
    dt = (time.time() - t0) * 1000.0

    arr = raw.astype(np.float32).flatten()
    # normalizăm scorul în [0..1]
    score = float(np.mean(arr) / 255.0)

    # debug opțional:
    # print(f"[Hailo] raw_mean={np.mean(arr):.2f}, score={score:.4f}, infer_time={dt:.2f} ms")

    return score, dt

####################################################
# MAIN LOOP
####################################################

def main():
    global hailo_enabled
    print("===== STARTING SYSTEM (AI) =====")

    sender = CommandSender(TCP_IP, TCP_PORT)
    logger = DataLogger(CSV_NAME, pretty_timestamp=True)

    mp_hands = None

    device = None
    hef = None
    network_group = None
    net_params = None
    in_params = None
    out_params = None

    if hailo_enabled:
        try:
            device, hef, network_group, net_params, in_params, out_params = init_hailo_full()
        except Exception as e:
            print("[Hailo] ERROR at init:", e)
            hailo_enabled = False

    cap = cv2.VideoCapture(0)
    last_cmd = ""
    sent_count = 0

    print("[SYSTEM] Running (AI ON if available). Press 'q' to quit.")

    # Dacă nu avem Hailo, rulăm doar logică clasică + MediaPipe
    if not hailo_enabled or network_group is None:
        print("[Hailo] DISABLED – running without AI filter.")
        hailo_available = False
    else:
        hailo_available = True

    if hailo_available:
        # Activăm rețeaua + creăm streamurile EXACT ca în probe
        with network_group.activate(net_params), \
             InputVStreams(network_group, in_params) as in_streams, \
             OutputVStreams(network_group, out_params) as out_streams:

            input_stream = list(in_streams)[0]
            # luăm primul stream de output (la probe avea shape [1])
            output_stream = list(out_streams)[0]
            print(f"[Hailo] Using output stream with shape: {output_stream.shape}")

            # ---- MAIN LOOP CU HAILO ----
            while True:
                start_t = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("[Camera] Frame error.")
                    break

                roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
                roi_h, roi_w, _ = roi.shape

                # Lazy init MediaPipe
                if mp_hands is None:
                    print("[MediaPipe] Initializing...")
                    mp_hands = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    print("[MediaPipe] READY.")

                # SEGMENTARE PIELĂ + CENTROID
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lower = np.array([0, 20, 70])
                upper = np.array([20, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.dilate(mask, None, iterations=2)
                mask = cv2.GaussianBlur(mask, (7, 7), 0)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cx, cy = -1, -1
                command = "No hand"

                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 3000:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            if cx < roi_w // 3:
                                command = "Move Left"
                            elif cx > 2 * (roi_w // 3):
                                command = "Move Right"
                            else:
                                command = "Centered"

                        cv2.circle(roi, (cx, cy), 7, (0, 255, 255), -1)

                # HAILO – rulează PE FIECARE CADRU
                hailo_score = 0.0
                hailo_valid = False
                hailo_dt = 0.0

                try:
                    hailo_score, hailo_dt = run_hailo_single(input_stream, output_stream, roi)
                    hailo_valid = hailo_score > HAILO_SCORE_THRESHOLD

                    if not hailo_valid:
                        command = "No hand"

                    print(f"[Hailo] score={hailo_score:.4f}, valid={hailo_valid}, cmd={command}")

                except Exception as e:
                    print("[Hailo] ERROR at inference:", e)
                    hailo_valid = False

                # MEDIAPIPE – landmark-uri (puncte verzi în ROI)
                mediapipe_points = 0
                mp_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result = mp_hands.process(mp_rgb)

                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0]
                    mediapipe_points = len(lm.landmark)

                    for p in lm.landmark:
                        lx = int(p.x * roi_w)
                        ly = int(p.y * roi_h)
                        cv2.circle(roi, (lx, ly), 3, (0, 255, 0), -1)

                # TRIMITERE COMANDĂ TCP
                tcp_sent = 0
                tcp_reconn = 0

                if command != last_cmd:
                    ok = sender.send(command)
                    tcp_sent = 1
                    if not ok:
                        tcp_reconn = 1

                    last_cmd = command
                    sent_count += 1
                    print(f"[CMD] {command} ({sent_count}/{MAX_COMMANDS})")

                if sent_count >= MAX_COMMANDS:
                    print("[SYSTEM] Max commands reached (AI).")
                    logger.close()
                    break

                # LOGGING
                dt = (time.time() - start_t) * 1000
                fps = 1000 / max(dt, 0.0001)

                logger.log(
                    fps=fps,
                    frame_time_ms=dt,
                    hailo_score=hailo_score,
                    hailo_valid=1 if hailo_valid else 0,
                    mediapipe_landmarks=mediapipe_points,
                    command=command,
                    tcp_sent=tcp_sent,
                    tcp_reconnected=tcp_reconn,
                    cx=cx,
                    cy=cy
                )

                # DISPLAY
                cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 255, 0), 2)
                cv2.putText(frame, command, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if hailo_valid else (0, 0, 255), 2)

                cv2.imshow("UI Final - AI", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.close()
                    break

    else:
        # ---- FALLBACK: fără Hailo, doar logică clasică + MediaPipe ----
        print("[Hailo] Not available, running fallback (NO AI in this script).")
        while True:
            start_t = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Frame error.")
                break

            roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
            roi_h, roi_w, _ = roi.shape

            if mp_hands is None:
                print("[MediaPipe] Initializing...")
                mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[MediaPipe] READY.")

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 20, 70])
            upper = np.array([20, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.dilate(mask, None, iterations=2)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cx, cy = -1, -1
            command = "No hand"

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 3000:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        if cx < roi_w // 3:
                            command = "Move Left"
                        elif cx > 2 * (roi_w // 3):
                            command = "Move Right"
                        else:
                            command = "Centered"

                    cv2.circle(roi, (cx, cy), 7, (0, 255, 255), -1)

            hailo_score = 0.0
            hailo_valid = False

            mediapipe_points = 0
            mp_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            result = mp_hands.process(mp_rgb)
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                mediapipe_points = len(lm.landmark)
                for p in lm.landmark:
                    lx = int(p.x * roi_w)
                    ly = int(p.y * roi_h)
                    cv2.circle(roi, (lx, ly), 3, (0, 255, 0), -1)

            tcp_sent = 0
            tcp_reconn = 0
            if command != last_cmd:
                ok = sender.send(command)
                tcp_sent = 1
                if not ok:
                    tcp_reconn = 1
                last_cmd = command
                sent_count += 1
                print(f"[CMD] {command} ({sent_count}/{MAX_COMMANDS})")

            if sent_count >= MAX_COMMANDS:
                print("[SYSTEM] Max commands reached (FALLBACK).")
                logger.close()
                break

            dt = (time.time() - start_t) * 1000
            fps = 1000 / max(dt, 0.0001)

            logger.log(
                fps=fps,
                frame_time_ms=dt,
                hailo_score=hailo_score,
                hailo_valid=1 if hailo_valid else 0,
                mediapipe_landmarks=mediapipe_points,
                command=command,
                tcp_sent=tcp_sent,
                tcp_reconnected=tcp_reconn,
                cx=cx,
                cy=cy
            )

            cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 255, 0), 2)
            cv2.putText(frame, command, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            cv2.imshow("UI Final - AI FALLBACK", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.close()
                break

    cap.release()
    cv2.destroyAllWindows()

####################################################
# RUN
####################################################

if __name__ == "__main__":
    main()
