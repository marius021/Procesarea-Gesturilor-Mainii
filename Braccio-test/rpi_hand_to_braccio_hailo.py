#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
import serial
import mediapipe as mp

# ============================================================
# CONFIG (smooth mode)
# ============================================================

SERIAL_PORT = "/dev/ttyACM0"#schimbă dacă e /dev/ttyUSB0
BAUD_RATE   = 115200
TIMEOUT     = 1

# Mai rar = mai stabil pentru servo-uri (Braccio nu are nevoie de 20Hz)
UPDATE_DT   = 0.10              # 10 Hz (foarte fluid + fără înecare serial)

# ROI (zona în care cauți mâna) - ajustează după camera ta
ROI_X1, ROI_Y1 = 100, 100
ROI_X2, ROI_Y2 = 540, 420

# Poziție neutră sigură
NEUTRAL = dict(
    base=90, shoulder=90, elbow=90,
    wrist_vert=90, wrist_rot=90,
    gripper=30
)

# Dacă mâna dispare: revine încet la neutral (secunde)
RETURN_TO_NEUTRAL_SECONDS = 1.0

# ------------------------------------------------------------
# “FLUID” CONTROL TUNING
# ------------------------------------------------------------

# Deadband: dacă schimbarea e mică, NU trimitem comandă (reduce spam, reduce jitter)
DEADBAND = {
    "base": 2,
    "shoulder": 2,
    "elbow": 2,
    "wrist_vert": 2,
    "wrist_rot": 3,
    "gripper": 2
}

# Slew-rate: limită de grad / update (face mișcarea “buttery smooth”)
MAX_STEP = {
    "base": 2,         # mai mic = mai fluid
    "shoulder": 2,
    "elbow": 2,
    "wrist_vert": 2,
    "wrist_rot": 3,
    "gripper": 2
}

# EMA smoothing (alpha mai mic = mai fluid, mai “soft”)
EMA_ALPHA = {
    "base": 0.18,
    "shoulder": 0.18,
    "elbow": 0.18,
    "wrist_vert": 0.18,
    "gripper": 0.22
}

# ------------------------------------------------------------
# HAILO (opțional)
# ------------------------------------------------------------
HAILO_ENABLED = True
HAILO_SCORE_THRESHOLD = 0.30
HEF_PATH = "hand_landmark_lite.hef"

# ============================================================
# SERIAL / BRACCIO (protocol: REQ/ACK + CMD,....\n)
# ============================================================

def init_braccio(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT) -> serial.Serial:
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    time.sleep(2)  # Arduino reset
    return ser

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def sanitize_braccio(cmd):
    """Aplică limite safe Braccio."""
    out = dict(cmd)
    out["base"]       = clamp(int(out["base"]),       0, 180)
    out["shoulder"]   = clamp(int(out["shoulder"]),  15, 165)
    out["elbow"]      = clamp(int(out["elbow"]),      0, 180)
    out["wrist_vert"] = clamp(int(out["wrist_vert"]), 0, 180)
    out["wrist_rot"]  = clamp(int(out["wrist_rot"]),  0, 180)
    out["gripper"]    = clamp(int(out["gripper"]),   10, 73)
    return out

def send_braccio(ser, cmd):
    cmd = sanitize_braccio(cmd)
    frame = f"CMD,{cmd['base']},{cmd['shoulder']},{cmd['elbow']},{cmd['wrist_vert']},{cmd['wrist_rot']},{cmd['gripper']}\n"
    ser.write(frame.encode("utf-8"))

# ============================================================
# HAILO (opțional) - streamuri persistente
# ============================================================

def try_init_hailo():
    try:
        from hailo_platform import (
            HEF, VDevice,
            HailoStreamInterface, ConfigureParams,
            InputVStreams, OutputVStreams,
            InputVStreamParams, OutputVStreamParams,
            FormatType
        )
    except Exception as e:
        print("[HAILO] hailo_platform import failed:", e)
        return False, None

    try:
        print("[HAILO] Initializing...")
        hef = HEF(HEF_PATH)
        device = VDevice()
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_groups = device.configure(hef, cfg)
        network_group = network_groups[0]
        net_params = network_group.create_params()

        in_params  = InputVStreamParams.make(network_group,  format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)

        in_streams  = InputVStreams(network_group, in_params)
        out_streams = OutputVStreams(network_group, out_params)
        activation  = network_group.activate(net_params)

        ctx = {
            "device": device,
            "network_group": network_group,
            "in_streams": in_streams,
            "out_streams": out_streams,
            "activation": activation,
            "input_stream": None,
            "output_stream": None,
        }

        # enter o singură dată
        ctx["in_streams"].__enter__()
        ctx["out_streams"].__enter__()
        ctx["activation"].__enter__()

        ctx["input_stream"]  = list(ctx["in_streams"])[0]
        ctx["output_stream"] = list(ctx["out_streams"])[0]

        print("[HAILO] READY.")
        return True, ctx
    except Exception as e:
        print("[HAILO] init failed:", e)
        return False, None

def close_hailo(ctx):
    if not ctx:
        return
    try: ctx["activation"].__exit__(None, None, None)
    except: pass
    try: ctx["out_streams"].__exit__(None, None, None)
    except: pass
    try: ctx["in_streams"].__exit__(None, None, None)
    except: pass

def hailo_score_frame(hailo_ctx, roi_bgr):
    img = cv2.resize(roi_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.uint8)

    hailo_ctx["input_stream"].send(img)
    raw = hailo_ctx["output_stream"].recv()

    arr = raw.astype(np.float32).flatten()
    return float(np.mean(arr) / 255.0)

# ============================================================
# Utils: mapping + smoothing + rate limiting
# ============================================================

class EMA:
    def __init__(self, alpha=0.2):
        self.alpha = float(alpha)
        self.v = None
    def update(self, x):
        x = float(x)
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def map_range(x, in_lo, in_hi, out_lo, out_hi):
    x = clamp(x, in_lo, in_hi)
    t = (x - in_lo) / (in_hi - in_lo + 1e-9)
    return out_lo + t * (out_hi - out_lo)

def significant_change(new_cmd, old_cmd, deadband):
    for k, db in deadband.items():
        if abs(int(new_cmd[k]) - int(old_cmd[k])) >= db:
            return True
    return False

def slew_limit(target, current, max_step):
    out = dict(current)
    for k, step in max_step.items():
        delta = int(target[k]) - int(current[k])
        if abs(delta) > step:
            out[k] = int(current[k]) + (step if delta > 0 else -step)
        else:
            out[k] = int(target[k])
    # chei care nu sunt în max_step (ex: wrist_rot) rămân ca în target dacă există
    for k in target:
        if k not in out:
            out[k] = int(target[k])
    return out

def lerp_to_neutral(current, neutral, t01):
    out = dict(current)
    for k in neutral:
        out[k] = int(round((1 - t01) * current[k] + t01 * neutral[k]))
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    print("=== RPi: Hand -> Braccio (VERY SMOOTH) ===")

    ser = init_braccio()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Verifică /dev/video0 și permisiunile.")

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    hailo_ok = False
    hailo_ctx = None
    if HAILO_ENABLED:
        hailo_ok, hailo_ctx = try_init_hailo()
    if not hailo_ok:
        print("[HAILO] Disabled (fallback to MediaPipe only).")

    ema = {
        "base": EMA(EMA_ALPHA["base"]),
        "shoulder": EMA(EMA_ALPHA["shoulder"]),
        "elbow": EMA(EMA_ALPHA["elbow"]),
        "wrist_vert": EMA(EMA_ALPHA["wrist_vert"]),
        "gripper": EMA(EMA_ALPHA["gripper"]),
    }

    # Ultima comandă TRIMISĂ (integer)
    last_sent = sanitize_braccio(NEUTRAL)

    # Ultima comandă “internă” (poate fi float la EMA, dar noi ținem int în final)
    last_cmd = dict(last_sent)

    last_send_t = 0.0

    # calibrare pinch (thumb-index) pentru gripper
    pinch_min = None
    pinch_max = None

    # pentru revenire lină la neutral când dispare mâna
    last_hand_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAM] frame read failed.")
                break

            h, w, _ = frame.shape
            x1, y1 = clamp(ROI_X1, 0, w-1), clamp(ROI_Y1, 0, h-1)
            x2, y2 = clamp(ROI_X2, 0, w-1), clamp(ROI_Y2, 0, h-1)
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError("Bad ROI. Check ROI coords.")

            roi = frame[y1:y2, x1:x2]
            roi_h, roi_w, _ = roi.shape

            # Hailo gating (optional)
            hailo_valid = True
            hailo_score = None
            if hailo_ok and hailo_ctx is not None:
                try:
                    hailo_score = hailo_score_frame(hailo_ctx, roi)
                    hailo_valid = (hailo_score > HAILO_SCORE_THRESHOLD)
                except Exception as e:
                    print("[HAILO] inference failed:", e)
                    hailo_valid = True

            # MediaPipe
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = mp_hands.process(roi_rgb)
            have_hand = bool(res.multi_hand_landmarks) and hailo_valid

            target_cmd = dict(last_cmd)

            if have_hand:
                last_hand_time = time.time()
                lm = res.multi_hand_landmarks[0].landmark

                def px(i):
                    return (lm[i].x * roi_w, lm[i].y * roi_h)

                # PALM CENTER (mai stabil decât index_mcp singur)
                p0  = px(0)    # wrist
                p5  = px(5)    # index_mcp
                p17 = px(17)   # pinky_mcp
                palm_x = (p0[0] + p5[0] + p17[0]) / 3.0
                palm_y = (p0[1] + p5[1] + p17[1]) / 3.0

                # Reach (aprox "adâncime") pentru elbow
                reach = dist(p0, p5)

                # pinch pentru gripper (thumb tip - index tip)
                thumb_tip = px(4)
                index_tip = px(8)
                pinch = dist(thumb_tip, index_tip)

                # auto-calibrare pinch (se stabilizează în timp)
                if pinch_min is None:
                    pinch_min = pinch
                    pinch_max = pinch
                pinch_min = min(pinch_min, pinch)
                pinch_max = max(pinch_max, pinch)

                # Mapări (tune “smooth”)
                base       = map_range(palm_x, 0, roi_w, 25, 155)
                shoulder   = map_range(palm_y, 0, roi_h, 55, 135)
                elbow      = map_range(reach,   45, 210, 135, 65)
                wrist_vert = map_range(palm_y, 0, roi_h, 115, 75)
                wrist_rot  = 90  # stabil (poți mapa din orientarea palmei dacă vrei)

                if pinch_max - pinch_min < 8:
                    gripper = NEUTRAL["gripper"]
                else:
                    # inversat? (dacă vrei: pinch mic = închis)
                    gripper = map_range(pinch, pinch_min, pinch_max, 10, 73)

                # EMA smoothing (float)
                target_cmd["base"]       = ema["base"].update(base)
                target_cmd["shoulder"]   = ema["shoulder"].update(shoulder)
                target_cmd["elbow"]      = ema["elbow"].update(elbow)
                target_cmd["wrist_vert"] = ema["wrist_vert"].update(wrist_vert)
                target_cmd["wrist_rot"]  = wrist_rot
                target_cmd["gripper"]    = ema["gripper"].update(gripper)

                # Draw landmarks
                for p in lm:
                    lx = int(p.x * roi_w)
                    ly = int(p.y * roi_h)
                    cv2.circle(roi, (lx, ly), 2, (0, 255, 0), -1)
                cv2.circle(roi, (int(palm_x), int(palm_y)), 6, (0, 255, 255), -1)
                cv2.line(
                    roi,
                    (int(thumb_tip[0]), int(thumb_tip[1])),
                    (int(index_tip[0]), int(index_tip[1])),
                    (255, 255, 0), 2
                )

            else:
                # Dacă nu e mână: nu “sări” instant la neutral; revino gradual pe ~RETURN_TO_NEUTRAL_SECONDS
                dt_since_hand = time.time() - last_hand_time
                t01 = clamp(dt_since_hand / max(RETURN_TO_NEUTRAL_SECONDS, 1e-6), 0.0, 1.0)

                neutral_like = lerp_to_neutral(
                    sanitize_braccio({
                        "base": last_cmd["base"],
                        "shoulder": last_cmd["shoulder"],
                        "elbow": last_cmd["elbow"],
                        "wrist_vert": last_cmd["wrist_vert"],
                        "wrist_rot": last_cmd["wrist_rot"],
                        "gripper": last_cmd["gripper"],
                    }),
                    NEUTRAL,
                    t01
                )

                # treci și prin EMA (ca să fie extra smooth)
                target_cmd["base"]       = ema["base"].update(neutral_like["base"])
                target_cmd["shoulder"]   = ema["shoulder"].update(neutral_like["shoulder"])
                target_cmd["elbow"]      = ema["elbow"].update(neutral_like["elbow"])
                target_cmd["wrist_vert"] = ema["wrist_vert"].update(neutral_like["wrist_vert"])
                target_cmd["wrist_rot"]  = NEUTRAL["wrist_rot"]
                target_cmd["gripper"]    = ema["gripper"].update(neutral_like["gripper"])

            # Candidate integer (după EMA)
            candidate = sanitize_braccio({
                "base": int(round(target_cmd["base"])),
                "shoulder": int(round(target_cmd["shoulder"])),
                "elbow": int(round(target_cmd["elbow"])),
                "wrist_vert": int(round(target_cmd["wrist_vert"])),
                "wrist_rot": int(round(target_cmd["wrist_rot"])),
                "gripper": int(round(target_cmd["gripper"])),
            })

            # Slew-limit față de ultima comandă TRIMISĂ (asta dă fluiditate reală)
            limited = slew_limit(candidate, last_sent, MAX_STEP)
            limited = sanitize_braccio(limited)

            # Rate-limit + Deadband: trimite rar și doar dacă are sens
            now = time.time()
            if now - last_send_t >= UPDATE_DT:
                if significant_change(limited, last_sent, DEADBAND):
                    send_braccio(ser, limited)
                    last_sent = dict(limited)
                    last_send_t = now

                # păstrează last_cmd ca float/int mix pentru logica internă
                last_cmd = {
                    "base": limited["base"],
                    "shoulder": limited["shoulder"],
                    "elbow": limited["elbow"],
                    "wrist_vert": limited["wrist_vert"],
                    "wrist_rot": limited["wrist_rot"],
                    "gripper": limited["gripper"],
                }

            # UI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            status = "HAND" if have_hand else "NO HAND"
            if hailo_score is not None:
                status += f" | Hailo={hailo_score:.2f} ({'OK' if hailo_valid else 'LOW'})"
            cv2.putText(frame, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # afișează și comanda trimisă (util pentru tuning)
            dbg = f"CMD: B{last_sent['base']} S{last_sent['shoulder']} E{last_sent['elbow']} WV{last_sent['wrist_vert']} G{last_sent['gripper']}"
            cv2.putText(frame, dbg, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand -> Braccio (Smooth)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        try:
            ser.close()
        except:
            pass
        try:
            close_hailo(hailo_ctx)
        except:
            pass

if __name__ == "__main__":
    main()
