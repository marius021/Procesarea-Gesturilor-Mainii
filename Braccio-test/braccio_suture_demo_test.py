#!/usr/bin/env python3
"""
Braccio test demo: hand height controls elbow, finger opening controls gripper.

This keeps base, shoulder, and wrist joints fixed so you can test a simple
vertical motion plus gripper actuation from the camera feed.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import serial


SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
READ_ACK = True

CAM_INDEX = 0
SEND_INTERVAL = 0.08
HEARTBEAT_INTERVAL = 0.50
HAND_LOST_TIMEOUT = 0.45

TARGET_HAND_LABEL = "Left"
FAILSAFE_MODE = "NEUTRAL"  # "HOLD" or "NEUTRAL"

JOINTS = (
    "base",
    "shoulder",
    "elbow",
    "wrist_vertical",
    "wrist_rotation",
    "gripper",
)

SERVO_LIMITS = {
    "base": (0, 180),
    "shoulder": (15, 165),
    "elbow": (15, 165),
    "wrist_vertical": (0, 180),
    "wrist_rotation": (0, 180),
    "gripper": (10, 73),
}

NEUTRAL_POSE = {
    "base": 96,
    "shoulder": 45,
    "elbow": 65,
    "wrist_vertical": 93,
    "wrist_rotation": 90,
    "gripper": 15,
}

FIXED_JOINTS = {
    "base": NEUTRAL_POSE["base"],
    "shoulder": NEUTRAL_POSE["shoulder"],
    "wrist_vertical": NEUTRAL_POSE["wrist_vertical"],
    "wrist_rotation": NEUTRAL_POSE["wrist_rotation"],
}

HAND_ACTIVE_TOP_RATIO = 0.18
HAND_ACTIVE_BOTTOM_RATIO = 0.88
ELBOW_UP_SERVO = 110
ELBOW_DOWN_SERVO = 25

FOLLOW_ALPHA = {
    "base": 0.18,
    "shoulder": 0.16,
    "elbow": 0.16,
    "wrist_vertical": 0.18,
    "wrist_rotation": 0.20,
    "gripper": 0.25,
}

MAX_STEP = {
    "base": 3,
    "shoulder": 2,
    "elbow": 2,
    "wrist_vertical": 3,
    "wrist_rotation": 3,
    "gripper": 3,
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def map_range(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    if abs(in_hi - in_lo) < 1e-9:
        return out_lo
    value = clamp(value, in_lo, in_hi)
    t = (value - in_lo) / (in_hi - in_lo)
    return out_lo + t * (out_hi - out_lo)


def sanitize_pose(pose: Dict[str, float]) -> Dict[str, int]:
    clean: Dict[str, int] = {}
    for joint in JOINTS:
        lo, hi = SERVO_LIMITS[joint]
        clean[joint] = int(round(clamp(pose[joint], lo, hi)))
    return clean


def apply_fixed_joints(pose: Dict[str, float]) -> Dict[str, float]:
    locked = dict(pose)
    for joint, fixed_value in FIXED_JOINTS.items():
        locked[joint] = fixed_value
    return locked


def smooth_pose(state: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
    for joint in JOINTS:
        alpha = FOLLOW_ALPHA[joint]
        state[joint] = (1.0 - alpha) * state[joint] + alpha * float(target[joint])
    return state


def rate_limit_pose(current: Dict[str, int], target: Dict[str, int]) -> Dict[str, int]:
    limited: Dict[str, int] = {}
    for joint in JOINTS:
        step = MAX_STEP[joint]
        delta = int(target[joint]) - int(current[joint])
        if delta > step:
            limited[joint] = int(current[joint]) + step
        elif delta < -step:
            limited[joint] = int(current[joint]) - step
        else:
            limited[joint] = int(target[joint])
    return limited


def pose_changed(a: Dict[str, int], b: Dict[str, int], threshold: int = 1) -> bool:
    return any(abs(int(a[j]) - int(b[j])) >= threshold for j in JOINTS)


def build_packet(pose: Dict[str, int]) -> str:
    # The Braccio Arduino sketch expects wrist rotation before wrist vertical.
    return (
        f"CMD,{pose['base']},{pose['shoulder']},{pose['elbow']},"
        f"{pose['wrist_rotation']},{pose['wrist_vertical']},{pose['gripper']}\n"
    )


def send_pose(ser: serial.Serial, pose: Dict[str, int]) -> Optional[str]:
    ser.write(build_packet(pose).encode("utf-8"))
    if not READ_ACK:
        return None
    try:
        ack = ser.readline().decode("utf-8", errors="ignore").strip()
        return ack or None
    except Exception:
        return None


def angle_3pts(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    dot = bax * bcx + bay * bcy
    norm_ba = math.hypot(bax, bay)
    norm_bc = math.hypot(bcx, bcy)
    cosine = dot / (norm_ba * norm_bc + 1e-9)
    cosine = clamp(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def thumb_pinky_mcp_angle(
    thumb_tip: Tuple[float, float],
    wrist: Tuple[float, float],
    pinky_mcp: Tuple[float, float],
) -> float:
    return angle_3pts(thumb_tip, wrist, pinky_mcp)


def map_fingers_to_m6(finger_angle: float) -> int:
    finger_angle = clamp(finger_angle, 15.0, 53.0)
    norm = (finger_angle - 15.0) / (53.0 - 15.0)
    norm = 1.0 - norm
    m6 = 15.0 + norm * (73.0 - 15.0)
    return int(round(m6))


def select_target_hand(results, label_wanted: str):
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None, None, None

    for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = hand_info.classification[0].label
        if label == label_wanted:
            return hand_lm, hand_lm.landmark, label

    if len(results.multi_hand_landmarks) == 1:
        hand_lm = results.multi_hand_landmarks[0]
        label = results.multi_handedness[0].classification[0].label
        return hand_lm, hand_lm.landmark, label

    return None, None, None


def compute_hand_features(landmarks, frame_w: int, frame_h: int) -> Tuple[float, float]:
    def pt(index: int) -> Tuple[float, float]:
        return (landmarks[index].x * frame_w, landmarks[index].y * frame_h)

    wrist = pt(0)
    thumb_tip = pt(4)
    index_mcp = pt(5)
    pinky_mcp = pt(17)

    palm_y = (wrist[1] + index_mcp[1] + pinky_mcp[1]) / 3.0
    finger_angle = thumb_pinky_mcp_angle(thumb_tip, wrist, pinky_mcp)
    return palm_y, finger_angle


def compute_manual_target(palm_y: float, finger_angle: float, frame_h: int) -> Dict[str, float]:
    top_px = frame_h * HAND_ACTIVE_TOP_RATIO
    bottom_px = frame_h * HAND_ACTIVE_BOTTOM_RATIO
    return {
        "base": FIXED_JOINTS["base"],
        "shoulder": FIXED_JOINTS["shoulder"],
        "elbow": map_range(
            palm_y,
            top_px,
            bottom_px,
            ELBOW_UP_SERVO,
            ELBOW_DOWN_SERVO,
        ),
        "wrist_vertical": FIXED_JOINTS["wrist_vertical"],
        "wrist_rotation": FIXED_JOINTS["wrist_rotation"],
        "gripper": map_fingers_to_m6(finger_angle),
    }


def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.70,
    )
    mp_draw = mp.solutions.drawing_utils

    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.10)
        time.sleep(1.5)
        arduino.reset_input_buffer()
    except Exception as exc:
        raise RuntimeError(f"Could not open {SERIAL_PORT}: {exc}") from exc

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    current_pose = dict(NEUTRAL_POSE)
    smoothed_follow = {joint: float(NEUTRAL_POSE[joint]) for joint in JOINTS}
    last_send_time = 0.0
    last_heartbeat_time = 0.0
    last_hand_seen_time = time.time()

    print("[INFO] Keys: ESC quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera read failed.")
                break

            now = time.time()
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            selected_hand, selected_landmarks, detected_label = select_target_hand(
                results, TARGET_HAND_LABEL
            )

            hand_detected = selected_hand is not None
            palm_y: Optional[float] = None
            finger_angle: Optional[float] = None
            mode = "VERTICAL_FOLLOW"
            phase = "manual"

            if hand_detected:
                last_hand_seen_time = now
                palm_y, finger_angle = compute_hand_features(selected_landmarks, frame_w, frame_h)
                manual_target = compute_manual_target(palm_y, finger_angle, frame_h)
                smooth_pose(smoothed_follow, manual_target)
                desired_pose = dict(smoothed_follow)
                mp_draw.draw_landmarks(frame, selected_hand, mp_hands.HAND_CONNECTIONS)
            else:
                time_since_hand = now - last_hand_seen_time
                if time_since_hand > HAND_LOST_TIMEOUT:
                    mode = "FAILSAFE"
                    phase = FAILSAFE_MODE.lower()
                    if FAILSAFE_MODE == "NEUTRAL":
                        desired_pose = dict(NEUTRAL_POSE)
                    else:
                        desired_pose = dict(current_pose)
                else:
                    mode = "WAIT"
                    phase = "hold"
                    desired_pose = dict(current_pose)

            candidate = sanitize_pose(apply_fixed_joints(desired_pose))
            limited = sanitize_pose(rate_limit_pose(current_pose, candidate))

            should_send = False
            if now - last_send_time >= SEND_INTERVAL and pose_changed(limited, current_pose):
                should_send = True
            if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                should_send = True

            ack = None
            if should_send:
                ack = send_pose(arduino, limited)
                current_pose = dict(limited)
                last_send_time = now
                last_heartbeat_time = now
                if finger_angle is not None:
                    print(
                        f"[ROLLING] finger_angle={finger_angle:.1f} "
                        f"gripper={current_pose['gripper']} ack={ack or '-'}"
                    )

            ack_text = ack if ack is not None else "-"
            palm_text = "---" if palm_y is None else f"{palm_y:.0f}"
            finger_text = "---" if finger_angle is None else f"{finger_angle:.1f}"

            cv2.putText(
                frame,
                f"Mode: {mode} | Phase: {phase} | Hand: {detected_label or 'none'}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0) if hand_detected else (0, 180, 255),
                2,
            )
            cv2.putText(
                frame,
                f"E{current_pose['elbow']} G{current_pose['gripper']} | PalmY: {palm_text} | Angle: {finger_text}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Elbow target: {candidate['elbow']} | Grip target: {candidate['gripper']} | ACK: {ack_text}",
                (10, 79),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Move hand up/down for elbow. Open/close thumb and index for gripper.",
                (10, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Braccio Suture Demo Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cap.release()
        try:
            arduino.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
