#!/usr/bin/env python3
"""
Hand -> operation-zone Cartesian control using inverse kinematics.

This keeps the operation pattern in Cartesian space and converts each target
into Braccio joint angles through the simplified kinematics module.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import serial

from braccio_kinematics import inverse_kinematics, load_calibration, sanitize_servo_pose


SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
READ_ACK = True

CAM_INDEX = 0
SEND_INTERVAL = 0.08
HEARTBEAT_INTERVAL = 0.50
HAND_LOST_TIMEOUT = 0.45

TARGET_HAND_LABEL = "Right"
FAILSAFE_MODE = "NEUTRAL"

TRIGGER_PINCH_THRESHOLD = 0.26
TRIGGER_RELEASE_THRESHOLD = 0.40
TRIGGER_HOLD_SECONDS = 0.80

JOINT_ORDER = (
    "base",
    "shoulder",
    "elbow",
    "wrist_vertical",
    "wrist_rotation",
    "gripper",
)

MAX_STEP = {
    "base": 3,
    "shoulder": 2,
    "elbow": 2,
    "wrist_vertical": 3,
    "wrist_rotation": 3,
    "gripper": 2,
}

SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVE_CALIBRATION = SCRIPT_DIR / "braccio_calibration.json"
TEMPLATE_CALIBRATION = SCRIPT_DIR / "braccio_calibration.template.json"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def map_range(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    if abs(in_hi - in_lo) < 1e-9:
        return out_lo
    value = clamp(value, in_lo, in_hi)
    t = (value - in_lo) / (in_hi - in_lo)
    return out_lo + t * (out_hi - out_lo)


def build_packet(pose: Dict[str, int]) -> str:
    return (
        f"CMD,{pose['base']},{pose['shoulder']},{pose['elbow']},"
        f"{pose['wrist_vertical']},{pose['wrist_rotation']},{pose['gripper']}\n"
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


def rate_limit_pose(current: Dict[str, int], target: Dict[str, int]) -> Dict[str, int]:
    limited: Dict[str, int] = {}
    for joint in JOINT_ORDER:
        delta = int(target[joint]) - int(current[joint])
        step = MAX_STEP[joint]
        if delta > step:
            limited[joint] = int(current[joint]) + step
        elif delta < -step:
            limited[joint] = int(current[joint]) - step
        else:
            limited[joint] = int(target[joint])
    return limited


def pose_changed(a: Dict[str, int], b: Dict[str, int], threshold: int = 1) -> bool:
    return any(abs(int(a[j]) - int(b[j])) >= threshold for j in JOINT_ORDER)


def select_target_hand(results, wanted: str):
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None, None, None

    for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = hand_info.classification[0].label
        if label == wanted:
            return hand_lm, hand_lm.landmark, label

    if len(results.multi_hand_landmarks) == 1:
        hand_lm = results.multi_hand_landmarks[0]
        label = results.multi_handedness[0].classification[0].label
        return hand_lm, hand_lm.landmark, label

    return None, None, None


def compute_hand_target(landmarks, frame_w: int, frame_h: int, calibration: Dict[str, object]):
    def pt(index: int) -> Tuple[float, float]:
        return (landmarks[index].x * frame_w, landmarks[index].y * frame_h)

    wrist = pt(0)
    thumb_tip = pt(4)
    index_tip = pt(8)
    index_mcp = pt(5)
    middle_mcp = pt(9)
    pinky_mcp = pt(17)

    palm_x = (wrist[0] + index_mcp[0] + pinky_mcp[0]) / 3.0
    palm_y = (wrist[1] + index_mcp[1] + pinky_mcp[1]) / 3.0

    hand_ref = dist(wrist, middle_mcp) + 1e-6
    pinch_norm = dist(thumb_tip, index_tip) / hand_ref

    hand_roll_deg = map_range(
        (pinky_mcp[1] - index_mcp[1]),
        -frame_h * 0.25,
        frame_h * 0.25,
        -25.0,
        25.0,
    )
    palm_pitch_deg = map_range(
        middle_mcp[1] - wrist[1],
        -frame_h * 0.20,
        frame_h * 0.25,
        -12.0,
        12.0,
    )

    zone = calibration["operation_zone"]
    tool = calibration["tool"]
    gripper = calibration["gripper"]

    hover_z = float(zone["surface_z_mm"]) + float(zone["hover_height_mm"])
    target = {
        "x_mm": map_range(palm_y, frame_h * 0.15, frame_h * 0.90, zone["x_max_mm"], zone["x_min_mm"]),
        "y_mm": map_range(palm_x, 0, frame_w, zone["y_min_mm"], zone["y_max_mm"]),
        "z_mm": hover_z,
        "tool_pitch_deg": float(tool["default_pitch_deg"]) + palm_pitch_deg,
        "tool_roll_deg": float(tool["default_roll_deg"]) + hand_roll_deg,
        "gripper_deg": map_range(
            pinch_norm,
            0.20,
            0.95,
            gripper["closed_deg"],
            gripper["open_deg"],
        ),
    }
    return target, pinch_norm


def build_schema_path(anchor: Dict[str, float], calibration: Dict[str, object]):
    zone = calibration["operation_zone"]
    hover_z = float(zone["surface_z_mm"]) + float(zone["hover_height_mm"])
    entry_z = float(zone["surface_z_mm"]) - float(zone["entry_depth_mm"])
    step_right = float(zone["step_right_mm"])

    sequence = []
    current_y = float(anchor["y_mm"])
    for index in range(3):
        down_target = dict(anchor)
        down_target["y_mm"] = current_y
        down_target["z_mm"] = entry_z
        sequence.append(
            {
                "name": f"down_{index + 1}",
                "duration": 0.55,
                "target": down_target,
            }
        )

        if index == 2:
            recover_target = dict(anchor)
            recover_target["y_mm"] = current_y
            recover_target["z_mm"] = hover_z
            sequence.append(
                {
                    "name": "recover",
                    "duration": 0.55,
                    "target": recover_target,
                }
            )
            break

        current_y += step_right
        lift_target = dict(anchor)
        lift_target["y_mm"] = current_y
        lift_target["z_mm"] = hover_z
        sequence.append(
            {
                "name": f"up_right_{index + 1}",
                "duration": 0.70,
                "target": lift_target,
            }
        )

    return tuple(sequence)


class CartesianPathRunner:
    def __init__(self):
        self.sequence = ()
        self.active = False
        self.index = 0
        self.phase_name = "idle"
        self.started_at = 0.0
        self.start_target: Dict[str, float] = {}
        self.end_target: Dict[str, float] = {}

    def start(self, start_target: Dict[str, float], sequence, now: float):
        self.sequence = sequence
        self.active = True
        self.index = 0
        self.phase_name = self.sequence[0]["name"]
        self.started_at = now
        self.start_target = dict(start_target)
        self.end_target = dict(self.sequence[0]["target"])

    def abort(self):
        self.active = False
        self.phase_name = "aborted"

    def update(self, now: float):
        if not self.active:
            return dict(self.end_target), self.phase_name, False

        segment = self.sequence[self.index]
        duration = max(float(segment["duration"]), 1e-3)
        t = clamp((now - self.started_at) / duration, 0.0, 1.0)

        target = {}
        for key in self.end_target:
            target[key] = (1.0 - t) * float(self.start_target[key]) + t * float(self.end_target[key])

        if t < 1.0:
            return target, self.phase_name, False

        self.index += 1
        if self.index >= len(self.sequence):
            self.active = False
            self.phase_name = "done"
            return dict(self.end_target), self.phase_name, True

        self.started_at = now
        self.start_target = dict(self.end_target)
        self.end_target = dict(self.sequence[self.index]["target"])
        self.phase_name = self.sequence[self.index]["name"]
        return dict(self.start_target), self.phase_name, False


def load_active_calibration() -> Dict[str, object]:
    path = ACTIVE_CALIBRATION if ACTIVE_CALIBRATION.exists() else TEMPLATE_CALIBRATION
    return load_calibration(path)


def main():
    calibration = load_active_calibration()
    neutral_pose = sanitize_servo_pose(calibration["neutral_pose"])
    elbow_up = bool(calibration.get("ik", {}).get("elbow_up", False))

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

    current_pose = dict(neutral_pose)
    current_cartesian = {
        "x_mm": calibration["operation_zone"]["x_min_mm"],
        "y_mm": 0.0,
        "z_mm": float(calibration["operation_zone"]["surface_z_mm"])
        + float(calibration["operation_zone"]["hover_height_mm"]),
        "tool_pitch_deg": calibration["tool"]["default_pitch_deg"],
        "tool_roll_deg": calibration["tool"]["default_roll_deg"],
        "gripper_deg": calibration["gripper"]["open_deg"],
    }

    last_send_time = 0.0
    last_heartbeat_time = 0.0
    last_hand_seen_time = time.time()
    trigger_started_at: Optional[float] = None
    trigger_armed = True
    path_runner = CartesianPathRunner()

    print("[INFO] Keys: ESC quit | S start path")

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
            pinch_norm = None
            mode = "FOLLOW"
            phase = "hover"
            manual_target = dict(current_cartesian)

            if hand_detected:
                last_hand_seen_time = now
                manual_target, pinch_norm = compute_hand_target(
                    selected_landmarks, frame_w, frame_h, calibration
                )
                mp_draw.draw_landmarks(frame, selected_hand, mp_hands.HAND_CONNECTIONS)

                if not path_runner.active:
                    if pinch_norm <= TRIGGER_PINCH_THRESHOLD and trigger_armed:
                        if trigger_started_at is None:
                            trigger_started_at = now
                        elif now - trigger_started_at >= TRIGGER_HOLD_SECONDS:
                            sequence = build_schema_path(manual_target, calibration)
                            path_runner.start(manual_target, sequence, now)
                            trigger_armed = False
                            trigger_started_at = None
                    elif pinch_norm >= TRIGGER_RELEASE_THRESHOLD:
                        trigger_started_at = None
                        trigger_armed = True
            else:
                trigger_started_at = None

            if path_runner.active and (now - last_hand_seen_time) > HAND_LOST_TIMEOUT:
                path_runner.abort()

            if path_runner.active:
                current_cartesian, phase, _ = path_runner.update(now)
                mode = "PATTERN"
            elif hand_detected:
                current_cartesian = dict(manual_target)
            else:
                time_since_hand = now - last_hand_seen_time
                if time_since_hand > HAND_LOST_TIMEOUT:
                    mode = "FAILSAFE"
                    phase = FAILSAFE_MODE.lower()
                    if FAILSAFE_MODE == "NEUTRAL":
                        desired_pose = dict(neutral_pose)
                    else:
                        desired_pose = dict(current_pose)
                    candidate = desired_pose
                else:
                    mode = "WAIT"
                    phase = "hold"
                    candidate = dict(current_pose)

            if hand_detected or path_runner.active:
                try:
                    candidate = inverse_kinematics(
                        current_cartesian, calibration, elbow_up=elbow_up
                    )
                except ValueError as exc:
                    mode = "UNREACHABLE"
                    phase = str(exc)
                    candidate = dict(current_pose)

            limited = rate_limit_pose(current_pose, sanitize_servo_pose(candidate))

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

            pinch_text = "---" if pinch_norm is None else f"{pinch_norm:.2f}"
            ack_text = ack if ack is not None else "-"
            xyz_text = (
                f"X{current_cartesian['x_mm']:.1f} Y{current_cartesian['y_mm']:.1f} "
                f"Z{current_cartesian['z_mm']:.1f}"
            )
            cv2.putText(
                frame,
                f"Mode: {mode} | Phase: {phase}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0) if hand_detected else (0, 180, 255),
                2,
            )
            cv2.putText(
                frame,
                xyz_text,
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Hand: {detected_label or 'none'} | Pinch: {pinch_text} | ACK: {ack_text}",
                (10, 79),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Pinch and hold to run Cartesian path, release to re-arm, S for manual start",
                (10, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Braccio Cartesian IK", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("s"), ord("S")) and not path_runner.active:
                sequence = build_schema_path(current_cartesian, calibration)
                path_runner.start(current_cartesian, sequence, time.time())

    finally:
        cap.release()
        try:
            arduino.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
