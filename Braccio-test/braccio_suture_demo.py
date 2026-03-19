#!/usr/bin/env python3
"""
Reference controller for a Braccio suturing demo.

This is for bench simulation and training only. It is not suitable for real
medical use. The goal is to keep manual hand-follow for coarse positioning and
trigger a repeatable stitch-like motion as a guarded state machine.
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

TRIGGER_PINCH_THRESHOLD = 0.26
TRIGGER_RELEASE_THRESHOLD = 0.40
TRIGGER_HOLD_SECONDS = 0.80

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
    "shoulder": 83,
    "elbow": 122,
    "wrist_vertical": 93,
    "wrist_rotation": 160,
    "gripper": 10,
}

FIXED_MANUAL_POSE = {
    "base": NEUTRAL_POSE["base"],
    "shoulder": NEUTRAL_POSE["shoulder"],
    "wrist_vertical": NEUTRAL_POSE["wrist_vertical"],
    "wrist_rotation": NEUTRAL_POSE["wrist_rotation"],
}

LOCKED_JOINTS = dict(FIXED_MANUAL_POSE)

ELBOW_IDLE_SERVO = NEUTRAL_POSE["elbow"]
ELBOW_LOWER_SERVO = 82
HAND_DROP_DEADBAND_RATIO = 0.04
HAND_FULL_DROP_RATIO = 0.32
GRIPPER_OPEN_SERVO = 10
GRIPPER_CLOSED_SERVO = 73

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
    "gripper": 2,
}

PATTERN_REPETITIONS = 3
PATTERN_HORIZONTAL_STEP = 10
PATTERN_DOWN_OFFSETS = {
    "shoulder": 10,
    "elbow": -10,
    "wrist_vertical": 24,
    "gripper": -3,
}
PATTERN_UP_RIGHT_OFFSETS = {
    "shoulder": -8,
    "elbow": 7,
    "wrist_vertical": -18,
    "wrist_rotation": 6,
}


def build_schema_sequence():
    sequence = []
    base_offset = 0

    for index in range(PATTERN_REPETITIONS):
        entry_offsets = {"base": base_offset}
        entry_offsets.update(PATTERN_DOWN_OFFSETS)
        sequence.append(
            {
                "name": f"down_{index + 1}",
                "duration": 0.55,
                "offsets": entry_offsets,
            }
        )

        if index == PATTERN_REPETITIONS - 1:
            sequence.append(
                {
                    "name": "recover",
                    "duration": 0.55,
                    "offsets": {"base": base_offset},
                }
            )
            break

        base_offset += PATTERN_HORIZONTAL_STEP
        lift_offsets = {"base": base_offset}
        lift_offsets.update(PATTERN_UP_RIGHT_OFFSETS)
        sequence.append(
            {
                "name": f"up_right_{index + 1}",
                "duration": 0.70,
                "offsets": lift_offsets,
            }
        )

    return tuple(sequence)


STITCH_SEQUENCE = build_schema_sequence()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sanitize_pose(pose: Dict[str, float]) -> Dict[str, int]:
    clean: Dict[str, int] = {}
    for joint in JOINTS:
        lo, hi = SERVO_LIMITS[joint]
        clean[joint] = int(round(clamp(pose[joint], lo, hi)))
    return clean


def apply_locked_joints(pose: Dict[str, float]) -> Dict[str, float]:
    locked = dict(pose)
    for joint, fixed_value in LOCKED_JOINTS.items():
        locked[joint] = fixed_value
    return locked


def lerp_pose(start: Dict[str, float], end: Dict[str, float], t: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for joint in JOINTS:
        out[joint] = (1.0 - t) * float(start[joint]) + t * float(end[joint])
    return out


def offset_pose(anchor: Dict[str, float], offsets: Dict[str, float]) -> Dict[str, float]:
    out = dict(anchor)
    for joint, delta in offsets.items():
        out[joint] = float(out[joint]) + float(delta)
    return out


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


def map_range(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    if abs(in_hi - in_lo) < 1e-9:
        return out_lo
    value = clamp(value, in_lo, in_hi)
    t = (value - in_lo) / (in_hi - in_lo)
    return out_lo + t * (out_hi - out_lo)


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def compute_hand_features(landmarks, frame_w: int, frame_h: int):
    def pt(index: int) -> Tuple[float, float]:
        return (landmarks[index].x * frame_w, landmarks[index].y * frame_h)

    wrist = pt(0)
    thumb_tip = pt(4)
    index_tip = pt(8)
    index_mcp = pt(5)
    middle_mcp = pt(9)
    pinky_mcp = pt(17)

    palm_y = (wrist[1] + index_mcp[1] + pinky_mcp[1]) / 3.0

    hand_ref = dist(wrist, middle_mcp) + 1e-6
    pinch_norm = dist(thumb_tip, index_tip) / hand_ref
    return palm_y, pinch_norm


def compute_manual_target(palm_y: float, pinch_norm: float, hand_reference_y: float, frame_h: int):
    down_deadband_px = frame_h * HAND_DROP_DEADBAND_RATIO
    full_drop_px = frame_h * HAND_FULL_DROP_RATIO
    downward_motion_px = max(0.0, palm_y - hand_reference_y - down_deadband_px)
    target = {
        "base": FIXED_MANUAL_POSE["base"],
        "shoulder": FIXED_MANUAL_POSE["shoulder"],
        "elbow": map_range(
            downward_motion_px,
            0.0,
            full_drop_px,
            ELBOW_IDLE_SERVO,
            ELBOW_LOWER_SERVO,
        ),
        "wrist_vertical": FIXED_MANUAL_POSE["wrist_vertical"],
        "wrist_rotation": FIXED_MANUAL_POSE["wrist_rotation"],
        "gripper": map_range(
            pinch_norm,
            0.20,
            0.95,
            GRIPPER_CLOSED_SERVO,
            GRIPPER_OPEN_SERVO,
        ),
    }
    return target


class StitchController:
    def __init__(self, sequence):
        self.sequence = sequence
        self.active = False
        self.anchor: Dict[str, int] = dict(NEUTRAL_POSE)
        self.segment_index = 0
        self.segment_started_at = 0.0
        self.segment_start_pose: Dict[str, float] = dict(NEUTRAL_POSE)
        self.segment_end_pose: Dict[str, float] = dict(NEUTRAL_POSE)
        self.phase_name = "idle"

    def _segment_target(self, index: int) -> Dict[str, int]:
        return sanitize_pose(
            apply_locked_joints(offset_pose(self.anchor, self.sequence[index]["offsets"]))
        )

    def start(self, anchor_pose: Dict[str, float], now: float):
        self.anchor = sanitize_pose(apply_locked_joints(anchor_pose))
        self.active = True
        self.segment_index = 0
        self.segment_started_at = now
        self.segment_start_pose = dict(self.anchor)
        self.segment_end_pose = dict(self._segment_target(0))
        self.phase_name = self.sequence[0]["name"]

    def abort(self):
        self.active = False
        self.phase_name = "aborted"

    def update(self, now: float) -> Tuple[Dict[str, float], str, bool]:
        if not self.active:
            return dict(self.anchor), self.phase_name, False

        segment = self.sequence[self.segment_index]
        duration = max(float(segment["duration"]), 1e-3)
        elapsed = now - self.segment_started_at
        t = clamp(elapsed / duration, 0.0, 1.0)
        pose = lerp_pose(self.segment_start_pose, self.segment_end_pose, t)

        if t < 1.0:
            return pose, self.phase_name, False

        self.segment_index += 1
        if self.segment_index >= len(self.sequence):
            self.active = False
            self.phase_name = "done"
            return dict(self.segment_end_pose), self.phase_name, True

        self.segment_started_at = now
        self.segment_start_pose = dict(self.segment_end_pose)
        self.segment_end_pose = dict(self._segment_target(self.segment_index))
        self.phase_name = self.sequence[self.segment_index]["name"]
        return dict(self.segment_start_pose), self.phase_name, False


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
    hand_reference_y: Optional[float] = None
    trigger_started_at: Optional[float] = None
    trigger_armed = True
    stitch = StitchController(STITCH_SEQUENCE)

    print("[INFO] Keys: ESC quit | S start stitch | R recalibrate hand height")

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
            manual_target = dict(current_pose)
            pinch_norm = None
            mode = "FOLLOW"
            phase = "manual"

            if hand_detected:
                last_hand_seen_time = now
                palm_y, pinch_norm = compute_hand_features(selected_landmarks, frame_w, frame_h)
                if hand_reference_y is None:
                    hand_reference_y = palm_y
                manual_target = compute_manual_target(
                    palm_y, pinch_norm, hand_reference_y, frame_h
                )
                smooth_pose(smoothed_follow, manual_target)
                mp_draw.draw_landmarks(frame, selected_hand, mp_hands.HAND_CONNECTIONS)

                if not stitch.active:
                    if pinch_norm <= TRIGGER_PINCH_THRESHOLD and trigger_armed:
                        if trigger_started_at is None:
                            trigger_started_at = now
                        elif now - trigger_started_at >= TRIGGER_HOLD_SECONDS:
                            stitch.start(smoothed_follow, now)
                            trigger_armed = False
                            trigger_started_at = None
                    elif pinch_norm >= TRIGGER_RELEASE_THRESHOLD:
                        trigger_started_at = None
                        trigger_armed = True
                phase = "manual"
            else:
                trigger_started_at = None
                hand_reference_y = None

            if stitch.active and (now - last_hand_seen_time) > HAND_LOST_TIMEOUT:
                stitch.abort()

            if stitch.active:
                desired_pose, phase, _ = stitch.update(now)
                mode = "STITCH"
            elif hand_detected:
                desired_pose = dict(smoothed_follow)
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

            candidate = sanitize_pose(apply_locked_joints(desired_pose))
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

            pinch_text = "---" if pinch_norm is None else f"{pinch_norm:.2f}"
            ack_text = ack if ack is not None else "-"
            ref_text = "---" if hand_reference_y is None else f"{hand_reference_y:.0f}"
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
                f"B{current_pose['base']} S{current_pose['shoulder']} E{current_pose['elbow']} "
                f"WV{current_pose['wrist_vertical']} WR{current_pose['wrist_rotation']} "
                f"G{current_pose['gripper']}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Pinch: {pinch_text} | RefY: {ref_text} | ACK: {ack_text}",
                (10, 79),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Hand starts at reference height. Move hand down to lower elbow. R resets reference.",
                (10, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Braccio Suture Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")) and hand_detected:
                hand_reference_y = palm_y
            if key in (ord("s"), ord("S")) and not stitch.active:
                stitch.start(current_pose, time.time())

    finally:
        cap.release()
        try:
            arduino.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
