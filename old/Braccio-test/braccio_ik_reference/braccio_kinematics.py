#!/usr/bin/env python3
"""
Simplified Braccio kinematics for tool-tip control.

This uses a reduced geometric model in the radial/vertical plane plus base
rotation. It is good enough to start Cartesian control and tool-tip
calibration, but the numbers in the template calibration file must be measured
on the real arm.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


JOINT_ORDER = (
    "base",
    "shoulder",
    "elbow",
    "wrist_vertical",
    "wrist_rotation",
    "gripper",
)

SERVO_LIMITS = {
    "base": (0.0, 180.0),
    "shoulder": (15.0, 165.0),
    "elbow": (15.0, 165.0),
    "wrist_vertical": (0.0, 180.0),
    "wrist_rotation": (0.0, 180.0),
    "gripper": (10.0, 73.0),
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_json(path: Path | str):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_calibration(path: Path | str) -> Dict[str, object]:
    return load_json(path)


def sanitize_servo_pose(pose: Dict[str, float]) -> Dict[str, int]:
    clean: Dict[str, int] = {}
    for joint in JOINT_ORDER:
        lo, hi = SERVO_LIMITS[joint]
        clean[joint] = int(round(clamp(float(pose[joint]), lo, hi)))
    return clean


def servo_to_model_deg(joint: str, servo_deg: float, calibration: Dict[str, object]) -> float:
    joint_cal = calibration["servo"][joint]
    return float(joint_cal["sign"]) * (float(servo_deg) - float(joint_cal["zero_deg"]))


def model_to_servo_deg(joint: str, model_deg: float, calibration: Dict[str, object]) -> float:
    joint_cal = calibration["servo"][joint]
    return float(joint_cal["zero_deg"]) + float(joint_cal["sign"]) * float(model_deg)


def forward_kinematics(pose: Dict[str, float], calibration: Dict[str, object]) -> Dict[str, float]:
    geometry = calibration["geometry"]

    base_azimuth = math.radians(servo_to_model_deg("base", pose["base"], calibration))
    shoulder = math.radians(servo_to_model_deg("shoulder", pose["shoulder"], calibration))
    elbow = math.radians(servo_to_model_deg("elbow", pose["elbow"], calibration))
    wrist_vertical = math.radians(
        servo_to_model_deg("wrist_vertical", pose["wrist_vertical"], calibration)
    )
    tool_roll_deg = servo_to_model_deg("wrist_rotation", pose["wrist_rotation"], calibration)

    shoulder_radius = float(geometry["shoulder_radius_mm"])
    base_height = float(geometry["base_height_mm"])
    upper_arm = float(geometry["upper_arm_mm"])
    forearm = float(geometry["forearm_mm"])
    tool_tip_offset = float(geometry["tool_tip_offset_mm"])

    elbow_sum = shoulder + elbow
    tool_pitch = elbow_sum + wrist_vertical

    wrist_r = shoulder_radius + upper_arm * math.cos(shoulder) + forearm * math.cos(elbow_sum)
    wrist_z = base_height + upper_arm * math.sin(shoulder) + forearm * math.sin(elbow_sum)

    tip_r = wrist_r + tool_tip_offset * math.cos(tool_pitch)
    tip_z = wrist_z + tool_tip_offset * math.sin(tool_pitch)

    return {
        "x_mm": tip_r * math.cos(base_azimuth),
        "y_mm": tip_r * math.sin(base_azimuth),
        "z_mm": tip_z,
        "r_mm": tip_r,
        "tool_pitch_deg": math.degrees(tool_pitch),
        "tool_roll_deg": tool_roll_deg,
    }


def inverse_kinematics(
    target: Dict[str, float],
    calibration: Dict[str, object],
    elbow_up: bool = False,
) -> Dict[str, int]:
    geometry = calibration["geometry"]

    x_mm = float(target["x_mm"])
    y_mm = float(target["y_mm"])
    z_mm = float(target["z_mm"])
    tool_pitch_deg = float(target["tool_pitch_deg"])
    tool_roll_deg = float(target.get("tool_roll_deg", calibration["tool"]["default_roll_deg"]))
    gripper_deg = float(target.get("gripper_deg", calibration["gripper"]["open_deg"]))

    base_azimuth = math.atan2(y_mm, x_mm)
    tip_r = math.hypot(x_mm, y_mm)
    tool_pitch = math.radians(tool_pitch_deg)

    shoulder_radius = float(geometry["shoulder_radius_mm"])
    base_height = float(geometry["base_height_mm"])
    upper_arm = float(geometry["upper_arm_mm"])
    forearm = float(geometry["forearm_mm"])
    tool_tip_offset = float(geometry["tool_tip_offset_mm"])

    wrist_r = tip_r - tool_tip_offset * math.cos(tool_pitch)
    wrist_z = z_mm - tool_tip_offset * math.sin(tool_pitch)

    radial = wrist_r - shoulder_radius
    vertical = wrist_z - base_height

    wrist_distance_sq = radial * radial + vertical * vertical
    cos_elbow = (
        wrist_distance_sq - upper_arm * upper_arm - forearm * forearm
    ) / (2.0 * upper_arm * forearm)

    if cos_elbow < -1.0 or cos_elbow > 1.0:
        raise ValueError(
            f"Target is outside reachable workspace: r={tip_r:.1f} z={z_mm:.1f} pitch={tool_pitch_deg:.1f}"
        )

    sin_elbow = math.sqrt(max(0.0, 1.0 - cos_elbow * cos_elbow))
    if elbow_up:
        sin_elbow = -sin_elbow

    elbow = math.atan2(sin_elbow, cos_elbow)
    shoulder = math.atan2(vertical, radial) - math.atan2(
        forearm * math.sin(elbow), upper_arm + forearm * math.cos(elbow)
    )
    wrist_vertical = tool_pitch - shoulder - elbow

    servo_pose = {
        "base": model_to_servo_deg("base", math.degrees(base_azimuth), calibration),
        "shoulder": model_to_servo_deg("shoulder", math.degrees(shoulder), calibration),
        "elbow": model_to_servo_deg("elbow", math.degrees(elbow), calibration),
        "wrist_vertical": model_to_servo_deg(
            "wrist_vertical", math.degrees(wrist_vertical), calibration
        ),
        "wrist_rotation": model_to_servo_deg("wrist_rotation", tool_roll_deg, calibration),
        "gripper": gripper_deg,
    }
    return sanitize_servo_pose(servo_pose)


def _sample_pose(sample: Dict[str, object]) -> Dict[str, float]:
    if "pose" in sample:
        return {joint: float(sample["pose"][joint]) for joint in JOINT_ORDER}
    return {joint: float(sample[joint]) for joint in JOINT_ORDER}


def same_point_rmse(samples: Iterable[Dict[str, object]], calibration: Dict[str, object]) -> float:
    points = [forward_kinematics(_sample_pose(sample), calibration) for sample in samples]
    if not points:
        raise ValueError("Need at least one sample")

    cx = sum(point["x_mm"] for point in points) / len(points)
    cy = sum(point["y_mm"] for point in points) / len(points)
    cz = sum(point["z_mm"] for point in points) / len(points)

    mean_square = 0.0
    for point in points:
        dx = point["x_mm"] - cx
        dy = point["y_mm"] - cy
        dz = point["z_mm"] - cz
        mean_square += dx * dx + dy * dy + dz * dz
    mean_square /= len(points)
    return math.sqrt(mean_square)


def fit_tool_tip_offset(
    samples: List[Dict[str, object]],
    calibration: Dict[str, object],
    tool_min_mm: float = 40.0,
    tool_max_mm: float = 180.0,
    tool_step_mm: float = 1.0,
    wrist_trim_min_deg: float = -20.0,
    wrist_trim_max_deg: float = 20.0,
    wrist_trim_step_deg: float = 1.0,
) -> Dict[str, float]:
    base_wrist_zero = float(calibration["servo"]["wrist_vertical"]["zero_deg"])
    best = None

    tool_offset = tool_min_mm
    while tool_offset <= tool_max_mm + 1e-9:
        trim_deg = wrist_trim_min_deg
        while trim_deg <= wrist_trim_max_deg + 1e-9:
            trial = copy.deepcopy(calibration)
            trial["geometry"]["tool_tip_offset_mm"] = tool_offset
            trial["servo"]["wrist_vertical"]["zero_deg"] = base_wrist_zero + trim_deg
            error = same_point_rmse(samples, trial)
            if best is None or error < best["rmse_mm"]:
                best = {
                    "rmse_mm": error,
                    "tool_tip_offset_mm": tool_offset,
                    "wrist_vertical_zero_deg": trial["servo"]["wrist_vertical"]["zero_deg"],
                }
            trim_deg += wrist_trim_step_deg
        tool_offset += tool_step_mm

    return best
