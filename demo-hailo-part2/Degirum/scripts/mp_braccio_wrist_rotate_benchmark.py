#!/usr/bin/env python3
"""
Benchmark copy of the wrist-rotate demo using MediaPipe instead of DeGirum/Hailo.

This keeps the original demos untouched and mirrors the Hailo benchmark CSV
format so the runs are easy to compare later.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "benchmark_results"
DEFAULT_SUMMARY_CSV = DEFAULT_RESULTS_DIR / "dg_wrist_rotate_summary.csv"
DEFAULT_MEDIAPIPE_LABEL = "mediapipe_hands"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import braccio_suture_demo_test as cfg


FRAME_FIELDNAMES = [
    "run_id",
    "run_label",
    "frame_index",
    "measured",
    "elapsed_measurement_s",
    "capture_time_ms",
    "detector_time_ms",
    "landmarks_time_ms",
    "control_time_ms",
    "serial_time_ms",
    "display_time_ms",
    "inference_time_ms",
    "pipeline_time_ms",
    "total_frame_time_ms",
    "fps",
    "hand_detected",
    "landmarks_found",
    "hands_found",
    "pose_sent",
    "chosen_score",
    "palm_y",
    "wrist_phase",
    "wrist_transition",
    "candidate_elbow",
    "candidate_wrist_rotation",
    "current_elbow",
    "current_wrist_rotation",
    "ack",
]

SUMMARY_FIELDNAMES = [
    "run_id",
    "run_label",
    "start_time_iso",
    "measurement_start_time_iso",
    "end_time_iso",
    "camera",
    "detector",
    "landmarks",
    "show",
    "dry_run_serial",
    "warmup_frames",
    "max_frames",
    "max_seconds",
    "total_frames",
    "measured_frames",
    "hand_detected_frames",
    "landmarks_frames",
    "pose_sent_frames",
    "measurement_duration_s",
    "fps_mean",
    "fps_median",
    "fps_std",
    "frame_time_mean_ms",
    "frame_time_median_ms",
    "frame_time_std_ms",
    "frame_time_min_ms",
    "frame_time_max_ms",
    "frame_time_p95_ms",
    "capture_time_mean_ms",
    "capture_time_p95_ms",
    "detector_time_mean_ms",
    "detector_time_p95_ms",
    "landmarks_time_mean_ms",
    "landmarks_time_p95_ms",
    "control_time_mean_ms",
    "control_time_p95_ms",
    "serial_time_mean_ms",
    "serial_time_p95_ms",
    "serial_time_sent_mean_ms",
    "display_time_mean_ms",
    "display_time_p95_ms",
    "inference_time_mean_ms",
    "inference_time_p95_ms",
    "pipeline_time_mean_ms",
    "pipeline_time_p95_ms",
    "frame_csv",
]


def _apply_fixed_joints_with_dynamic_wrist(pose, gripper_fixed: float):
    locked = dict(pose)
    for joint, fixed_value in cfg.FIXED_JOINTS.items():
        if joint == "wrist_rotation":
            continue
        locked[joint] = fixed_value
    locked["gripper"] = gripper_fixed
    return locked


def _clamp_wrist_rotation(angle: float) -> float:
    lo, hi = cfg.SERVO_LIMITS["wrist_rotation"]
    return cfg.clamp(angle, lo, hi)


def _clamp_gripper(angle: float) -> float:
    lo, hi = cfg.SERVO_LIMITS["gripper"]
    return cfg.clamp(angle, lo, hi)


def _active_zone(frame_h: int):
    top_px = frame_h * cfg.HAND_ACTIVE_TOP_RATIO
    bottom_px = frame_h * cfg.HAND_ACTIVE_BOTTOM_RATIO
    return top_px, bottom_px


def _compute_palm_y(landmarks, frame_w: int, frame_h: int) -> float:
    def pt(index: int):
        return (landmarks[index].x * frame_w, landmarks[index].y * frame_h)

    wrist = pt(0)
    index_mcp = pt(5)
    pinky_mcp = pt(17)
    return (wrist[1] + index_mcp[1] + pinky_mcp[1]) / 3.0


def _compute_pose_target(
    palm_y: float,
    frame_h: int,
    wrist_target: float,
    gripper_fixed: float,
):
    top_px, bottom_px = _active_zone(frame_h)
    return {
        "base": cfg.FIXED_JOINTS["base"],
        "shoulder": cfg.FIXED_JOINTS["shoulder"],
        "elbow": cfg.map_range(
            palm_y,
            top_px,
            bottom_px,
            cfg.ELBOW_UP_SERVO,
            cfg.ELBOW_DOWN_SERVO,
        ),
        "wrist_vertical": cfg.FIXED_JOINTS["wrist_vertical"],
        "wrist_rotation": wrist_target,
        "gripper": gripper_fixed,
    }


def _percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _mean(values) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values) -> float:
    return float(statistics.median(values)) if values else 0.0


def _std(values) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_frame_csv_path(run_label: str) -> Path:
    filename = f"{run_label}_{_timestamp_slug()}_frames.csv"
    return DEFAULT_RESULTS_DIR / filename


def _append_summary_row(path: Path, row: dict) -> None:
    _ensure_parent(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _print_summary(summary_row: dict) -> None:
    print("\nBenchmark summary")
    print(f"  Run label: {summary_row['run_label']}")
    print(f"  Run id: {summary_row['run_id']}")
    print(
        f"  Measured frames: {summary_row['measured_frames']} / total frames: "
        f"{summary_row['total_frames']}"
    )
    print(
        f"  Duration: {summary_row['measurement_duration_s']:.2f} s | "
        f"mean FPS: {summary_row['fps_mean']:.2f} | "
        f"median FPS: {summary_row['fps_median']:.2f}"
    )
    print(
        f"  Frame time mean/median/p95: {summary_row['frame_time_mean_ms']:.2f} / "
        f"{summary_row['frame_time_median_ms']:.2f} / "
        f"{summary_row['frame_time_p95_ms']:.2f} ms"
    )
    print(
        f"  Capture/detector/landmarks mean: {summary_row['capture_time_mean_ms']:.2f} / "
        f"{summary_row['detector_time_mean_ms']:.2f} / "
        f"{summary_row['landmarks_time_mean_ms']:.2f} ms"
    )
    print(
        f"  Control/serial/display mean: {summary_row['control_time_mean_ms']:.2f} / "
        f"{summary_row['serial_time_mean_ms']:.2f} / "
        f"{summary_row['display_time_mean_ms']:.2f} ms"
    )
    print(
        f"  Hand detected frames: {summary_row['hand_detected_frames']} | "
        f"landmark frames: {summary_row['landmarks_frames']} | "
        f"pose sends: {summary_row['pose_sent_frames']}"
    )
    print(f"  Frame CSV: {summary_row['frame_csv']}")


def main():
    default_wrist_home = cfg.NEUTRAL_POSE["wrist_rotation"]
    default_wrist_rotated = _clamp_wrist_rotation(default_wrist_home - 90.0)
    default_gripper_fixed = float(cfg.SERVO_LIMITS["gripper"][1])

    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument(
        "--run-label",
        default="mp_wrist_rotate_cpu",
        help="Short label stored in the CSV files for this run.",
    )
    ap.add_argument(
        "--frame-csv",
        default=None,
        help="Per-frame CSV output path. Defaults to a timestamped file under benchmark_results/.",
    )
    ap.add_argument(
        "--summary-csv",
        default=str(DEFAULT_SUMMARY_CSV),
        help="CSV file that receives one appended summary row per run.",
    )
    ap.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Initial frames to log but exclude from measured statistics.",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Number of measured frames to collect after warm-up. Default 0 means frame count will not stop the run.",
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=25.0,
        help="Maximum measured duration in seconds after warm-up. Default is 25 seconds; use 0 for no limit.",
    )
    ap.add_argument(
        "--dry-run-serial",
        action="store_true",
        help="Skip Arduino writes but keep the same pose generation logic for perception-only timing.",
    )
    ap.add_argument(
        "--wrist-home",
        type=float,
        default=default_wrist_home,
        help="Wrist angle held while the hand is moving down and waiting at the bottom.",
    )
    ap.add_argument(
        "--wrist-rotated",
        type=float,
        default=default_wrist_rotated,
        help="Wrist angle reached as the hand rises from bottom to top.",
    )
    ap.add_argument(
        "--gripper-fixed",
        type=float,
        default=default_gripper_fixed,
        help="Fixed M6 angle to hold during the wrist-roll demo.",
    )
    ap.add_argument(
        "--max-num-hands",
        type=int,
        default=1,
        help="Maximum number of hands for the MediaPipe detector.",
    )
    ap.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.70,
        help="MediaPipe minimum detection confidence.",
    )
    ap.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.70,
        help="MediaPipe minimum tracking confidence.",
    )
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    wrist_home = _clamp_wrist_rotation(args.wrist_home)
    wrist_rotated = _clamp_wrist_rotation(args.wrist_rotated)
    gripper_fixed = _clamp_gripper(args.gripper_fixed)

    frame_csv = Path(args.frame_csv) if args.frame_csv else _build_frame_csv_path(args.run_label)
    frame_csv = frame_csv.expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    _ensure_parent(frame_csv)

    run_id = f"{args.run_label}_{_timestamp_slug()}"
    run_start_epoch = time.time()
    run_start_iso = datetime.fromtimestamp(run_start_epoch).isoformat(timespec="seconds")
    measurement_start_iso = ""

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    ser = None
    if not args.dry_run_serial:
        import serial

        ser = serial.Serial(cfg.SERIAL_PORT, cfg.BAUD_RATE, timeout=0.10)
        time.sleep(1.5)
        ser.reset_input_buffer()

    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera}")

    current_pose = dict(cfg.NEUTRAL_POSE)
    follow_state = {joint: float(cfg.NEUTRAL_POSE[joint]) for joint in cfg.JOINTS}
    current_pose["wrist_rotation"] = int(round(wrist_home))
    current_pose["gripper"] = int(round(gripper_fixed))
    follow_state["wrist_rotation"] = float(wrist_home)
    follow_state["gripper"] = float(gripper_fixed)

    if ser is not None:
        initial_pose = cfg.sanitize_pose(dict(current_pose))
        cfg.send_pose(ser, initial_pose)

    last_send = time.time()
    last_heartbeat = last_send
    last_hand_seen = time.time()
    wrist_phase = "TOP_HOME"
    wrist_cycle_target = float(wrist_home)

    total_frames = 0
    measured_frames = 0
    hand_detected_frames = 0
    landmarks_frames = 0
    pose_sent_frames = 0

    measurement_start_monotonic = None

    capture_times = []
    detector_times = []
    landmarks_times = []
    control_times = []
    serial_times = []
    serial_times_sent = []
    display_times = []
    inference_times = []
    pipeline_times = []
    frame_times = []
    fps_values = []

    frame_file = frame_csv.open("w", newline="", encoding="utf-8")
    frame_writer = csv.DictWriter(frame_file, fieldnames=FRAME_FIELDNAMES)
    frame_writer.writeheader()

    print("Running MediaPipe benchmark wrist rotation demo. ESC to quit.")
    print(f"Per-frame CSV: {frame_csv}")
    print(f"Summary CSV:   {summary_csv}")

    try:
        while True:
            total_frames += 1
            frame_start_perf = time.perf_counter()

            capture_start_perf = time.perf_counter()
            ok, frame = cap.read()
            capture_time_ms = (time.perf_counter() - capture_start_perf) * 1000.0
            if not ok:
                break

            frame_h, frame_w = frame.shape[:2]
            palm_y = None
            hand_detected = False
            landmarks_found = False
            wrist_target = float(wrist_cycle_target)
            wrist_transition = ""
            chosen_score = 0.0
            ack = None

            mp_process_start_perf = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            mp_process_time_ms = (time.perf_counter() - mp_process_start_perf) * 1000.0

            detector_time_ms = 0.0
            landmarks_time_ms = mp_process_time_ms

            control_start_perf = time.perf_counter()
            selected_hand_lm = None
            selected_landmarks = None
            selected_label = None
            if results.multi_hand_landmarks:
                selected_hand_lm, selected_landmarks, selected_label = cfg.select_target_hand(
                    results, cfg.TARGET_HAND_LABEL
                )

            hands_found = len(results.multi_hand_landmarks or [])
            hand_detected = selected_landmarks is not None

            if hand_detected:
                last_hand_seen = time.time()
                landmarks_found = True

                if selected_label and results.multi_handedness:
                    for hand_info in results.multi_handedness:
                        if hand_info.classification[0].label == selected_label:
                            chosen_score = float(hand_info.classification[0].score)
                            break
                if chosen_score == 0.0 and results.multi_handedness:
                    chosen_score = float(results.multi_handedness[0].classification[0].score)

                palm_y = _compute_palm_y(
                    selected_landmarks,
                    frame_w,
                    frame_h,
                )
                top_px, bottom_px = _active_zone(frame_h)
                if palm_y >= bottom_px:
                    if wrist_phase != "BOTTOM_ROTATED":
                        wrist_transition = "BOTTOM_ROTATED"
                    wrist_phase = "BOTTOM_ROTATED"
                    wrist_cycle_target = float(wrist_rotated)
                elif palm_y <= top_px:
                    if wrist_phase != "TOP_HOME":
                        wrist_transition = "TOP_HOME"
                    wrist_phase = "TOP_HOME"
                    wrist_cycle_target = float(wrist_home)

                wrist_target = float(wrist_cycle_target)

                if wrist_transition == "BOTTOM_ROTATED":
                    follow_state["wrist_rotation"] = float(wrist_rotated)
                elif wrist_transition == "TOP_HOME":
                    follow_state["wrist_rotation"] = float(wrist_home)

                manual_target = _compute_pose_target(
                    palm_y,
                    frame_h,
                    wrist_target,
                    gripper_fixed,
                )

                cfg.smooth_pose(follow_state, manual_target)
                desired_pose = dict(follow_state)
            else:
                dt = time.time() - last_hand_seen
                if dt > cfg.HAND_LOST_TIMEOUT:
                    if cfg.FAILSAFE_MODE == "NEUTRAL":
                        desired_pose = dict(cfg.NEUTRAL_POSE)
                    else:
                        desired_pose = dict(current_pose)
                else:
                    desired_pose = dict(current_pose)

            candidate = cfg.sanitize_pose(
                _apply_fixed_joints_with_dynamic_wrist(desired_pose, gripper_fixed)
            )
            limited = cfg.sanitize_pose(cfg.rate_limit_pose(current_pose, candidate))

            if wrist_transition in {"BOTTOM_ROTATED", "TOP_HOME"}:
                limited["wrist_rotation"] = candidate["wrist_rotation"]

            now = time.time()
            should_send = False
            if now - last_send >= cfg.SEND_INTERVAL and cfg.pose_changed(limited, current_pose):
                should_send = True
            if now - last_heartbeat >= cfg.HEARTBEAT_INTERVAL:
                should_send = True
            if wrist_transition in {"BOTTOM_ROTATED", "TOP_HOME"}:
                should_send = True

            control_time_ms = (time.perf_counter() - control_start_perf) * 1000.0

            serial_time_ms = 0.0
            if should_send:
                serial_start_perf = time.perf_counter()
                if ser is not None:
                    ack = cfg.send_pose(ser, limited)
                serial_time_ms = (time.perf_counter() - serial_start_perf) * 1000.0
                current_pose = dict(limited)
                last_send = now
                last_heartbeat = now
                if ack:
                    print(
                        "[ACK]",
                        ack,
                        "| elbow=",
                        current_pose["elbow"],
                        "wrist_roll=",
                        current_pose["wrist_rotation"],
                    )

            display_start_perf = time.perf_counter()
            if args.show:
                if selected_hand_lm is not None:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        selected_hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                    )

                wrist_text = f"{candidate['wrist_rotation']}"
                palm_text = "---" if palm_y is None else f"{palm_y:.0f}"

                cv2.putText(
                    frame,
                    f"Hand: {'yes' if hand_detected else 'no'} | PalmY: {palm_text}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0) if hand_detected else (0, 180, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    (
                        f"Elbow {current_pose['elbow']} | Wrist roll {current_pose['wrist_rotation']} "
                        f"(target {wrist_text}) | M6 fixed {current_pose['gripper']}"
                    ),
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    (
                        f"Top={int(round(wrist_home))} | "
                        f"Bottom={int(round(wrist_rotated))} | "
                        f"Gripper={int(round(gripper_fixed))} | {wrist_phase}"
                    ),
                    (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    (
                        f"Frame {total_frames} | Capture {capture_time_ms:.2f} ms | "
                        f"MP {mp_process_time_ms:.2f} ms"
                    ),
                    (10, 112),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("MediaPipe wrist rotate benchmark -> Braccio", frame)
                stop_requested = (cv2.waitKey(1) & 0xFF) == 27
            else:
                stop_requested = False
            display_time_ms = (time.perf_counter() - display_start_perf) * 1000.0

            inference_time_ms = detector_time_ms + landmarks_time_ms
            pipeline_time_ms = (
                capture_time_ms
                + detector_time_ms
                + landmarks_time_ms
                + control_time_ms
                + serial_time_ms
            )
            total_frame_time_ms = (time.perf_counter() - frame_start_perf) * 1000.0
            fps = 1000.0 / max(total_frame_time_ms, 0.0001)

            measured = total_frames > args.warmup_frames
            if measured and measurement_start_monotonic is None:
                measurement_start_monotonic = time.monotonic()
                measurement_start_iso = datetime.now().isoformat(timespec="seconds")

            elapsed_measurement_s = (
                time.monotonic() - measurement_start_monotonic
                if measurement_start_monotonic is not None
                else 0.0
            )

            frame_writer.writerow(
                {
                    "run_id": run_id,
                    "run_label": args.run_label,
                    "frame_index": total_frames,
                    "measured": int(measured),
                    "elapsed_measurement_s": f"{elapsed_measurement_s:.6f}",
                    "capture_time_ms": f"{capture_time_ms:.6f}",
                    "detector_time_ms": f"{detector_time_ms:.6f}",
                    "landmarks_time_ms": f"{landmarks_time_ms:.6f}",
                    "control_time_ms": f"{control_time_ms:.6f}",
                    "serial_time_ms": f"{serial_time_ms:.6f}",
                    "display_time_ms": f"{display_time_ms:.6f}",
                    "inference_time_ms": f"{inference_time_ms:.6f}",
                    "pipeline_time_ms": f"{pipeline_time_ms:.6f}",
                    "total_frame_time_ms": f"{total_frame_time_ms:.6f}",
                    "fps": f"{fps:.6f}",
                    "hand_detected": int(hand_detected),
                    "landmarks_found": int(landmarks_found),
                    "hands_found": hands_found,
                    "pose_sent": int(should_send),
                    "chosen_score": f"{chosen_score:.6f}",
                    "palm_y": "" if palm_y is None else f"{palm_y:.6f}",
                    "wrist_phase": wrist_phase,
                    "wrist_transition": wrist_transition,
                    "candidate_elbow": candidate["elbow"],
                    "candidate_wrist_rotation": candidate["wrist_rotation"],
                    "current_elbow": current_pose["elbow"],
                    "current_wrist_rotation": current_pose["wrist_rotation"],
                    "ack": ack or "",
                }
            )

            if measured:
                measured_frames += 1
                if hand_detected:
                    hand_detected_frames += 1
                if landmarks_found:
                    landmarks_frames += 1

                capture_times.append(capture_time_ms)
                detector_times.append(detector_time_ms)
                landmarks_times.append(landmarks_time_ms)
                control_times.append(control_time_ms)
                serial_times.append(serial_time_ms)
                display_times.append(display_time_ms)
                inference_times.append(inference_time_ms)
                pipeline_times.append(pipeline_time_ms)
                frame_times.append(total_frame_time_ms)
                fps_values.append(fps)
                if should_send:
                    pose_sent_frames += 1
                    serial_times_sent.append(serial_time_ms)

            if total_frames % 30 == 0:
                frame_file.flush()

            if stop_requested:
                break
            if args.max_frames > 0 and measured_frames >= args.max_frames:
                break
            if (
                args.max_seconds > 0.0
                and measurement_start_monotonic is not None
                and elapsed_measurement_s >= args.max_seconds
            ):
                break

    finally:
        frame_file.close()
        cap.release()
        hands.close()
        if ser is not None:
            ser.close()
        if args.show:
            cv2.destroyAllWindows()

    end_time_iso = datetime.now().isoformat(timespec="seconds")
    measurement_duration_s = (
        time.monotonic() - measurement_start_monotonic
        if measurement_start_monotonic is not None
        else 0.0
    )

    summary_row = {
        "run_id": run_id,
        "run_label": args.run_label,
        "start_time_iso": run_start_iso,
        "measurement_start_time_iso": measurement_start_iso,
        "end_time_iso": end_time_iso,
        "camera": args.camera,
        "detector": DEFAULT_MEDIAPIPE_LABEL,
        "landmarks": DEFAULT_MEDIAPIPE_LABEL,
        "show": int(args.show),
        "dry_run_serial": int(args.dry_run_serial),
        "warmup_frames": args.warmup_frames,
        "max_frames": args.max_frames,
        "max_seconds": args.max_seconds,
        "total_frames": total_frames,
        "measured_frames": measured_frames,
        "hand_detected_frames": hand_detected_frames,
        "landmarks_frames": landmarks_frames,
        "pose_sent_frames": pose_sent_frames,
        "measurement_duration_s": round(measurement_duration_s, 6),
        "fps_mean": round(_mean(fps_values), 6),
        "fps_median": round(_median(fps_values), 6),
        "fps_std": round(_std(fps_values), 6),
        "frame_time_mean_ms": round(_mean(frame_times), 6),
        "frame_time_median_ms": round(_median(frame_times), 6),
        "frame_time_std_ms": round(_std(frame_times), 6),
        "frame_time_min_ms": round(min(frame_times), 6) if frame_times else 0.0,
        "frame_time_max_ms": round(max(frame_times), 6) if frame_times else 0.0,
        "frame_time_p95_ms": round(_percentile(frame_times, 95.0), 6),
        "capture_time_mean_ms": round(_mean(capture_times), 6),
        "capture_time_p95_ms": round(_percentile(capture_times, 95.0), 6),
        "detector_time_mean_ms": round(_mean(detector_times), 6),
        "detector_time_p95_ms": round(_percentile(detector_times, 95.0), 6),
        "landmarks_time_mean_ms": round(_mean(landmarks_times), 6),
        "landmarks_time_p95_ms": round(_percentile(landmarks_times, 95.0), 6),
        "control_time_mean_ms": round(_mean(control_times), 6),
        "control_time_p95_ms": round(_percentile(control_times, 95.0), 6),
        "serial_time_mean_ms": round(_mean(serial_times), 6),
        "serial_time_p95_ms": round(_percentile(serial_times, 95.0), 6),
        "serial_time_sent_mean_ms": round(_mean(serial_times_sent), 6),
        "display_time_mean_ms": round(_mean(display_times), 6),
        "display_time_p95_ms": round(_percentile(display_times, 95.0), 6),
        "inference_time_mean_ms": round(_mean(inference_times), 6),
        "inference_time_p95_ms": round(_percentile(inference_times, 95.0), 6),
        "pipeline_time_mean_ms": round(_mean(pipeline_times), 6),
        "pipeline_time_p95_ms": round(_percentile(pipeline_times, 95.0), 6),
        "frame_csv": str(frame_csv),
    }

    _append_summary_row(summary_csv, summary_row)
    _print_summary(summary_row)


if __name__ == "__main__":
    main()
