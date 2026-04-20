import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ZOO_DIR = SCRIPT_DIR.parents[1] / "Degirum" / "zoo"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import braccio_suture_demo_test as cfg


def _import_degirum():
    try:
        import degirum as dg
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python package 'degirum'. Activate Degirum/venv_hailo_rpi_examples first, "
            "or install degirum in the interpreter you are using."
        ) from exc
    return dg


def _normalize_zoo_url(zoo_arg: str) -> str:
    if zoo_arg.startswith("file://"):
        parsed = urlparse(zoo_arg)
        zoo_path = Path(unquote(parsed.path)).expanduser().resolve()
    else:
        zoo_path = Path(zoo_arg).expanduser().resolve()
    return f"file://{zoo_path}"


def _open_camera(camera_arg: str):
    camera_source = int(camera_arg) if camera_arg.isdigit() else camera_arg
    return cv2.VideoCapture(camera_source)


def _as_pixels_bbox(bbox, frame_w, frame_h):
    """
    DeGirum bbox is usually [xtop, ytop, xbot, ybot].
    It can be normalized (0..1) or pixels; handle both.
    """
    x0, y0, x1, y1 = bbox
    if x1 <= 1.5 and y1 <= 1.5:
        x0, x1 = int(x0 * frame_w), int(x1 * frame_w)
        y0, y1 = int(y0 * frame_h), int(y1 * frame_h)
    else:
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    x0 = max(0, min(frame_w - 1, x0))
    x1 = max(0, min(frame_w - 1, x1))
    y0 = max(0, min(frame_h - 1, y0))
    y1 = max(0, min(frame_h - 1, y1))
    if x1 <= x0:
        x1 = min(frame_w - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(frame_h - 1, y0 + 1)
    return x0, y0, x1, y1


def _expand_bbox(x0, y0, x1, y1, frame_w, frame_h, extent_pct=30.0):
    """
    Expand bbox by a percent of the bbox size to give landmarks more context.
    """
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    expand_x = int(bbox_w * extent_pct / 100.0)
    expand_y = int(bbox_h * extent_pct / 100.0)

    nx0 = max(0, x0 - expand_x)
    ny0 = max(0, y0 - expand_y)
    nx1 = min(frame_w - 1, x1 + expand_x)
    ny1 = min(frame_h - 1, y1 + expand_y)
    return nx0, ny0, nx1, ny1


def _landmarks_to_fullframe_norm(
    landmarks_list,
    crop_w,
    crop_h,
    x0,
    y0,
    bbox_w,
    bbox_h,
    frame_w,
    frame_h,
):
    """
    Convert DeGirum hand landmarks to full-frame normalized coordinates so we
    can reuse cfg.compute_hand_features().
    """
    norm_landmarks = []
    for lm in landmarks_list:
        x, y, *_ = lm["landmark"]

        if x <= 1.5 and y <= 1.5:
            xn = float(x)
            yn = float(y)
        else:
            xn = float(x) / float(crop_w)
            yn = float(y) / float(crop_h)

        x_full = x0 + xn * bbox_w
        y_full = y0 + yn * bbox_h

        norm_landmarks.append(
            SimpleNamespace(
                x=float(x_full) / float(frame_w),
                y=float(y_full) / float(frame_h),
            )
        )

    return norm_landmarks


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


def main():
    default_wrist_home = cfg.NEUTRAL_POSE["wrist_rotation"]
    default_wrist_rotated = _clamp_wrist_rotation(default_wrist_home + 90.0)
    default_gripper_fixed = float(cfg.SERVO_LIMITS["gripper"][1])

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zoo",
        default=str(DEFAULT_ZOO_DIR),
        help="Local DeGirum zoo folder or file:// URL. Defaults to Degirum/zoo in this repo.",
    )
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--detector", default="yolov8n_relu6_hand--640x640_quant_hailort_hailo8_1")
    ap.add_argument("--landmarks", default="hand_landmark_lite--224x224_quant_hailort_hailo8_1")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--crop-extent", type=float, default=30.0)
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
        help="Fixed M6 angle to hold during the wrist-roll demo. On this setup, higher values close more.",
    )
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    wrist_home = _clamp_wrist_rotation(args.wrist_home)
    wrist_rotated = _clamp_wrist_rotation(args.wrist_rotated)
    gripper_fixed = _clamp_gripper(args.gripper_fixed)

    dg = _import_degirum()
    zoo_url = _normalize_zoo_url(args.zoo)
    zoo = dg.connect(inference_host_address="@local", zoo_url=zoo_url)

    det = zoo.load_model(
        model_name=args.detector,
        output_confidence_threshold=args.conf,
        overlay_show_labels=False,
        overlay_show_probabilities=False,
    )
    det.input_numpy_colorspace = "BGR"

    lmk = zoo.load_model(
        model_name=args.landmarks,
        overlay_show_labels=False,
        overlay_show_probabilities=False,
    )
    lmk.input_numpy_colorspace = "BGR"

    import serial

    ser = serial.Serial(cfg.SERIAL_PORT, cfg.BAUD_RATE, timeout=0.10)
    time.sleep(1.5)
    ser.reset_input_buffer()

    cap = _open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera}")

    current_pose = dict(cfg.NEUTRAL_POSE)
    follow_state = {joint: float(cfg.NEUTRAL_POSE[joint]) for joint in cfg.JOINTS}
    current_pose["wrist_rotation"] = int(round(wrist_home))
    current_pose["gripper"] = int(round(gripper_fixed))
    follow_state["wrist_rotation"] = float(wrist_home)
    follow_state["gripper"] = float(gripper_fixed)

    initial_pose = cfg.sanitize_pose(dict(current_pose))
    cfg.send_pose(ser, initial_pose)

    last_send = time.time()
    last_heartbeat = last_send
    last_hand_seen = time.time()
    wrist_phase = "TOP_HOME"
    wrist_cycle_target = float(wrist_home)
    wrist_transition = None

    print("Running wrist rotation demo. ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_h, frame_w = frame.shape[:2]
            palm_y = None
            hand_detected = False
            wrist_target = float(wrist_cycle_target)
            wrist_transition = None

            det_res = det(frame)
            hands = []
            for result in det_res.results or []:
                if "bbox" in result and "score" in result:
                    hands.append(result)

            chosen = max(hands, key=lambda result: result.get("score", 0.0)) if hands else None
            hand_detected = chosen is not None

            if hand_detected:
                last_hand_seen = time.time()

                x0, y0, x1, y1 = _as_pixels_bbox(chosen["bbox"], frame_w, frame_h)
                x0, y0, x1, y1 = _expand_bbox(
                    x0,
                    y0,
                    x1,
                    y1,
                    frame_w,
                    frame_h,
                    extent_pct=args.crop_extent,
                )

                crop = frame[y0:y1, x0:x1].copy()
                bbox_w = x1 - x0
                bbox_h = y1 - y0
                lmk_res = lmk(crop)

                if lmk_res.results and "landmarks" in lmk_res.results[0]:
                    crop_landmarks = lmk_res.results[0]["landmarks"]
                    norm_landmarks = _landmarks_to_fullframe_norm(
                        crop_landmarks,
                        crop_w=crop.shape[1],
                        crop_h=crop.shape[0],
                        x0=x0,
                        y0=y0,
                        bbox_w=bbox_w,
                        bbox_h=bbox_h,
                        frame_w=frame_w,
                        frame_h=frame_h,
                    )

                    palm_y = _compute_palm_y(
                        norm_landmarks,
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

                    # Snap M5 immediately at the stitch endpoints so the wrist
                    # clearly reaches the intended rotation instead of lagging.
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
                    desired_pose = dict(current_pose)

                if args.show:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
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

            if should_send:
                ack = cfg.send_pose(ser, limited)
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

            if args.show:
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
                cv2.imshow("DeGirum hand wrist rotate -> Braccio", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    finally:
        cap.release()
        ser.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
