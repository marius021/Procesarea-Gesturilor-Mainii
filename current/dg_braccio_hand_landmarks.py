import argparse
import sys
import time
from types import SimpleNamespace
from pathlib import Path
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


def _as_pixels_bbox(bbox, w, h):
    """
    DeGirum bbox is usually [xtop, ytop, xbot, ybot].
    It can be normalized (0..1) or pixels; handle both.
    """
    x0, y0, x1, y1 = bbox
    if x1 <= 1.5 and y1 <= 1.5:  # normalized
        x0, x1 = int(x0 * w), int(x1 * w)
        y0, y1 = int(y0 * h), int(y1 * h)
    else:  # pixels
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if x1 <= x0: x1 = min(w - 1, x0 + 1)
    if y1 <= y0: y1 = min(h - 1, y0 + 1)
    return x0, y0, x1, y1


def _expand_bbox(x0, y0, x1, y1, w, h, extent_pct=30.0):
    """
    Expand bbox by percent of bbox size (like DeGirum examples often do for crops).
    """
    bw = x1 - x0
    bh = y1 - y0
    ex = int(bw * extent_pct / 100.0)
    ey = int(bh * extent_pct / 100.0)

    nx0 = max(0, x0 - ex)
    ny0 = max(0, y0 - ey)
    nx1 = min(w - 1, x1 + ex)
    ny1 = min(h - 1, y1 + ey)
    return nx0, ny0, nx1, ny1


def _landmarks_to_fullframe_norm(landmarks_list, crop_w, crop_h, x0, y0, bw, bh, frame_w, frame_h):
    """
    Convert DeGirum hand landmarks to "MediaPipe-like" normalized landmarks
    in full-frame coordinates so we can reuse cfg.compute_hand_features().
    DeGirum landmarks can be normalized or pixel; handle both.
    """
    norm_landmarks = []
    for lm in landmarks_list:
        x, y, *_ = lm["landmark"]

        # normalize relative to crop model input
        if x <= 1.5 and y <= 1.5:
            xn = float(x)
            yn = float(y)
        else:
            xn = float(x) / float(crop_w)
            yn = float(y) / float(crop_h)

        # map to full-frame pixels via bbox
        x_full = x0 + xn * bw
        y_full = y0 + yn * bh

        # convert to normalized full-frame for cfg.compute_hand_features()
        norm_landmarks.append(SimpleNamespace(
            x=float(x_full) / float(frame_w),
            y=float(y_full) / float(frame_h),
        ))

    return norm_landmarks


def main():
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
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    dg = _import_degirum()
    zoo_url = _normalize_zoo_url(args.zoo)
    zoo = dg.connect(inference_host_address="@local", zoo_url=zoo_url)

    # Load detector
    det = zoo.load_model(
        model_name=args.detector,
        output_confidence_threshold=args.conf,
        overlay_show_labels=False,
        overlay_show_probabilities=False,
    )
    det.input_numpy_colorspace = "BGR"

    # Load landmarks
    lmk = zoo.load_model(
        model_name=args.landmarks,
        overlay_show_labels=False,
        overlay_show_probabilities=False,
    )
    lmk.input_numpy_colorspace = "BGR"

    # Serial (reuse your packet format + ACK behavior)
    import serial
    ser = serial.Serial(cfg.SERIAL_PORT, cfg.BAUD_RATE, timeout=0.10)
    time.sleep(1.5)
    ser.reset_input_buffer()

    cap = _open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera}")

    # State (reuse your smoothing/rate limit logic)
    current_pose = dict(cfg.NEUTRAL_POSE)
    follow_state = {j: float(cfg.NEUTRAL_POSE[j]) for j in cfg.JOINTS}

    last_send = 0.0
    last_heartbeat = 0.0
    last_hand_seen = time.time()

    print("Running. ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]

            # ---- 1) Detect hands (bbox) ----
            det_res = det(frame)
            hands = []
            for r in (det_res.results or []):
                # Expect detector results with bbox/score/label per DeGirum result structure :contentReference[oaicite:2]{index=2}
                if "bbox" in r and "score" in r:
                    hands.append(r)

            # choose best hand bbox (highest score)
            chosen = None
            if hands:
                chosen = max(hands, key=lambda x: x.get("score", 0.0))

            hand_detected = chosen is not None

            if hand_detected:
                last_hand_seen = time.time()

                x0, y0, x1, y1 = _as_pixels_bbox(chosen["bbox"], w, h)
                x0, y0, x1, y1 = _expand_bbox(x0, y0, x1, y1, w, h, extent_pct=args.crop_extent)

                crop = frame[y0:y1, x0:x1].copy()
                bw = (x1 - x0)
                bh = (y1 - y0)

                # ---- 2) Landmarks on cropped hand ----
                lmk_res = lmk(crop)

                # For hand palm/landmark postprocessors, results contain `landmarks` list :contentReference[oaicite:3]{index=3}
                if lmk_res.results and "landmarks" in lmk_res.results[0]:
                    crop_landmarks = lmk_res.results[0]["landmarks"]
                    # Convert to full-frame normalized landmarks for your existing compute_hand_features()
                    norm_landmarks = _landmarks_to_fullframe_norm(
                        crop_landmarks,
                        crop_w=crop.shape[1],
                        crop_h=crop.shape[0],
                        x0=x0, y0=y0, bw=bw, bh=bh,
                        frame_w=w, frame_h=h,
                    )

                    palm_y, finger_angle = cfg.compute_hand_features(norm_landmarks, w, h)
                    manual_target = cfg.compute_manual_target(palm_y, finger_angle, h)

                    cfg.smooth_pose(follow_state, manual_target)
                    desired_pose = dict(follow_state)
                else:
                    # Landmark model failed this frame
                    desired_pose = dict(current_pose)

                # optional debug draw
                if args.show:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            else:
                # ---- failsafe behavior (same as your script) ----
                dt = time.time() - last_hand_seen
                if dt > cfg.HAND_LOST_TIMEOUT:
                    if cfg.FAILSAFE_MODE == "NEUTRAL":
                        desired_pose = dict(cfg.NEUTRAL_POSE)
                    else:
                        desired_pose = dict(current_pose)
                else:
                    desired_pose = dict(current_pose)

            # Lock fixed joints + clamp + rate-limit
            candidate = cfg.sanitize_pose(cfg.apply_fixed_joints(desired_pose))
            limited = cfg.sanitize_pose(cfg.rate_limit_pose(current_pose, candidate))

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
                    print("[ACK]", ack, "| elbow=", current_pose["elbow"], "grip=", current_pose["gripper"])

            if args.show:
                cv2.imshow("DeGirum hand->landmarks->Braccio", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    finally:
        cap.release()
        ser.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
