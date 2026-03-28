import argparse
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2


DEFAULT_ZOO_DIR = Path(__file__).resolve().parents[1] / "zoo"


def _import_degirum():
    try:
        import degirum as dg
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python package 'degirum'. Install it in the interpreter you are using, "
            "then rerun this script."
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zoo",
        default=str(DEFAULT_ZOO_DIR),
        help="Local DeGirum zoo folder or file:// URL. Defaults to the local Degirum/zoo folder.",
    )
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()

    dg = _import_degirum()

    zoo_url = _normalize_zoo_url(args.zoo)
    print("Using zoo:", zoo_url)

    zoo = dg.connect(inference_host_address="@local", zoo_url=zoo_url)

    # List models that can run on Hailo8
    models = zoo.list_models(device_type=["HAILORT/HAILO8"])
    if not models:
        raise RuntimeError("No HAILORT/HAILO8 models found in this zoo folder.")
    print("\nModels found:")
    for i, m in enumerate(models):
        print(f"  [{i}] {m}")

    cap = _open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {args.camera}")

    idx = 0
    model = None
    window = "DeGirum Hailo Hand Detector Test"

    def load_model(i: int):
        name = models[i]
        print(f"\n[LOAD] {name}")
        mdl = zoo.load_model(
            model_name=name,
            output_confidence_threshold=args.conf,
            overlay_show_labels=True,
            overlay_show_probabilities=False,
        )
        # OpenCV frames are BGR; PySDK defaults match this for OpenCV backend. :contentReference[oaicite:3]{index=3}
        mdl.input_numpy_colorspace = "BGR"
        return mdl

    model = load_model(idx)

    last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame)  # Model.predict() via __call__ :contentReference[oaicite:4]{index=4}
        overlay = result.image_overlay  # overlay image :contentReference[oaicite:5]{index=5}

        # Count detections (result.results is a list of dicts; detection dicts typically include bbox/label/score) :contentReference[oaicite:6]{index=6}
        det_count = len(getattr(result, "results", []) or [])
        cv2.putText(overlay, f"{models[idx]} | det={det_count} | fps={fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window, overlay)

        # FPS estimate
        now = time.time()
        dt = now - last
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last = now

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('n'), ord('N')):  # next model
            idx = (idx + 1) % len(models)
            model = load_model(idx)
        elif key in (ord('p'), ord('P')):  # previous model
            idx = (idx - 1) % len(models)
            model = load_model(idx)
        elif key == ord('+'):
            args.conf = min(0.9, args.conf + 0.05)
            print("conf =", args.conf)
            model.output_confidence_threshold = args.conf
        elif key == ord('-'):
            args.conf = max(0.05, args.conf - 0.05)
            print("conf =", args.conf)
            model.output_confidence_threshold = args.conf

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
