# Procesarea Gesturilor Mâinii — Hand-Gesture Control for an Arduino Braccio Arm

Control an **Arduino Braccio** robotic arm in real time using **hand gestures**
captured from a webcam. A camera frame is turned into hand landmarks, the
landmarks are mapped to joint angles, the angles are filtered for safety, and
the resulting pose is streamed to the arm over a signed serial link.

```
camera → perception → hand features → motion mapping → safety filters → serial command → robot
```

The project supports **two perception backends** and includes tooling to
benchmark them against each other:

- **CPU / MediaPipe** — runs anywhere with a webcam, no accelerator needed.
- **Hailo / DeGirum** — hardware-accelerated inference on a Raspberry Pi 5 with
  a Hailo-8 module, using a local DeGirum model zoo (YOLOv8n hand detector +
  hand-landmark model).

## What it does

- Detects a single hand and extracts geometric features (palm height, finger
  opening, wrist orientation).
- Maps those features to Braccio joint targets — e.g. hand height drives the
  elbow, finger pinch drives the gripper, and a vertical sweep drives a
  wrist-rotation demo.
- Smooths and rate-limits motion, clamps every joint to safe servo limits, and
  falls back to a neutral pose when the hand is lost.
- Signs every serial command with a truncated **HMAC-SHA256** tag; the Arduino
  firmware rejects unsigned or tampered commands and returns to neutral if
  commands stop arriving (watchdog).

## Repository layout

| Path         | Contents |
|--------------|----------|
| `current/`   | Active implementation — CPU/MediaPipe and Hailo/DeGirum scripts, benchmarks, plotting tools, and the local DeGirum model `zoo/`. |
| `Arduino/`   | Braccio firmware sketches. `Script-Arduino.ino` is the HMAC-verifying bridge; the `*_safe.ino` variants are simpler reference bridges. |
| `results/`   | Benchmark CSVs, generated plots, and the `benchmark_comparison.md` report. |
| `Schemas/`   | System / data-flow diagrams (`draw.io`). |
| `old/`       | Legacy and exploratory implementations kept for history. |
| `Progress 25.04.md` | Detailed narrative of how the project evolved. |

The most important scripts live in `current/`:

- `braccio_suture_demo_test.py` — shared config + the CPU/MediaPipe demo
  (servo limits, smoothing, HMAC signing, serial protocol live here).
- `dg_braccio_wrist_rotate.py` / `dg_braccio_hand_landmarks.py` — Hailo/DeGirum
  live control.
- `*_benchmark.py`, `compare_benchmark_results.py`, `plot_benchmark_trajectories.py`
  — measurement and reporting.
- `test_hmac.py` — standalone check that the firmware accepts signed and rejects
  tampered commands.

## Hardware

- Arduino Braccio arm (6 servos) on an Arduino board with the Braccio shield.
- Webcam (USB or Pi camera).
- For the accelerated path: Raspberry Pi 5 + Hailo-8.
- Serial link at **115200 baud** (default port `/dev/ttyACM0`).

## Getting started (broad strokes)

1. **Flash the firmware.** Open a sketch from `Arduino/` in the Arduino IDE and
   upload it to the Braccio. `Script-Arduino.ino` requires the
   [rweather/Crypto](https://rweather.github.io/arduinolibs/) library for
   SHA-256; flash a `*_safe.ino` variant first if you just want to verify wiring
   without HMAC.

2. **Set up Python.** From `current/`, create a virtual environment and install
   the dependencies the scripts use — `degirum`, `opencv-python`, `pyserial`,
   `mediapipe`, `matplotlib`, `psutil`. (On the Pi, the venv is recreated locally
   and is not tracked in Git.)

3. **Match your setup.** Edit the config near the top of
   `braccio_suture_demo_test.py` if your serial port differs from
   `/dev/ttyACM0`, and keep the `HMAC_SECRET_KEY` in sync with `SECRET_KEY` in
   the firmware.

4. **Run a demo.** Launch one of the `current/` scripts pointed at your camera —
   the CPU demo (`braccio_suture_demo_test.py`) or the Hailo wrist-rotate /
   landmark demos. The DeGirum scripts default to the bundled `current/zoo`.

5. **Benchmark (optional).** Run the `*_benchmark.py` scripts to log per-frame
   timing to `results/`, then `compare_benchmark_results.py` and
   `plot_benchmark_trajectories.py` to generate the comparison report and plots.

Each script takes `--help`; see [current/README.md](current/README.md) for the
exact commands and flags.

## Benchmark snapshot

From `results/benchmark_comparison.md` (wrist-rotate demo):

| Backend        | Avg FPS | Avg frame latency | Avg inference |
|----------------|---------|-------------------|---------------|
| Hailo/DeGirum  | ~28     | ~38 ms            | ~29 ms        |
| CPU/MediaPipe  | ~13     | ~220 ms           | ~44 ms        |

The Hailo path delivers markedly lower and more consistent frame latency; the
CPU path runs without dedicated hardware but is bottlenecked under load.

## Safety notes

Every command is HMAC-signed and the firmware clamps all joint angles to
per-servo limits, so a malformed or unauthenticated command can never drive a
joint out of range. If the host stops sending commands, the arm returns to its
neutral pose automatically.
