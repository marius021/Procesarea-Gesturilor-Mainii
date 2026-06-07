# current/ — active CPU + Hailo implementation

This folder holds the implementations actually used by the project (both the
CPU/MediaPipe and the Hailo/DeGirum paths), plus the local DeGirum model zoo.

Layout:

- `*.py` — the active scripts (run them directly from this folder)
- `zoo/` — the local DeGirum model zoo
- `venv_hailo_rpi_examples/` — the Hailo virtual environment, kept **in this
  same folder** as the scripts (not tracked in Git; recreate it on the Pi)

Benchmark outputs (CSVs, plots, comparison report) are written to the
top-level `../results/` folder. Old/experimental implementations live in
`../old/`, and the Arduino firmware sketches are in `../Arduino/`.

## Create and activate the environment

The venv lives next to the scripts, so create it here:

```bash
cd current
python3 -m venv venv_hailo_rpi_examples
source venv_hailo_rpi_examples/bin/activate
pip install degirum opencv-python pyserial mediapipe matplotlib
```

## Run the webcam model browser

```bash
python3 dg_webcam_test.py --camera /dev/video0 --conf 0.4
```

The script defaults to the local zoo at `current/zoo`.

## Run the Braccio hand-control script (Hailo)

```bash
python3 dg_braccio_hand_landmarks.py --camera /dev/video0 --show
```

This also defaults to `current/zoo`.

## Benchmarks

```bash
# Hailo / DeGirum benchmark
python3 dg_braccio_wrist_rotate_benchmark.py --camera /dev/video0

# CPU / MediaPipe benchmark
python3 mp_braccio_wrist_rotate_benchmark.py --camera /dev/video0

# Compare + plot (outputs under ../results/)
python3 compare_benchmark_results.py
python3 plot_benchmark_trajectories.py
```

All benchmark CSVs and plots are written under `../results/`.
