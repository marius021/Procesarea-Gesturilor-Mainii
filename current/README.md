# DeGirum Folder

This folder contains only the DeGirum assets used by this project on Raspberry Pi:

- `scripts/`
- `zoo/`

The Python virtual environment is intentionally not tracked in Git. Recreate it on the target machine and install the required packages there.

## Create and activate an environment

```bash
python3 -m venv Degirum/venv_hailo_rpi_examples
source Degirum/venv_hailo_rpi_examples/bin/activate
pip install degirum opencv-python pyserial
```

## Run the webcam model browser

```bash
python3 Degirum/scripts/dg_webcam_test.py --camera /dev/video0 --conf 0.4
```

The script defaults to the local zoo at `Degirum/zoo`.

## Run the Braccio hand-control script

```bash
python3 Degirum/scripts/dg_braccio_hand_landmarks.py --camera /dev/video0 --show
```

This also defaults to `Degirum/zoo`.
