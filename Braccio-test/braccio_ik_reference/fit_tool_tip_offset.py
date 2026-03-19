#!/usr/bin/env python3
"""
Fit tool-tip offset and wrist-vertical zero trim from same-point samples.
"""

from __future__ import annotations

import json
from pathlib import Path

from braccio_kinematics import fit_tool_tip_offset, load_calibration, load_json


SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = SCRIPT_DIR / "braccio_calibration.json"
TEMPLATE_PATH = SCRIPT_DIR / "braccio_calibration.template.json"
SAMPLES_PATH = SCRIPT_DIR / "same_point_samples.json"


def main():
    calibration_file = CALIBRATION_PATH if CALIBRATION_PATH.exists() else TEMPLATE_PATH
    if not SAMPLES_PATH.exists():
        raise SystemExit(
            f"Create {SAMPLES_PATH} from same_point_samples.template.json before running this fitter."
        )

    calibration = load_calibration(calibration_file)
    samples = load_json(SAMPLES_PATH)
    result = fit_tool_tip_offset(samples, calibration)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
