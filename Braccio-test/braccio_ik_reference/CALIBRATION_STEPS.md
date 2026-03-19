# Calibration Steps

1. Copy `braccio_calibration.template.json` to `braccio_calibration.json`.
2. Measure and update:
   - `geometry.base_height_mm`: table to shoulder joint center
   - `geometry.upper_arm_mm`: shoulder joint center to elbow joint center
   - `geometry.forearm_mm`: elbow joint center to wrist-vertical joint center
   - `geometry.tool_tip_offset_mm`: wrist-vertical joint center to the actual needle tip
3. Set the arm to a visual neutral pose and adjust:
   - `servo.base.zero_deg`
   - `servo.shoulder.zero_deg`
   - `servo.elbow.zero_deg`
   - `servo.wrist_vertical.zero_deg`
   - `servo.wrist_rotation.zero_deg`
4. Set the operation plane:
   - `operation_zone.surface_z_mm`
   - `operation_zone.x_min_mm`
   - `operation_zone.x_max_mm`
   - `operation_zone.y_min_mm`
   - `operation_zone.y_max_mm`
5. For tool-tip fitting:
   - touch the same physical point with the needle tip from 3 or more different joint poses
   - save those poses in `same_point_samples.json`
   - run `fit_tool_tip_offset.py`
6. After the geometry is stable, run `braccio_cartesian_controller.py`.

The first pass should be done with the needle removed or replaced by a blunt training tip.
