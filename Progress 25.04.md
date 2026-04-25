# Progress 25.04

Detailed project progress report for the hand-gesture-to-Braccio control project.

This document is based on the current repository state on 25.04.2026, the tracked commit history, the benchmark outputs already stored in the repo, and the project notes that are currently available. It aims to explain not only what files exist, but also what they mean in the evolution of the project and what each stage accomplished.

## 1. Project Goal

The main goal of this project is to control an Arduino Braccio robotic arm using human hand gestures captured by a webcam. In practice, the project evolved into several related sub-goals:

- detect a hand reliably in a camera stream;
- extract useful geometric information from the hand;
- convert that information into robot commands;
- send those commands safely to the Braccio arm through Arduino;
- stabilize robot motion with smoothing, limits, and failsafe behavior;
- experiment with higher-level behaviors such as suturing-like motion patterns;
- migrate part of the perception pipeline from CPU/MediaPipe to DeGirum/Hailo acceleration on Raspberry Pi;
- measure performance and compare Hailo versus CPU execution using repeatable benchmark scripts.

So the project is not just "gesture detection." It is an end-to-end mechatronic pipeline:

`camera -> perception -> hand features -> motion mapping -> safety filters -> serial/TCP command -> robot behavior`

## 2. Big Picture: What Has Been Built

By this point, the repository shows that the project has already passed through several meaningful stages:

1. early arm-control tests and servo-lock/fix iterations;
2. a first simplified gesture-control concept written as pseudocode;
3. a MediaPipe-based live control demo for the Braccio arm;
4. safety-oriented Arduino bridge code for receiving servo commands;
5. exploratory scripts for a suturing-like motion pattern;
6. calibration and inverse-kinematics work for Cartesian control;
7. a first Hailo-based prototype for gesture UI experiments;
8. DeGirum/Hailo integration with local model zoo assets;
9. a dedicated wrist-rotation demo implemented on the Hailo/DeGirum stack;
10. benchmark scripts that log per-frame timings and per-run summaries;
11. a comparison report showing current Hailo versus CPU performance.

That means the work done so far spans computer vision, embedded communication, robot safety, motion mapping, calibration, and performance evaluation.

## 3. Repository Areas and Their Meaning

The repository currently contains several important work areas:

### `Braccio-test/`

This is the main experimental area for the Braccio robot. It includes:

- MediaPipe-based robot control scripts;
- Arduino sketches used to drive the Braccio safely;
- reference inverse-kinematics and calibration scripts;
- older test utilities and manual control helpers.

### `demo hailo part1/`

This folder contains an earlier Hailo-focused exploration. It is not the current finalized Hailo pipeline, but it is important historically because it shows the first attempt to combine:

- webcam capture;
- a hand-related Hailo inference path;
- MediaPipe landmarks;
- TCP command sending;
- session logging.

In other words, this folder documents the first Hailo experimentation phase before the cleaner DeGirum-based version.

### `demo-hailo-part2/Degirum/`

This is the newer and much more structured Hailo/DeGirum area. It contains:

- the local model zoo;
- the DeGirum webcam test;
- the DeGirum Braccio hand landmark script;
- the wrist-rotation control demo;
- the benchmark versions of the wrist-rotation demo;
- stored benchmark CSV files and a markdown comparison report.

This is currently the most mature part of the accelerated perception pipeline.

### `Test Scripts/`

This folder contains additional experiments and development scripts, including the suturing demo and other validation utilities. It reflects exploratory development and testing outside the more polished demo folders.

## 4. Progress Timeline So Far

Below is the best reconstruction of the project timeline from the commit history and files currently present in the repository.

### Phase A: Initial robot bring-up and stabilization

The earliest visible commit messages show work such as:

- `To be tested ( Servo Lock )` on 2026-03-12;
- multiple `arm_fix` and `Demo_fix` commits on 2026-03-19;
- `to be tested - new demo implementation` on 2026-03-23.

From these commit names, the first phase appears to have focused on basic robot usability:

- locking joints when needed;
- correcting servo behavior;
- repairing demo logic;
- making the arm controllable enough for further experimentation.

This is a normal and necessary phase in robotics projects. Before advanced gesture mapping can work, the low-level arm behavior has to be stable enough to trust.

### Phase B: Formalizing the idea in pseudocode

On 2026-03-24, the commit `Pseudo-code added` introduced the first structured description of the system. The file `PROJECT_PSEUDOCODE.md` explains the core idea:

- use a camera;
- detect a hand;
- extract simple features;
- map hand height to elbow motion;
- map finger opening to gripper motion;
- keep the remaining joints fixed;
- add smoothing, safety, and a neutral fallback.

This was a very important step because it turned a rough idea into an explicit control design. It defined the initial interaction model clearly enough to implement and test.

### Phase C: MediaPipe-based Braccio demo refinement

On 2026-03-26 and shortly after, the commit history shows:

- `Wrist_update_demo`;
- `FIxed_Elbow`;
- later `Pinky angle calculation`.

These point to refinement of the first live gesture-to-robot demo.

The MediaPipe-based Braccio scripts show that the following ideas were implemented:

- use `mediapipe.solutions.hands` for hand landmark extraction;
- track one hand;
- compute a palm vertical position from selected landmarks;
- compute a geometric angle from the hand;
- map that information to Braccio servo targets;
- smooth the resulting commands;
- limit per-step motion for safety;
- send commands through serial;
- use a heartbeat and failsafe timeout so the arm does not remain uncontrolled.

This phase turned the project into a real closed-loop prototype.

### Phase D: DeGirum/Hailo integration

On 2026-03-28, the repo gained DeGirum scripts, Hailo models, and setup documentation.

This was a major milestone because it introduced hardware acceleration into the project. The new assets include:

- hand detector model files;
- hand landmark model files;
- scripts for webcam testing;
- scripts for Braccio control using DeGirum inference.

This means the perception stack was no longer limited to a CPU-only MediaPipe approach. The project now had two pathways:

- a classic CPU-based path;
- an accelerated Hailo-based path.

### Phase E: Wrist-rotation implementation

On 2026-04-20, the commit `implementare_rotatie_incheietura` added the dedicated wrist-rotation work.

This marks an important expansion of the control space. The project moved beyond the original simple mapping of:

- hand height -> elbow;
- finger opening -> gripper.

It now introduced a specific dynamic wrist-rotation behavior:

- the hand moving between top and bottom regions drives elbow motion;
- the wrist rotation servo switches between a home angle and a rotated angle;
- the rotation snaps at motion endpoints so the gesture is visually and mechanically clear.

This is a much more explicit motion pattern and suggests the project is moving from "basic teleoperation" toward structured robot gestures or task primitives.

### Phase F: Benchmark instrumentation

On 2026-04-21, dedicated benchmark scripts were added for both backends:

- `dg_braccio_wrist_rotate_benchmark.py`;
- `mp_braccio_wrist_rotate_benchmark.py`.

This is another major milestone because it means the project stopped being only qualitative and became measurable. Instead of saying "Hailo feels faster," the code now records:

- frame timing;
- inference timing;
- control time;
- serial time;
- display time;
- FPS;
- hand detection frequency;
- landmark output frequency;
- pose-send frequency.

This moves the project from demo engineering into performance analysis.

### Phase G: Tests, result collection, and comparison report

On 2026-04-23, the repo gained:

- summary CSV files;
- per-frame benchmark CSV files;
- `compare_benchmark_results.py`;
- `benchmark_comparison.md`.

At this point, the project produced not only working demos, but also a reproducible evidence trail:

- raw frame-by-frame measurements;
- run-level summaries;
- backend-level aggregate comparison.

This is the point where the project becomes much easier to explain in a report, presentation, or thesis chapter because the implementation now has supporting metrics.

## 5. Detailed Technical Explanation of What Was Built

## 5.1. The First Practical Control Idea

The core initial idea of the project is preserved very clearly in `Braccio-test/PROJECT_PSEUDOCODE.md`.

The logic is intentionally simple and practical:

- keep most robot joints fixed so the robot is easier to control;
- use hand vertical position to drive elbow motion;
- use a finger opening metric to drive the gripper;
- smooth commands to avoid jerks;
- limit motion speed for safety;
- send a structured command string to Arduino.

This was a strong starting point because it reduced the control problem to two hand-derived signals. That made testing easier and gave the project a stable base before adding more complex gestures.

## 5.2. MediaPipe-Based Braccio Control

The script `Braccio-test/braccio_suture_demo_test.py` represents the main MediaPipe-driven Braccio control pattern.

The important ideas inside this script are:

- serial configuration for the Braccio Arduino connection;
- camera initialization;
- joint definitions and servo limits;
- a neutral pose used for safe startup and fallback;
- a set of fixed joints so only the intended joints move;
- smoothing coefficients per joint;
- maximum step sizes per joint;
- a timeout after hand loss;
- packet construction in the form:

`CMD,base,shoulder,elbow,wrist_vertical,wrist_rotation,gripper`

The script computes hand-derived features from MediaPipe landmarks. In the version in `Braccio-test/`, palm height is derived from:

- wrist;
- index MCP;
- pinky MCP.

The finger control metric uses an angle based on:

- thumb tip;
- wrist;
- pinky MCP.

This metric is then mapped to the gripper servo. The use of angle geometry instead of just distance suggests an attempt to make the gripper control more robust to hand scale and camera position.

Other important design details in this stage:

- poses are clamped to servo limits;
- movement is smoothed with an exponential follow model;
- motion is rate-limited before sending;
- the script only sends when the pose changes enough or when a heartbeat is due;
- if the hand disappears, the system either holds or returns to neutral, depending on the configured failsafe mode.

This is already a good example of applied human-in-the-loop robot control, not just a vision demo.

## 5.3. Arduino Safety Bridge

The file `Braccio-test/Script-Arduino/control_arm_bridge_safe.ino` shows that the robot side was also treated carefully.

This Arduino sketch:

- starts serial communication;
- initializes the Braccio library;
- moves the arm to a neutral pose on startup;
- reads newline-terminated `CMD,...` packets;
- parses six servo values;
- clamps all joints to safe limits;
- executes the movement through `Braccio.ServoMovement`;
- returns `OK` for valid commands and `ERR` for invalid ones;
- contains a watchdog that automatically returns the arm to neutral if commands stop arriving for too long.

This is very important. It means the project does not rely only on the PC-side script being correct. There is also a second safety layer on the Arduino side.

That is one of the strongest engineering choices in the repo so far.

## 5.4. Manual Serial Test Utilities

Inside `Braccio-test/python_scripts_Emil/`, the scripts `SerialComArduino.py` and `testArm.py` provide a simpler manual testing path.

These scripts make it possible to:

- open a serial link manually;
- send direct servo commands;
- test the arm without the vision system;
- verify communication independently.

This kind of tooling is useful because it separates robot debugging from camera debugging. If the arm fails, you can test the serial path alone.

## 5.5. Suturing-Like Motion Experiment

The file `Test Scripts/braccio_suture_demo.py` shows that the project evolved beyond direct gesture mirroring. It introduced a higher-level structured motion sequence inspired by suturing-like behavior.

This script contains:

- a reference manual pose;
- a stitch sequence built from repeated phases such as `down`, `up_right`, and `recover`;
- offsets applied to specific joints during each phase;
- trigger thresholds based on hand pinch behavior;
- timing for phase durations;
- smoothing and safety logic similar to the simpler demo.

This is a major conceptual step. Instead of mapping every hand change continuously to every robot joint, the system begins to interpret hand input as a trigger for a predesigned robotic action pattern.

That shows the project is exploring not only teleoperation, but also task-level interaction.

## 5.6. Calibration and Inverse Kinematics

The folder `Braccio-test/braccio_ik_reference/` documents another important direction: moving from joint-space heuristics toward Cartesian control.

The file `CALIBRATION_STEPS.md` describes a calibration procedure for:

- arm geometry;
- servo zero offsets;
- operation zone dimensions;
- tool-tip offset;
- sample-based fitting of the tool tip.

The file `braccio_kinematics.py` provides:

- servo clamping and normalization;
- forward kinematics;
- inverse kinematics;
- calibration loading;
- tool-tip fitting support.

The file `braccio_cartesian_controller.py` builds on that and adds:

- operation-zone mapping from hand motion;
- tool pitch and roll estimation from the hand;
- pinch-based gripper behavior;
- path generation in Cartesian space;
- a runner that executes a multi-stage path through time.

This means the project has already gone beyond raw servo tuning and started formal robot modeling. That is an advanced direction and gives the project a clear future path if more precise manipulation is needed.

## 5.7. Early Hailo Exploration in `demo hailo part1`

The scripts in `demo hailo part1/` show the earlier acceleration experiments before the DeGirum version was organized.

The non-AI script `rpi_hand_ui_hailo_noai.py` includes:

- ROI-based camera processing;
- simple skin segmentation in HSV;
- contour extraction;
- centroid-based left/right/center command logic;
- MediaPipe landmarks for hand visualization;
- TCP sending of commands;
- CSV logging via `logger.py`.

The AI script `rpi_hand_ui_hailo_ai.py` extends this by adding:

- direct Hailo model loading through `hailo_platform`;
- manual HEF/network setup;
- an ROI inference path;
- a Hailo score used as an extra validation signal;
- integration of the result with the UI and TCP command sending.

This phase is important because it shows that the project first experimented with Hailo in a more direct, lower-level way before adopting the cleaner DeGirum interface later.

So this folder is not obsolete; it represents the exploratory bridge between concept and the current Hailo path.

## 5.8. DeGirum/Hailo Braccio Hand-Landmarks Demo

The script `demo-hailo-part2/Degirum/scripts/dg_braccio_hand_landmarks.py` shows a more mature accelerated implementation.

Its design is very clever because it reuses logic from the earlier Braccio script instead of rewriting everything from scratch. The script:

- loads a DeGirum detector model and landmark model;
- captures frames from the camera;
- chooses the best detected hand bounding box;
- expands that box slightly for landmark robustness;
- runs the landmark model on the crop;
- converts crop-relative landmarks back into full-frame normalized coordinates;
- reuses the existing hand-feature and motion-mapping functions from the Braccio configuration module;
- reuses the same smoothing, pose sanitization, rate limiting, heartbeat, and serial send logic.

This is a strong design choice because it isolates the backend change mainly to perception. The motion-control logic remains conceptually consistent.

In other words, the project did not build two unrelated systems. It built one control concept that can run on two perception backends.

## 5.9. Wrist-Rotation Demo on DeGirum/Hailo

The script `demo-hailo-part2/Degirum/scripts/dg_braccio_wrist_rotate.py` is one of the clearest signs of feature maturity.

It implements a specific and intentional gesture-controlled routine:

- the hand still controls elbow movement through palm height;
- the wrist rotation joint is no longer fixed;
- instead, it toggles between a home angle and a rotated angle;
- the system tracks whether the hand is in a top zone or bottom zone;
- when the hand reaches one of these endpoints, the wrist rotation snaps immediately to the expected angle;
- the gripper is held fixed during this demo so the experiment isolates wrist-roll behavior.

This creates a repeatable movement cycle that is easier to observe, test, and benchmark than unconstrained live gesture control.

That is probably why this script became the basis for the benchmark setup.

## 5.10. Benchmark Framework

The benchmark scripts:

- `demo-hailo-part2/Degirum/scripts/dg_braccio_wrist_rotate_benchmark.py`
- `demo-hailo-part2/Degirum/scripts/mp_braccio_wrist_rotate_benchmark.py`

are a very significant achievement.

Together, these scripts add:

- warm-up support;
- frame-by-frame CSV logging;
- a summary CSV row per run;
- run labels and timestamps;
- measurement limits by frame count or duration;
- measurement of capture, detector, landmark, control, serial, display, inference, pipeline, and total frame times;
- counts of hand-detected frames, landmark frames, and pose-sent frames;
- optional dry-run serial mode;
- a consistent schema across Hailo and CPU backends.

This matters because it converts the demos into measurable experiments. The project is no longer relying only on visual impression.

Another strength is that these benchmark scripts preserve the behavior of the original demos while adding instrumentation around them. That reduces the risk that the benchmark accidentally measures a different behavior than the real demo.

## 5.11. Benchmark Comparison Report

The script `demo-hailo-part2/Degirum/scripts/compare_benchmark_results.py` reads the Hailo and CPU summary CSVs and produces a markdown report.

The script:

- loads one row per benchmark run;
- converts counts into percentages using measured frames;
- computes per-backend averages;
- computes FPS min/max range;
- generates a backend comparison table;
- generates a per-run comparison table;
- writes the result to `benchmark_results/benchmark_comparison.md`.

This is valuable because it gives the project a reusable reporting step. Once new benchmark data exists, a comparison report can be rebuilt immediately.

## 6. What the Current Benchmark Data Shows

The current benchmark results already stored in the repository compare:

- `Hailo/DeGirum`, with 3 runs;
- `CPU/MediaPipe`, with 4 runs.

### Aggregate result currently stored

| Backend | Runs | Avg duration (s) | Avg FPS | FPS range | Avg frame mean (ms) | Avg inference mean (ms) | Avg control mean (ms) | Avg serial mean (ms) | Avg hand detect % | Avg landmarks % | Avg pose sends % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hailo/DeGirum | 3 | 40.07 | 27.99 | 27.60 - 28.55 | 37.57 | 28.57 | 9.93 | 0.79 | 82.16 | 74.70 | 28.35 |
| CPU/MediaPipe | 4 | 25.19 | 12.77 | 5.71 - 19.92 | 220.52 | 44.48 | 0.10 | 169.47 | 68.78 | 68.78 | 63.48 |

### Important interpretation

The CPU average is strongly affected by two degraded runs at approximately:

- `6.05 FPS`
- `5.71 FPS`

In those degraded runs, the serial time becomes extremely large, around:

- `324.96 ms`
- `348.12 ms`

This is important because it suggests that the CPU comparison is not only about MediaPipe inference speed. Some of the slowdown appears to come from the serial/communication side of the loop.

### Stable CPU-only view

If we isolate the two good CPU runs, the average becomes approximately:

- `19.65 FPS`
- `53.60 ms` average frame time
- `43.40 ms` average inference time
- `2.41 ms` average serial time

### Current conclusion from the benchmark

The most defensible conclusion right now is:

- Hailo/DeGirum is clearly faster overall than the full CPU result, about `2.19x` in average FPS.
- Hailo is also much more stable, because its FPS varies only slightly across the recorded runs.
- Even if the degraded CPU runs are excluded, Hailo still leads by about `1.42x` in FPS.
- Hailo has meaningfully lower inference time than CPU in this setup.
- The CPU backend can perform reasonably in good runs, but it is currently less stable under the recorded test conditions.

There is also an important nuance:

- `hand detect %`, `landmarks %`, and `pose sends %` are event rates, not direct accuracy metrics.

So the benchmark currently gives a strong performance comparison, but not a full accuracy comparison.

## 7. What Has Been Achieved So Far

At this stage, the project has already achieved a lot:

- a working concept for mapping hand gestures to robot motion;
- a MediaPipe-based live control implementation;
- a safe serial command protocol for the Braccio arm;
- Arduino-side clamping and timeout-to-neutral behavior;
- joint smoothing and rate limiting on the host side;
- a suturing-like patterned motion experiment;
- a calibration and inverse-kinematics reference path;
- an early Hailo experimental prototype;
- a more mature DeGirum/Hailo implementation;
- a dedicated wrist-rotation demo;
- a reusable benchmark framework;
- stored measurement data and a generated comparison report.

This is not a small amount of work. The repository shows substantial progress across software, robotics logic, and experimental validation.

## 8. Why This Progress Matters

From a project-development perspective, the work done so far is valuable because it solved the problem in layers:

### First, basic actuation had to work

Without reliable command sending and safe arm motion, gesture control would have been meaningless.

### Then, simple gesture mapping had to be made usable

The move to fixed joints, elbow mapping, and gripper mapping reduced complexity and made testing realistic.

### Then, more advanced behavior had to be explored

This led to suturing patterns, wrist rotation, and Cartesian-control thinking.

### Finally, performance had to be measured

This is what the benchmark framework and comparison report now provide.

This staged evolution is exactly how a real robotics/computer-vision project should grow.

## 9. Current Limitations and Open Issues

The repo also shows that some areas are still in progress or would benefit from cleanup.

### 9.1. Benchmark instability on CPU

The CPU benchmark contains two clearly degraded runs. This means:

- the comparison is useful, but should be interpreted carefully;
- serial timing or blocking behavior likely needs more investigation;
- a filtered "stable runs only" table would improve clarity.

### 9.2. Some logic exists in multiple places

There are related scripts across:

- `Braccio-test/`
- `demo hailo part1/`
- `demo-hailo-part2/`
- `Test Scripts/`

This is natural in an experimental project, but it means long-term maintainability would improve if the shared control logic were consolidated.

### 9.3. Accuracy and task success are not fully benchmarked yet

The current benchmark focuses on timing and event rates. It does not yet fully measure:

- landmark quality;
- gesture classification correctness;
- robot endpoint precision;
- task completion quality for suturing-like movement.

### 9.4. Calibration work appears promising but not fully integrated everywhere

The inverse-kinematics and calibration work is advanced, but it still looks like a reference branch of the project rather than the one unified control stack used everywhere.

## 10. Recommended Next Steps

Based on the current state of the repository, the most logical next steps would be:

1. add a filtered benchmark summary that excludes the obviously degraded CPU runs;
2. investigate the source of high serial times in the slow CPU benchmark runs;
3. consolidate shared helper logic so MediaPipe and DeGirum variants reuse more common code;
4. decide whether the future direction is:
   - direct gesture teleoperation,
   - predefined task triggers,
   - or full Cartesian control with calibration;
5. extend the evaluation with task-level metrics, not only timing metrics;
6. document hardware setup and exact run conditions more formally for reproducibility.

## 11. Final Summary

So far, this project has evolved from a simple idea of "move a robot arm with hand gestures" into a much richer system with:

- real-time camera-based perception;
- robot-safe actuation;
- multiple control strategies;
- Hailo acceleration;
- reproducible benchmarking;
- measurable evidence of backend performance differences.

The strongest current achievements are:

- the complete end-to-end gesture-to-Braccio control path;
- the safe command bridge on Arduino;
- the DeGirum/Hailo wrist-rotation pipeline;
- the benchmark framework and the first comparative performance results.

In short, the project is no longer in the "concept only" stage. It already has working demos, structured experiments, hardware acceleration, and performance data. That is strong and meaningful progress.

## 12. Short One-Paragraph Version for a Report

Up to 25.04.2026, the project has progressed from early Braccio arm stabilization and servo-control tests to a complete vision-based gesture-control pipeline. A first MediaPipe implementation was developed to map hand position and finger geometry to robot joint commands, with smoothing, rate limiting, and failsafe behavior. In parallel, safe Arduino bridge code was created to parse commands, clamp servo limits, and return the arm to neutral on timeout. The project then expanded into suturing-like motion experiments and an inverse-kinematics/calibration reference branch for Cartesian control. After that, a Hailo-based perception path was introduced, first through direct experiments and later through a cleaner DeGirum-based implementation using local hand-detection and landmark models. This led to a dedicated wrist-rotation demo and, finally, to a benchmark framework that logs frame-by-frame and run-level timing data for both Hailo and CPU backends. The current measurements show that Hailo/DeGirum is clearly faster and more stable than the CPU pipeline in the recorded tests, while the CPU results are affected by several degraded runs with very high serial timing.
