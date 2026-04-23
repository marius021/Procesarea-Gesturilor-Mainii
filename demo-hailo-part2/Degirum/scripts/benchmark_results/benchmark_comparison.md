# Benchmark Comparison

Aggregate backend comparison

| Backend | Runs | Avg duration (s) | Avg FPS | FPS range | Avg frame mean (ms) | Avg frame p95 (ms) | Avg inference mean (ms) | Avg control mean (ms) | Avg serial mean (ms) | Avg hand detect % | Avg landmarks % | Avg pose sends % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hailo/DeGirum | 3 | 40.07 | 27.99 | 27.60 - 28.55 | 37.57 | 48.44 | 28.57 | 9.93 | 0.79 | 82.16 | 74.70 | 28.35 |
| CPU/MediaPipe | 4 | 25.19 | 12.77 | 5.71 - 19.92 | 220.52 | 297.81 | 44.48 | 0.10 | 169.47 | 68.78 | 68.78 | 63.48 |

Per-run comparison

| Backend | Run ID | Duration (s) | FPS mean | Frame mean (ms) | Frame p95 (ms) | Inference mean (ms) | Control mean (ms) | Serial mean (ms) | Hand detect % | Landmarks % | Pose sends % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU/MediaPipe | mp_wrist_rotate_cpu_20260423_213918 | 25.13 | 19.39 | 53.78 | 71.19 | 43.16 | 0.09 | 2.01 | 87.12 | 87.12 | 42.06 |
| CPU/MediaPipe | mp_wrist_rotate_cpu_20260423_214013 | 25.09 | 19.92 | 53.41 | 94.34 | 43.64 | 0.09 | 2.80 | 87.82 | 87.82 | 44.02 |
| CPU/MediaPipe | mp_wrist_rotate_cpu_20260423_214051 | 25.37 | 6.05 | 374.99 | 513.47 | 44.36 | 0.16 | 324.96 | 72.06 | 72.06 | 89.71 |
| CPU/MediaPipe | mp_wrist_rotate_cpu_20260423_214132 | 25.15 | 5.71 | 399.91 | 512.26 | 46.77 | 0.05 | 348.12 | 28.12 | 28.12 | 78.12 |
| Hailo/DeGirum | dg_wrist_rotate_hailo_20260423_212157 | 40.07 | 28.55 | 37.03 | 48.12 | 26.35 | 8.34 | 0.89 | 68.55 | 58.81 | 29.59 |
| Hailo/DeGirum | dg_wrist_rotate_hailo_20260423_212251 | 40.07 | 27.82 | 38.13 | 49.06 | 29.50 | 10.75 | 1.29 | 89.11 | 86.91 | 28.08 |
| Hailo/DeGirum | dg_wrist_rotate_hailo_20260423_212441 | 40.06 | 27.60 | 37.55 | 48.13 | 29.85 | 10.70 | 0.18 | 88.81 | 78.36 | 27.38 |

Source files

- Hailo summary: `/home/maurice/Desktop/Procesarea-Gesturilor-Mainii/demo-hailo-part2/Degirum/scripts/benchmark_results/dg_wrist_rotate_summary.csv`
- CPU summary: `/home/maurice/Desktop/Procesarea-Gesturilor-Mainii/demo-hailo-part2/Degirum/scripts/benchmark_results/mp_wrist_rotate_summary.csv`
