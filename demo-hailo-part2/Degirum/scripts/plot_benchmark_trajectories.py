#!/usr/bin/env python3
"""
Generate trajectory and summary plots from stored wrist-rotation benchmark CSVs.

The script reads the existing summary CSVs to discover per-run frame CSVs,
filters to measured frames only, and writes PNG plots under benchmark_results/.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "benchmark_results"
DEFAULT_HAILO_SUMMARY = RESULTS_DIR / "dg_wrist_rotate_summary.csv"
DEFAULT_CPU_SUMMARY = RESULTS_DIR / "mp_wrist_rotate_summary.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "plots"

MPLCONFIG_DIR = Path("/tmp/codex-matplotlib-cache")
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing matplotlib. Run this script with a repo venv, for example "
        "'./Braccio-test/.venv/bin/python' or "
        "'./Braccio-test/Degirum/venv_hailo_rpi_examples/bin/python'."
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_STYLE = {
    "hailo": {
        "label": "Hailo/DeGirum",
        "color": "#1f77b4",
    },
    "cpu": {
        "label": "CPU/MediaPipe",
        "color": "#ff7f0e",
    },
}

PHASE_BG = {
    "TOP_HOME": "#dceefa",
    "BOTTOM_ROTATED": "#fff2cc",
}


@dataclass
class SummaryRow:
    backend_key: str
    backend_label: str
    run_id: str
    run_label: str
    frame_csv: Path
    fps_mean: float
    frame_time_mean_ms: float
    inference_time_mean_ms: float
    control_time_mean_ms: float
    serial_time_mean_ms: float
    hand_detect_rate_pct: float
    landmarks_rate_pct: float
    pose_send_rate_pct: float


def _to_float(value: str) -> float:
    return float(value) if value else 0.0


def _to_int(value: str) -> int:
    return int(value) if value else 0


def _rate_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def _resolve_frame_csv(raw_path: str, summary_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (summary_path.parent / path).resolve()


def _read_summary_rows(summary_path: Path, backend_key: str) -> list[SummaryRow]:
    backend_label = BACKEND_STYLE[backend_key]["label"]
    rows: list[SummaryRow] = []
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            measured_frames = _to_int(raw["measured_frames"])
            hand_detected_frames = _to_int(raw["hand_detected_frames"])
            landmarks_frames = _to_int(raw["landmarks_frames"])
            pose_sent_frames = _to_int(raw["pose_sent_frames"])
            rows.append(
                SummaryRow(
                    backend_key=backend_key,
                    backend_label=backend_label,
                    run_id=raw["run_id"],
                    run_label=raw["run_label"],
                    frame_csv=_resolve_frame_csv(raw["frame_csv"], summary_path),
                    fps_mean=_to_float(raw["fps_mean"]),
                    frame_time_mean_ms=_to_float(raw["frame_time_mean_ms"]),
                    inference_time_mean_ms=_to_float(raw["inference_time_mean_ms"]),
                    control_time_mean_ms=_to_float(raw["control_time_mean_ms"]),
                    serial_time_mean_ms=_to_float(raw["serial_time_mean_ms"]),
                    hand_detect_rate_pct=_rate_pct(hand_detected_frames, measured_frames),
                    landmarks_rate_pct=_rate_pct(landmarks_frames, measured_frames),
                    pose_send_rate_pct=_rate_pct(pose_sent_frames, measured_frames),
                )
            )
    return rows


def _read_frame_rows(frame_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with frame_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if _to_int(raw["measured"]) != 1:
                continue
            rows.append(
                {
                    "time_s": _to_float(raw["elapsed_measurement_s"]),
                    "total_frame_time_ms": _to_float(raw["total_frame_time_ms"]),
                    "inference_time_ms": _to_float(raw["inference_time_ms"]),
                    "serial_time_ms": _to_float(raw["serial_time_ms"]),
                    "fps": _to_float(raw["fps"]),
                    "hand_detected": _to_int(raw["hand_detected"]),
                    "landmarks_found": _to_int(raw["landmarks_found"]),
                    "pose_sent": _to_int(raw["pose_sent"]),
                    "palm_y": float(raw["palm_y"]) if raw["palm_y"] else None,
                    "wrist_phase": raw["wrist_phase"],
                    "wrist_transition": raw["wrist_transition"],
                    "candidate_elbow": _to_float(raw["candidate_elbow"]),
                    "candidate_wrist_rotation": _to_float(raw["candidate_wrist_rotation"]),
                    "current_elbow": _to_float(raw["current_elbow"]),
                    "current_wrist_rotation": _to_float(raw["current_wrist_rotation"]),
                }
            )
    return rows


def _short_run_name(run_id: str) -> str:
    parts = run_id.split("_")
    if len(parts) >= 3:
        return f"{parts[-2]}-{parts[-1]}"
    return run_id


def _phase_segments(rows: list[dict]) -> list[tuple[str, float, float]]:
    if not rows:
        return []

    segments: list[tuple[str, float, float]] = []
    start_time = rows[0]["time_s"]
    phase = rows[0]["wrist_phase"]
    last_time = rows[0]["time_s"]

    for row in rows[1:]:
        row_phase = row["wrist_phase"]
        row_time = row["time_s"]
        if row_phase != phase:
            segments.append((phase, start_time, last_time))
            phase = row_phase
            start_time = row_time
        last_time = row_time

    segments.append((phase, start_time, last_time))
    return segments


def _apply_phase_background(ax, rows: list[dict]) -> None:
    for phase, start_time, end_time in _phase_segments(rows):
        bg = PHASE_BG.get(phase)
        if bg is None:
            continue
        ax.axvspan(start_time, end_time, facecolor=bg, alpha=0.35, linewidth=0)


def _event_times(rows: list[dict], key: str) -> list[float]:
    return [row["time_s"] for row in rows if row[key]]


def _transition_times(rows: list[dict]) -> list[tuple[float, str]]:
    transitions: list[tuple[float, str]] = []
    for row in rows:
        if row["wrist_transition"]:
            transitions.append((row["time_s"], row["wrist_transition"]))
    return transitions


def _normalized_time(rows: list[dict]) -> list[float]:
    if not rows:
        return []
    max_time = max(row["time_s"] for row in rows)
    if max_time <= 1e-9:
        return [0.0 for _ in rows]
    return [row["time_s"] / max_time for row in rows]


def _plot_run_trajectory(summary: SummaryRow, rows: list[dict], output_dir: Path) -> Path:
    style = BACKEND_STYLE[summary.backend_key]
    output_path = output_dir / f"{summary.run_id}_trajectory.png"

    times = [row["time_s"] for row in rows]
    elbow_error = [abs(row["candidate_elbow"] - row["current_elbow"]) for row in rows]
    wrist_error = [
        abs(row["candidate_wrist_rotation"] - row["current_wrist_rotation"])
        for row in rows
    ]
    palm_times = [row["time_s"] for row in rows if row["palm_y"] is not None]
    palm_values = [row["palm_y"] for row in rows if row["palm_y"] is not None]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 13),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        f"{summary.backend_label} trajectory: {summary.run_id}",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    _apply_phase_background(ax, rows)
    ax.plot(
        times,
        [row["candidate_elbow"] for row in rows],
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label="Elbow target",
    )
    ax.plot(
        times,
        [row["current_elbow"] for row in rows],
        color="#1f77b4",
        linewidth=2.1,
        label="Elbow current",
    )
    ax.plot(
        times,
        [row["candidate_wrist_rotation"] for row in rows],
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label="Wrist target",
    )
    ax.plot(
        times,
        [row["current_wrist_rotation"] for row in rows],
        color="#ff7f0e",
        linewidth=2.1,
        label="Wrist current",
    )
    ax.set_ylabel("Servo angle (deg)")
    ax.set_title("Joint command trajectories")
    ax.legend(loc="upper right", ncol=2)

    ax = axes[1]
    _apply_phase_background(ax, rows)
    if palm_times:
        ax.plot(
            palm_times,
            palm_values,
            color="#4d4d4d",
            linewidth=2.0,
            label="Palm Y",
        )
    for transition_time, transition_name in _transition_times(rows):
        ax.axvline(
            transition_time,
            color=style["color"],
            linestyle=":",
            linewidth=1.4,
            alpha=0.8,
        )
        ax.text(
            transition_time,
            ax.get_ylim()[0] if palm_times else 0.0,
            transition_name,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color=style["color"],
        )
    if palm_times:
        ax.invert_yaxis()
    ax.set_ylabel("Palm Y (px)")
    ax.set_title("Hand trajectory and wrist phase")
    if palm_times:
        ax.legend(loc="upper right")

    ax = axes[2]
    _apply_phase_background(ax, rows)
    ax.plot(times, elbow_error, color="#1f77b4", linewidth=1.9, label="|Elbow error|")
    ax.plot(times, wrist_error, color="#ff7f0e", linewidth=1.9, label="|Wrist error|")
    send_times = _event_times(rows, "pose_sent")
    if send_times:
        ax.scatter(
            send_times,
            [0.0] * len(send_times),
            color="#222222",
            marker="|",
            s=200,
            linewidths=1.2,
            label="Pose sent",
            zorder=3,
        )
    ax.set_ylabel("Tracking error (deg)")
    ax.set_title("Tracking error and command send events")
    ax.legend(loc="upper right")

    ax = axes[3]
    _apply_phase_background(ax, rows)
    ax.plot(
        times,
        [row["total_frame_time_ms"] for row in rows],
        color="#6a3d9a",
        linewidth=1.8,
        label="Frame time",
    )
    ax.plot(
        times,
        [row["inference_time_ms"] for row in rows],
        color=style["color"],
        linewidth=1.7,
        label="Inference time",
    )
    ax.plot(
        times,
        [row["serial_time_ms"] for row in rows],
        color="#8c564b",
        linewidth=1.4,
        label="Serial time",
    )
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Measurement time (s)")
    ax.set_title("Per-frame timing trajectory")
    ax.legend(loc="upper right")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_backend_overlay(
    run_rows: list[tuple[SummaryRow, list[dict]]],
    output_dir: Path,
) -> Path:
    output_path = output_dir / "backend_trajectory_overlay.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Backend trajectory overlay across measured runs", fontsize=14, fontweight="bold")

    legend_done: set[str] = set()
    for summary, rows in run_rows:
        if not rows:
            continue
        norm_t = _normalized_time(rows)
        style = BACKEND_STYLE[summary.backend_key]
        label = style["label"] if summary.backend_key not in legend_done else None
        axes[0].plot(
            norm_t,
            [row["current_elbow"] for row in rows],
            color=style["color"],
            alpha=0.62,
            linewidth=1.8,
            label=label,
        )
        axes[1].plot(
            norm_t,
            [row["current_wrist_rotation"] for row in rows],
            color=style["color"],
            alpha=0.62,
            linewidth=1.8,
            label=label,
        )
        axes[2].plot(
            norm_t,
            [row["total_frame_time_ms"] for row in rows],
            color=style["color"],
            alpha=0.55,
            linewidth=1.6,
            label=label,
        )
        legend_done.add(summary.backend_key)

    axes[0].set_title("Current elbow trajectory")
    axes[0].set_ylabel("Elbow (deg)")
    axes[1].set_title("Current wrist-rotation trajectory")
    axes[1].set_ylabel("Wrist rotation (deg)")
    axes[2].set_title("Frame-time trajectory")
    axes[2].set_ylabel("Frame time (ms)")
    axes[2].set_xlabel("Normalized measurement progress")

    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.legend(loc="upper right")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_summary_metrics(summaries: list[SummaryRow], output_dir: Path) -> Path:
    output_path = output_dir / "backend_run_metrics.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Benchmark run metrics", fontsize=14, fontweight="bold")

    ordered = sorted(summaries, key=lambda row: (row.backend_key, row.run_id))
    labels = [_short_run_name(row.run_id) for row in ordered]
    colors = [BACKEND_STYLE[row.backend_key]["color"] for row in ordered]
    x = list(range(len(ordered)))

    metric_specs = [
        ("fps_mean", "FPS mean", "FPS"),
        ("frame_time_mean_ms", "Frame time mean", "ms"),
        ("inference_time_mean_ms", "Inference time mean", "ms"),
        ("control_time_mean_ms", "Control time mean", "ms"),
        ("serial_time_mean_ms", "Serial time mean", "ms"),
    ]

    flat_axes = [axes[0][0], axes[0][1], axes[0][2], axes[1][0], axes[1][1]]
    for ax, (field_name, title, unit) in zip(flat_axes, metric_specs):
        values = [getattr(row, field_name) for row in ordered]
        ax.bar(x, values, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticks(x, labels, rotation=22, ha="right")

    rates_ax = axes[1][2]
    bar_width = 0.24
    hand_rates = [row.hand_detect_rate_pct for row in ordered]
    landmark_rates = [row.landmarks_rate_pct for row in ordered]
    pose_rates = [row.pose_send_rate_pct for row in ordered]
    rates_ax.bar([pos - bar_width for pos in x], hand_rates, width=bar_width, color="#1f77b4", label="Hand detect %")
    rates_ax.bar(x, landmark_rates, width=bar_width, color="#2ca02c", label="Landmarks %")
    rates_ax.bar([pos + bar_width for pos in x], pose_rates, width=bar_width, color="#ff7f0e", label="Pose sends %")
    rates_ax.set_title("Measured-frame event rates")
    rates_ax.set_ylabel("Percent")
    rates_ax.set_xticks(x, labels, rotation=22, ha="right")
    rates_ax.set_ylim(0.0, 105.0)
    rates_ax.legend(loc="upper right")

    handles = [
        plt.Line2D([0], [0], color=BACKEND_STYLE["hailo"]["color"], lw=6, label=BACKEND_STYLE["hailo"]["label"]),
        plt.Line2D([0], [0], color=BACKEND_STYLE["cpu"]["color"], lw=6, label=BACKEND_STYLE["cpu"]["label"]),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_backend_averages(summaries: list[SummaryRow], output_dir: Path) -> Path:
    output_path = output_dir / "backend_average_metrics.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Backend averages", fontsize=14, fontweight="bold")

    grouped: dict[str, list[SummaryRow]] = {"hailo": [], "cpu": []}
    for row in summaries:
        grouped[row.backend_key].append(row)

    labels = [BACKEND_STYLE[key]["label"] for key in ("hailo", "cpu") if grouped[key]]
    colors = [BACKEND_STYLE[key]["color"] for key in ("hailo", "cpu") if grouped[key]]

    def avg(field_name: str) -> list[float]:
        values = []
        for key in ("hailo", "cpu"):
            rows = grouped[key]
            if not rows:
                continue
            values.append(mean(getattr(row, field_name) for row in rows))
        return values

    plots = [
        (axes[0][0], "fps_mean", "Mean FPS", "FPS"),
        (axes[0][1], "frame_time_mean_ms", "Mean frame time", "ms"),
        (axes[1][0], "inference_time_mean_ms", "Mean inference time", "ms"),
        (axes[1][1], "serial_time_mean_ms", "Mean serial time", "ms"),
    ]

    for ax, field_name, title, ylabel in plots:
        values = avg(field_name)
        ax.bar(labels, values, color=colors, alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def build_plots(hailo_summary: Path, cpu_summary: Path, output_dir: Path) -> list[Path]:
    summaries = _read_summary_rows(hailo_summary, "hailo") + _read_summary_rows(cpu_summary, "cpu")
    if not summaries:
        raise SystemExit("No benchmark rows found in the provided summary CSVs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    run_rows: list[tuple[SummaryRow, list[dict]]] = []

    for summary in summaries:
        if not summary.frame_csv.exists():
            print(f"[WARN] Missing frame CSV, skipping: {summary.frame_csv}")
            continue
        rows = _read_frame_rows(summary.frame_csv)
        if not rows:
            print(f"[WARN] No measured rows found, skipping: {summary.frame_csv}")
            continue
        run_rows.append((summary, rows))
        written_paths.append(_plot_run_trajectory(summary, rows, output_dir))

    if not run_rows:
        raise SystemExit("No measured frame rows were available to plot.")

    valid_summaries = [summary for summary, _ in run_rows]
    written_paths.append(_plot_backend_overlay(run_rows, output_dir))
    written_paths.append(_plot_summary_metrics(valid_summaries, output_dir))
    written_paths.append(_plot_backend_averages(valid_summaries, output_dir))
    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hailo-summary",
        default=str(DEFAULT_HAILO_SUMMARY),
        help="Path to the Hailo/DeGirum benchmark summary CSV.",
    )
    parser.add_argument(
        "--cpu-summary",
        default=str(DEFAULT_CPU_SUMMARY),
        help="Path to the CPU/MediaPipe benchmark summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the plot PNGs should be written.",
    )
    args = parser.parse_args()

    written_paths = build_plots(
        Path(args.hailo_summary).expanduser().resolve(),
        Path(args.cpu_summary).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )

    print("Generated plots:")
    for path in written_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
