#!/usr/bin/env python3
"""
Build a markdown comparison table from the Hailo and CPU benchmark summary CSVs.

The output is designed to be easy to read in the terminal or save as a report
under benchmark_results/ after collecting new runs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "benchmark_results"
DEFAULT_HAILO_SUMMARY = DEFAULT_RESULTS_DIR / "dg_wrist_rotate_summary.csv"
DEFAULT_CPU_SUMMARY = DEFAULT_RESULTS_DIR / "mp_wrist_rotate_summary.csv"
DEFAULT_OUTPUT_MD = DEFAULT_RESULTS_DIR / "benchmark_comparison.md"


def _read_summary_rows(path: Path, backend: str) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            measured_frames = _to_int(raw["measured_frames"])
            hand_detected_frames = _to_int(raw["hand_detected_frames"])
            landmarks_frames = _to_int(raw["landmarks_frames"])
            pose_sent_frames = _to_int(raw["pose_sent_frames"])

            row = {
                "backend": backend,
                "run_id": raw["run_id"],
                "run_label": raw["run_label"],
                "measurement_duration_s": _to_float(raw["measurement_duration_s"]),
                "fps_mean": _to_float(raw["fps_mean"]),
                "fps_median": _to_float(raw["fps_median"]),
                "frame_time_mean_ms": _to_float(raw["frame_time_mean_ms"]),
                "frame_time_p95_ms": _to_float(raw["frame_time_p95_ms"]),
                "inference_time_mean_ms": _to_float(raw["inference_time_mean_ms"]),
                "control_time_mean_ms": _to_float(raw["control_time_mean_ms"]),
                "serial_time_mean_ms": _to_float(raw["serial_time_mean_ms"]),
                "display_time_mean_ms": _to_float(raw["display_time_mean_ms"]),
                "measured_frames": measured_frames,
                "hand_detect_rate_pct": _rate_pct(hand_detected_frames, measured_frames),
                "landmarks_rate_pct": _rate_pct(landmarks_frames, measured_frames),
                "pose_send_rate_pct": _rate_pct(pose_sent_frames, measured_frames),
                "frame_csv": raw["frame_csv"],
            }
            rows.append(row)
    return rows


def _to_float(value: str) -> float:
    return float(value) if value else 0.0


def _to_int(value: str) -> int:
    return int(value) if value else 0


def _rate_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def _fmt(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def _make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _aggregate_backend(rows: list[dict]) -> dict:
    return {
        "runs": len(rows),
        "duration_avg_s": mean(row["measurement_duration_s"] for row in rows),
        "fps_mean_avg": mean(row["fps_mean"] for row in rows),
        "fps_mean_min": min(row["fps_mean"] for row in rows),
        "fps_mean_max": max(row["fps_mean"] for row in rows),
        "frame_time_mean_avg_ms": mean(row["frame_time_mean_ms"] for row in rows),
        "frame_time_p95_avg_ms": mean(row["frame_time_p95_ms"] for row in rows),
        "inference_time_mean_avg_ms": mean(row["inference_time_mean_ms"] for row in rows),
        "control_time_mean_avg_ms": mean(row["control_time_mean_ms"] for row in rows),
        "serial_time_mean_avg_ms": mean(row["serial_time_mean_ms"] for row in rows),
        "display_time_mean_avg_ms": mean(row["display_time_mean_ms"] for row in rows),
        "hand_detect_rate_avg_pct": mean(row["hand_detect_rate_pct"] for row in rows),
        "landmarks_rate_avg_pct": mean(row["landmarks_rate_pct"] for row in rows),
        "pose_send_rate_avg_pct": mean(row["pose_send_rate_pct"] for row in rows),
    }


def build_report(hailo_rows: list[dict], cpu_rows: list[dict], hailo_path: Path, cpu_path: Path) -> str:
    all_rows = hailo_rows + cpu_rows
    grouped = {
        "Hailo/DeGirum": hailo_rows,
        "CPU/MediaPipe": cpu_rows,
    }

    aggregate_rows = []
    for backend, rows in grouped.items():
        stats = _aggregate_backend(rows)
        aggregate_rows.append(
            [
                backend,
                str(stats["runs"]),
                _fmt(stats["duration_avg_s"]),
                _fmt(stats["fps_mean_avg"]),
                f"{_fmt(stats['fps_mean_min'])} - {_fmt(stats['fps_mean_max'])}",
                _fmt(stats["frame_time_mean_avg_ms"]),
                _fmt(stats["frame_time_p95_avg_ms"]),
                _fmt(stats["inference_time_mean_avg_ms"]),
                _fmt(stats["control_time_mean_avg_ms"]),
                _fmt(stats["serial_time_mean_avg_ms"]),
                _fmt(stats["hand_detect_rate_avg_pct"]),
                _fmt(stats["landmarks_rate_avg_pct"]),
                _fmt(stats["pose_send_rate_avg_pct"]),
            ]
        )

    per_run_rows = []
    for row in sorted(all_rows, key=lambda item: (item["backend"], item["run_id"])):
        per_run_rows.append(
            [
                row["backend"],
                row["run_id"],
                _fmt(row["measurement_duration_s"]),
                _fmt(row["fps_mean"]),
                _fmt(row["frame_time_mean_ms"]),
                _fmt(row["frame_time_p95_ms"]),
                _fmt(row["inference_time_mean_ms"]),
                _fmt(row["control_time_mean_ms"]),
                _fmt(row["serial_time_mean_ms"]),
                _fmt(row["hand_detect_rate_pct"]),
                _fmt(row["landmarks_rate_pct"]),
                _fmt(row["pose_send_rate_pct"]),
            ]
        )

    aggregate_table = _make_markdown_table(
        [
            "Backend",
            "Runs",
            "Avg duration (s)",
            "Avg FPS",
            "FPS range",
            "Avg frame mean (ms)",
            "Avg frame p95 (ms)",
            "Avg inference mean (ms)",
            "Avg control mean (ms)",
            "Avg serial mean (ms)",
            "Avg hand detect %",
            "Avg landmarks %",
            "Avg pose sends %",
        ],
        aggregate_rows,
    )

    per_run_table = _make_markdown_table(
        [
            "Backend",
            "Run ID",
            "Duration (s)",
            "FPS mean",
            "Frame mean (ms)",
            "Frame p95 (ms)",
            "Inference mean (ms)",
            "Control mean (ms)",
            "Serial mean (ms)",
            "Hand detect %",
            "Landmarks %",
            "Pose sends %",
        ],
        per_run_rows,
    )

    return "\n".join(
        [
            "# Benchmark Comparison",
            "",
            "Aggregate backend comparison",
            "",
            aggregate_table,
            "",
            "Per-run comparison",
            "",
            per_run_table,
            "",
            "Source files",
            "",
            f"- Hailo summary: `{hailo_path}`",
            f"- CPU summary: `{cpu_path}`",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hailo-summary",
        default=str(DEFAULT_HAILO_SUMMARY),
        help="Hailo/DeGirum benchmark summary CSV.",
    )
    parser.add_argument(
        "--cpu-summary",
        default=str(DEFAULT_CPU_SUMMARY),
        help="CPU/MediaPipe benchmark summary CSV.",
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_MD),
        help="Markdown file to write with the comparison tables.",
    )
    args = parser.parse_args()

    hailo_path = Path(args.hailo_summary).expanduser().resolve()
    cpu_path = Path(args.cpu_summary).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()

    hailo_rows = _read_summary_rows(hailo_path, "Hailo/DeGirum")
    cpu_rows = _read_summary_rows(cpu_path, "CPU/MediaPipe")
    report = build_report(hailo_rows, cpu_rows, hailo_path, cpu_path)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWrote markdown report to: {output_md}")


if __name__ == "__main__":
    main()
