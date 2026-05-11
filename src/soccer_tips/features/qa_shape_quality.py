"""
Script: qa_shape_quality.py
Project: Soccer TIPS

Overview:
    QA runner for Layer 2 v0.2.3 shape metric quality flags and Layer 3
    readiness filter.

Outputs:
    data/processed/features/layer2_shape_metric_quality_flags.csv
    data/processed/features/layer2_shape_metric_quality_summary.json

Usage from project root:
    $env:PYTHONPATH="src"
    python -m soccer_tips.features.qa_shape_quality
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from soccer_tips.data import common
from soccer_tips.features import build_layer2
from soccer_tips.features import shape_quality


SUMMARY_OUTPUT_PATH = build_layer2.FEATURES_DIR / "layer2_shape_metric_quality_summary.json"
FAIL_EXCLUSION_PERCENTAGE_THRESHOLD = 1.0


def _percentage(count: int, total: int) -> float:
    """Return percentage as a float from 0 to 100."""
    if total == 0:
        return 0.0

    return (count / total) * 100.0


def _count_true(flags: pd.DataFrame, column: str) -> int:
    """Count True values in a boolean flag column."""
    if column not in flags.columns:
        return 0

    return int(flags[column].fillna(False).astype(bool).sum())


def build_shape_quality_summary(flags: pd.DataFrame) -> dict[str, Any]:
    """Build a structured QA summary for shape metric quality flags."""
    total_rows = int(len(flags))
    rows_with_strict_width_warning = _count_true(flags, "team_width_strict_bound_flag")
    rows_with_width_tolerance_failure = _count_true(flags, "team_width_tolerance_fail_flag")
    rows_with_strict_depth_warning = _count_true(flags, "team_depth_strict_bound_flag")
    rows_with_depth_tolerance_failure = _count_true(flags, "team_depth_tolerance_fail_flag")
    low_player_count_rows = _count_true(flags, "low_player_count_flag")
    high_extrapolated_count_rows = _count_true(flags, "high_extrapolated_count_flag")
    high_extrapolated_rate_rows = _count_true(flags, "high_extrapolated_rate_flag")

    if "layer3_shape_metric_include" in flags.columns:
        rows_included_for_layer3 = int(flags["layer3_shape_metric_include"].fillna(False).astype(bool).sum())
    else:
        rows_included_for_layer3 = 0

    rows_excluded_from_layer3 = total_rows - rows_included_for_layer3
    exclusion_percentage = _percentage(rows_excluded_from_layer3, total_rows)

    if "quality_warning_count" in flags.columns:
        rows_with_any_warning = int((pd.to_numeric(flags["quality_warning_count"], errors="coerce").fillna(0) > 0).sum())
    else:
        rows_with_any_warning = 0

    warning_percentage = _percentage(rows_with_any_warning, total_rows)

    status = "pass"
    status_reason = (
        "No tolerance failures or low-player-count exclusions were found. Strict-bound and "
        "extrapolation flags remain available as non-destructive warnings."
    )

    if rows_excluded_from_layer3 > 0:
        status = "warning"
        status_reason = (
            "Layer 3 exclusions exist but are documented through the quality flags table. "
            "Review exclusion counts before downstream tactical profiling."
        )

    if exclusion_percentage > FAIL_EXCLUSION_PERCENTAGE_THRESHOLD:
        status = "fail"
        status_reason = (
            "More than 1% of rows are excluded from Layer 3 shape metric usage. "
            "This may materially threaten Layer 3 usability."
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "status_reason": status_reason,
        "input_file": str(shape_quality.FRAME_TEAM_SHAPE_PATH),
        "flags_file": str(shape_quality.SHAPE_QUALITY_FLAGS_PATH),
        "total_rows_checked": total_rows,
        "rows_with_strict_width_warning": rows_with_strict_width_warning,
        "rows_with_width_tolerance_failure": rows_with_width_tolerance_failure,
        "rows_with_strict_depth_warning": rows_with_strict_depth_warning,
        "rows_with_depth_tolerance_failure": rows_with_depth_tolerance_failure,
        "low_player_count_rows": low_player_count_rows,
        "high_extrapolated_count_rows": high_extrapolated_count_rows,
        "high_extrapolated_rate_rows": high_extrapolated_rate_rows,
        "rows_included_for_layer3": rows_included_for_layer3,
        "rows_excluded_from_layer3": rows_excluded_from_layer3,
        "exclusion_percentage": exclusion_percentage,
        "rows_with_any_warning": rows_with_any_warning,
        "warning_percentage": warning_percentage,
        "thresholds": {
            "team_width_strict_bound": "team_width > pitch_width",
            "team_width_tolerance_failure": "team_width > pitch_width * 1.05",
            "team_depth_strict_bound": "team_depth > pitch_length",
            "team_depth_tolerance_failure": "team_depth > pitch_length * 1.05",
            "low_player_count_flag": f"player_count_used < {shape_quality.LOW_PLAYER_COUNT_THRESHOLD}",
            "high_extrapolated_count_flag": (
                f"extrapolated_player_count >= {shape_quality.HIGH_EXTRAPOLATED_COUNT_THRESHOLD}"
            ),
            "high_extrapolated_rate_flag": (
                f"extrapolated_player_count / player_count_used >= "
                f"{shape_quality.HIGH_EXTRAPOLATED_RATE_THRESHOLD:.2f}"
            ),
            "fail_exclusion_percentage_threshold": FAIL_EXCLUSION_PERCENTAGE_THRESHOLD,
        },
        "layer3_shape_metric_include_rule": (
            "False only when team_width_tolerance_fail_flag, "
            "team_depth_tolerance_fail_flag, or low_player_count_flag is True. "
            "Strict-bound and extrapolation flags are warnings, not hard exclusions."
        ),
    }


def write_shape_quality_summary(summary: dict[str, Any], output_path: Path = SUMMARY_OUTPUT_PATH) -> Path:
    """Write the shape quality summary JSON."""
    common.ensure_directory(output_path.parent)
    return common.write_json(summary, output_path)


def build_and_write_shape_quality_outputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build quality flags, write the CSV, build summary, and write JSON."""
    flags, _ = shape_quality.build_and_write_shape_metric_quality_flags()
    summary = build_shape_quality_summary(flags)
    write_shape_quality_summary(summary)
    return flags, summary


def print_shape_quality_summary(summary: dict[str, Any]) -> None:
    """Print a concise command-line summary."""
    print("Layer 2 v0.2.3 Shape Metric Quality QA")
    print(f"Status: {summary['status']}")
    print(f"Rows checked: {summary['total_rows_checked']:,}")
    print(f"Rows included for Layer 3: {summary['rows_included_for_layer3']:,}")
    print(f"Rows excluded from Layer 3: {summary['rows_excluded_from_layer3']:,}")
    print(f"Exclusion percentage: {summary['exclusion_percentage']:.6f}%")
    print(f"Warning percentage: {summary['warning_percentage']:.6f}%")
    print("")
    print("Flag counts:")
    print(f"- strict width warnings: {summary['rows_with_strict_width_warning']:,}")
    print(f"- width tolerance failures: {summary['rows_with_width_tolerance_failure']:,}")
    print(f"- strict depth warnings: {summary['rows_with_strict_depth_warning']:,}")
    print(f"- depth tolerance failures: {summary['rows_with_depth_tolerance_failure']:,}")
    print(f"- low player count rows: {summary['low_player_count_rows']:,}")
    print(f"- high extrapolated count rows: {summary['high_extrapolated_count_rows']:,}")
    print(f"- high extrapolated rate rows: {summary['high_extrapolated_rate_rows']:,}")
    print("")
    print(f"Flags CSV: {shape_quality.SHAPE_QUALITY_FLAGS_PATH}")
    print(f"Summary JSON: {SUMMARY_OUTPUT_PATH}")
    print(f"Status reason: {summary['status_reason']}")


def main() -> None:
    """Command-line entry point."""
    _, summary = build_and_write_shape_quality_outputs()
    print_shape_quality_summary(summary)


if __name__ == "__main__":
    main()
