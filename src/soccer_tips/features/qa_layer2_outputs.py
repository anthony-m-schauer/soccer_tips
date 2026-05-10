"""
Script: qa_layer2_outputs.py
Project: Soccer TIPS

Overview:
    QA utility for Layer 2 v0.1 feature outputs.

Checks:
    - required output files exist
    - schemas match expected columns
    - no negative width/depth values
    - convex hull area is not computed for invalid player counts
    - phase-linking behavior
    - missing phase context count
    - metric warning counts

Important phase-linking logic:
    SkillCorner phases may contain gaps between phase intervals. Missing phase
    context is acceptable when a tracking frame is outside all actual phase
    intervals. It is not acceptable when a frame falls inside an actual phase
    interval but still fails to link.

Usage from project root:
    $env:PYTHONPATH="src"
    python -m soccer_tips.features.qa_layer2_outputs
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from soccer_tips import config
from soccer_tips.data import common
from soccer_tips.features import build_layer2
from soccer_tips.features import layer2_schemas


QA_SUMMARY_OUTPUT_PATH = build_layer2.FEATURES_DIR / "layer2_v01_qa_summary.json"


REQUIRED_OUTPUTS = {
    "frame_team_shape_metrics": build_layer2.FRAME_TEAM_SHAPE_OUTPUT_PATH,
    "phase_segment_summaries": build_layer2.PHASE_SEGMENT_SUMMARIES_OUTPUT_PATH,
    "match_tactical_profiles": build_layer2.MATCH_TACTICAL_PROFILES_OUTPUT_PATH,
}


EXPECTED_SCHEMAS = {
    "frame_team_shape_metrics": layer2_schemas.FRAME_TEAM_SHAPE_METRICS_COLUMNS,
    "phase_segment_summaries": layer2_schemas.PHASE_SEGMENT_SUMMARY_COLUMNS,
    "match_tactical_profiles": layer2_schemas.MATCH_TACTICAL_PROFILE_COLUMNS,
}


def _inspect_file(path: Path) -> dict[str, Any]:
    """Return file existence and shape information for a CSV output."""
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "shape": None,
            "columns": [],
        }

    df_header = pd.read_csv(path, nrows=0)

    with path.open("r", encoding="utf-8", errors="replace") as file:
        row_count = max(sum(1 for _ in file) - 1, 0)

    return {
        "exists": True,
        "path": str(path),
        "shape": [row_count, len(df_header.columns)],
        "columns": list(df_header.columns),
    }


def _schema_checks(file_info: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build schema checks for required Layer 2 outputs."""
    checks: dict[str, dict[str, Any]] = {}

    for label, info in file_info.items():
        expected_columns = EXPECTED_SCHEMAS[label]
        observed_columns = info.get("columns", [])
        checks[label] = layer2_schemas.schema_check_from_columns(observed_columns, expected_columns)

        if not info.get("exists"):
            checks[label]["status"] = "error"

    return checks


def _metric_checks(frame_metrics: pd.DataFrame | None) -> dict[str, Any]:
    """Run metric sanity checks against frame_team_shape_metrics."""
    if frame_metrics is None:
        return {
            "status": "error",
            "negative_team_width_count": None,
            "negative_team_depth_count": None,
            "invalid_convex_hull_count": None,
            "metric_warning_count": None,
        }

    negative_team_width_count = int((pd.to_numeric(frame_metrics["team_width"], errors="coerce") < 0).sum())
    negative_team_depth_count = int((pd.to_numeric(frame_metrics["team_depth"], errors="coerce") < 0).sum())

    invalid_hull_mask = (
        pd.to_numeric(frame_metrics["player_count_used"], errors="coerce") < 3
    ) & frame_metrics["convex_hull_area"].notna()

    invalid_convex_hull_count = int(invalid_hull_mask.sum())
    metric_warning_count = int(frame_metrics["metric_warning_flag"].fillna(False).astype(bool).sum())

    status = "pass"
    if negative_team_width_count or negative_team_depth_count or invalid_convex_hull_count:
        status = "error"

    return {
        "status": status,
        "negative_team_width_count": negative_team_width_count,
        "negative_team_depth_count": negative_team_depth_count,
        "invalid_convex_hull_count": invalid_convex_hull_count,
        "metric_warning_count": metric_warning_count,
    }


def _skillcorner_phases_path(match_id: str | int) -> Path:
    """Return the Layer 1 processed SkillCorner phases path for a match."""
    return config.PROCESSED_SKILLCORNER_DIR / str(match_id) / "phases_of_play.csv"


def _safe_int(value: Any) -> int | None:
    """Convert a value to int safely."""
    if pd.isna(value):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _phase_coverage_for_match(match_df: pd.DataFrame) -> dict[str, Any]:
    """Check whether missing phase context is due to true link failure or phase gaps."""
    provider = str(match_df["provider"].iloc[0])
    match_id = str(match_df["match_id"].iloc[0])

    if provider != config.SKILLCORNER_PROVIDER:
        return {
            "provider": provider,
            "match_id": match_id,
            "status": "warning",
            "note": "Layer 2 v0.1 phase coverage diagnostics are SkillCorner-specific.",
        }

    phases_path = _skillcorner_phases_path(match_id)

    if not phases_path.exists():
        return {
            "provider": provider,
            "match_id": match_id,
            "status": "error",
            "phase_file_path": str(phases_path),
            "error": "Missing SkillCorner phases_of_play.csv needed for phase-linking QA.",
        }

    phases = pd.read_csv(phases_path)

    required_phase_columns = ["frame_start", "frame_end", "period"]
    missing_phase_columns = [column for column in required_phase_columns if column not in phases.columns]

    if missing_phase_columns:
        return {
            "provider": provider,
            "match_id": match_id,
            "status": "error",
            "phase_file_path": str(phases_path),
            "error": f"SkillCorner phases file missing required columns: {missing_phase_columns}",
        }

    frame_periods = match_df[["frame", "period"]].dropna().drop_duplicates().copy()
    missing_frame_periods = (
        match_df[match_df["phase_index"].isna()][["frame", "period"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    linked_frame_periods = (
        match_df[match_df["phase_index"].notna()][["frame", "period"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    frame_periods["frame"] = frame_periods["frame"].astype(int)
    frame_periods["period"] = frame_periods["period"].astype(int)
    missing_frame_periods["frame"] = missing_frame_periods["frame"].astype(int)
    missing_frame_periods["period"] = missing_frame_periods["period"].astype(int)
    linked_frame_periods["frame"] = linked_frame_periods["frame"].astype(int)
    linked_frame_periods["period"] = linked_frame_periods["period"].astype(int)

    per_period: dict[str, dict[str, Any]] = {}

    total_tracking_frames = 0
    total_linked_frames = 0
    total_missing_frames = 0
    total_phase_interval_covered_tracking_frames = 0
    total_missing_inside_phase_intervals = 0
    total_missing_outside_phase_intervals = 0
    total_missing_in_phase_gaps = 0
    total_missing_before_first_phase = 0
    total_missing_after_last_phase = 0
    invalid_phase_interval_count = 0

    for period in sorted(frame_periods["period"].unique()):
        period_int = int(period)

        period_frames = set(
            frame_periods.loc[frame_periods["period"] == period_int, "frame"].astype(int)
        )
        period_missing_frames = set(
            missing_frame_periods.loc[missing_frame_periods["period"] == period_int, "frame"].astype(int)
        )
        period_linked_frames = set(
            linked_frame_periods.loc[linked_frame_periods["period"] == period_int, "frame"].astype(int)
        )

        period_phases = phases[phases["period"] == period_int].copy()
        period_phases = period_phases.sort_values("frame_start")

        phase_covered_frames: set[int] = set()
        valid_intervals: list[tuple[int, int]] = []

        for _, phase_row in period_phases.iterrows():
            start = _safe_int(phase_row["frame_start"])
            end = _safe_int(phase_row["frame_end"])

            if start is None or end is None or end < start:
                invalid_phase_interval_count += 1
                continue

            valid_intervals.append((start, end))
            phase_covered_frames.update(range(start, end + 1))

        phase_interval_covered_tracking_frames = period_frames.intersection(phase_covered_frames)
        missing_inside_phase_intervals = period_missing_frames.intersection(phase_covered_frames)
        missing_outside_phase_intervals = period_missing_frames.difference(phase_covered_frames)

        if valid_intervals:
            first_phase_start = min(start for start, _ in valid_intervals)
            last_phase_end = max(end for _, end in valid_intervals)

            missing_before_first_phase = {
                frame for frame in missing_outside_phase_intervals if frame < first_phase_start
            }
            missing_after_last_phase = {
                frame for frame in missing_outside_phase_intervals if frame > last_phase_end
            }
            missing_in_phase_gaps = missing_outside_phase_intervals.difference(
                missing_before_first_phase
            ).difference(
                missing_after_last_phase
            )
        else:
            first_phase_start = None
            last_phase_end = None
            missing_before_first_phase = set()
            missing_after_last_phase = set()
            missing_in_phase_gaps = set(missing_outside_phase_intervals)

        total_tracking_frames += len(period_frames)
        total_linked_frames += len(period_linked_frames)
        total_missing_frames += len(period_missing_frames)
        total_phase_interval_covered_tracking_frames += len(phase_interval_covered_tracking_frames)
        total_missing_inside_phase_intervals += len(missing_inside_phase_intervals)
        total_missing_outside_phase_intervals += len(missing_outside_phase_intervals)
        total_missing_in_phase_gaps += len(missing_in_phase_gaps)
        total_missing_before_first_phase += len(missing_before_first_phase)
        total_missing_after_last_phase += len(missing_after_last_phase)

        per_period[str(period_int)] = {
            "tracking_frame_count": len(period_frames),
            "linked_frame_count": len(period_linked_frames),
            "missing_phase_context_frame_count": len(period_missing_frames),
            "phase_interval_covered_tracking_frame_count": len(phase_interval_covered_tracking_frames),
            "missing_inside_phase_interval_frame_count": len(missing_inside_phase_intervals),
            "missing_outside_phase_interval_frame_count": len(missing_outside_phase_intervals),
            "missing_in_phase_gap_frame_count": len(missing_in_phase_gaps),
            "missing_before_first_phase_frame_count": len(missing_before_first_phase),
            "missing_after_last_phase_frame_count": len(missing_after_last_phase),
            "first_phase_start": first_phase_start,
            "last_phase_end": last_phase_end,
        }

    status = "pass"

    if invalid_phase_interval_count > 0:
        status = "error"

    if total_missing_inside_phase_intervals > 0:
        status = "error"

    phase_linking_coverage_rate = total_linked_frames / total_tracking_frames if total_tracking_frames else 0.0
    phase_interval_coverage_rate = (
        total_phase_interval_covered_tracking_frames / total_tracking_frames
        if total_tracking_frames
        else 0.0
    )

    if status == "pass" and total_missing_outside_phase_intervals > 0:
        interpretation = (
            "Phase linker passed. Missing phase context is explained by frames outside actual "
            "SkillCorner phase intervals, including gaps between phase windows."
        )
    elif status == "pass":
        interpretation = "Phase linker passed. All frame/team rows have phase context."
    else:
        interpretation = (
            "Phase linker failed. Some tracking frames inside actual SkillCorner phase intervals "
            "did not receive phase context, or the phase file contains invalid intervals."
        )

    return {
        "provider": provider,
        "match_id": match_id,
        "status": status,
        "phase_file_path": str(phases_path),
        "tracking_frame_count": total_tracking_frames,
        "linked_frame_count": total_linked_frames,
        "missing_phase_context_frame_count": total_missing_frames,
        "phase_linking_coverage_rate": phase_linking_coverage_rate,
        "phase_interval_coverage_rate": phase_interval_coverage_rate,
        "phase_interval_covered_tracking_frame_count": total_phase_interval_covered_tracking_frames,
        "missing_inside_phase_interval_frame_count": total_missing_inside_phase_intervals,
        "missing_outside_phase_interval_frame_count": total_missing_outside_phase_intervals,
        "missing_in_phase_gap_frame_count": total_missing_in_phase_gaps,
        "missing_before_first_phase_frame_count": total_missing_before_first_phase,
        "missing_after_last_phase_frame_count": total_missing_after_last_phase,
        "invalid_phase_interval_count": invalid_phase_interval_count,
        "interpretation": interpretation,
        "per_period": per_period,
    }


def _phase_linking_checks(frame_metrics: pd.DataFrame | None) -> dict[str, Any]:
    """Return phase-linking QA checks with phase-gap-aware interpretation."""
    if frame_metrics is None:
        return {
            "status": "error",
            "row_count": None,
            "linked_phase_row_count": None,
            "missing_phase_context_count": None,
            "phase_linking_coverage_rate": None,
            "missing_inside_phase_interval_frame_count": None,
            "match_checks": {},
        }

    row_count = int(len(frame_metrics))
    linked_count = int(frame_metrics["phase_index"].notna().sum())
    missing_count = row_count - linked_count
    coverage_rate = linked_count / row_count if row_count else 0.0

    if row_count == 0:
        return {
            "status": "error",
            "row_count": 0,
            "linked_phase_row_count": 0,
            "missing_phase_context_count": 0,
            "phase_linking_coverage_rate": 0.0,
            "missing_inside_phase_interval_frame_count": None,
            "match_checks": {},
            "interpretation": "Frame metrics output is empty.",
        }

    match_checks: dict[str, dict[str, Any]] = {}

    for (provider, match_id), match_df in frame_metrics.groupby(["provider", "match_id"], dropna=False):
        key = f"{provider}:{match_id}"
        match_checks[key] = _phase_coverage_for_match(match_df)

    match_statuses = [check.get("status") for check in match_checks.values()]
    total_missing_inside = sum(
        int(check.get("missing_inside_phase_interval_frame_count", 0) or 0)
        for check in match_checks.values()
    )
    total_missing_outside = sum(
        int(check.get("missing_outside_phase_interval_frame_count", 0) or 0)
        for check in match_checks.values()
    )

    status = "pass"

    if any(match_status == "error" for match_status in match_statuses):
        status = "error"
    elif any(match_status == "warning" for match_status in match_statuses):
        status = "warning"

    return {
        "status": status,
        "row_count": row_count,
        "linked_phase_row_count": linked_count,
        "missing_phase_context_count": missing_count,
        "phase_linking_coverage_rate": coverage_rate,
        "missing_inside_phase_interval_frame_count": total_missing_inside,
        "missing_outside_phase_interval_frame_count": total_missing_outside,
        "match_checks": match_checks,
        "interpretation": (
            "Phase linking passes when no missing frame falls inside an actual SkillCorner phase interval. "
            "Low raw phase coverage can be acceptable when caused by gaps between phase intervals."
        ),
    }


def build_layer2_qa_report() -> dict[str, Any]:
    """Build the Layer 2 v0.1 QA report."""
    file_info = {label: _inspect_file(path) for label, path in REQUIRED_OUTPUTS.items()}
    schema_checks = _schema_checks(file_info)

    frame_metrics = None
    if REQUIRED_OUTPUTS["frame_team_shape_metrics"].exists():
        frame_metrics = pd.read_csv(REQUIRED_OUTPUTS["frame_team_shape_metrics"])

    metric_checks = _metric_checks(frame_metrics)
    phase_linking = _phase_linking_checks(frame_metrics)

    warnings: list[str] = []
    errors: list[str] = []

    for label, info in file_info.items():
        if not info["exists"]:
            errors.append(f"Missing required Layer 2 output: {label}")

    for label, check in schema_checks.items():
        if check["status"] == "warning":
            warnings.append(
                f"Schema warning for {label}: "
                f"missing={check['missing_columns']}, extra={check['extra_columns']}"
            )
        elif check["status"] == "error":
            errors.append(f"Schema error for {label}: file missing")

    if metric_checks["status"] == "error":
        errors.append("Metric sanity check failed for width/depth or convex hull validity.")

    if phase_linking["status"] == "warning":
        warnings.append("Phase-linking produced a provider-specific warning. Review phase_linking.match_checks.")
    elif phase_linking["status"] == "error":
        errors.append(
            "Phase-linking check failed. At least one frame inside an actual SkillCorner phase interval "
            "did not receive phase context, or the phase file is invalid/missing."
        )

    status = "pass"
    if warnings:
        status = "pass_with_warnings"
    if errors:
        status = "error"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_dir": str(build_layer2.FEATURES_DIR),
        "status": status,
        "files": file_info,
        "schema_checks": schema_checks,
        "metric_checks": metric_checks,
        "phase_linking": phase_linking,
        "warnings": warnings,
        "errors": errors,
    }


def write_layer2_qa_report(report: dict[str, Any]) -> Path:
    """Write the Layer 2 v0.1 QA report JSON."""
    common.ensure_directory(build_layer2.FEATURES_DIR)
    return common.write_json(report, QA_SUMMARY_OUTPUT_PATH)


def print_layer2_qa_report(report: dict[str, Any]) -> None:
    """Print a concise human-readable Layer 2 QA summary."""
    print("Layer 2 v0.1 QA report")
    print(f"Features directory: {report['features_dir']}")
    print(f"Status: {report['status']}")
    print("")

    for label, info in report["files"].items():
        print(f"- {label}: exists={info['exists']}, shape={info['shape']}")
        schema_status = report["schema_checks"][label]["status"]
        print(f"  schema={schema_status}")

    print("")
    print(f"Metric checks: {report['metric_checks']}")
    print(f"Phase linking: {report['phase_linking']}")

    if report["warnings"]:
        print(f"Warnings: {report['warnings']}")

    if report["errors"]:
        print(f"Errors: {report['errors']}")

    print(f"Structured QA report path: {QA_SUMMARY_OUTPUT_PATH}")


def main() -> None:
    """Command-line entry point."""
    report = build_layer2_qa_report()
    write_layer2_qa_report(report)
    print_layer2_qa_report(report)


if __name__ == "__main__":
    main()