"""
Script: football_sanity.py
Project: Soccer TIPS

Overview:
    Layer 2 v0.2 football sanity validation utilities.

Purpose:
    Validate that accepted Layer 2 v0.1 shape metrics behave plausibly as
    football tracking-derived metrics before the project expands into deeper
    tactical metric design, Layer 3 modeling, AI interpretation, dashboards,
    reports, or recommendations.

Scope:
    This module does not add new tactical metrics. It summarizes and validates
    existing Layer 2 v0.1 outputs:
        - frame_team_shape_metrics.csv
        - phase_segment_summaries.csv
        - match_tactical_profiles.csv

Expected outputs:
    data/processed/features/layer2_v02_football_sanity_summary.csv
    data/processed/features/layer2_v02_football_sanity_summary.json
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from soccer_tips import config
from soccer_tips.data import common
from soccer_tips.features import build_layer2


FEATURES_DIR = build_layer2.FEATURES_DIR
FOOTBALL_SANITY_CSV_PATH = FEATURES_DIR / "layer2_v02_football_sanity_summary.csv"
FOOTBALL_SANITY_JSON_PATH = FEATURES_DIR / "layer2_v02_football_sanity_summary.json"

PHASE_LABELS_OF_INTEREST = [
    "build_up",
    "create",
    "finish",
    "transition",
    "quick_break",
    "low_block",
    "medium_block",
    "high_block",
    "defending_transition",
]

SHAPE_METRICS = [
    "team_width",
    "team_depth",
    "convex_hull_area",
    "avg_distance_to_centroid",
    "centroid_x",
    "centroid_y",
]

REASONABLENESS_TOLERANCE_METERS = 5.0
CONVEX_HULL_AREA_TOLERANCE_RATE = 1.05
WARNING_OUTLIER_RATE = 0.01
FAIL_OUTLIER_RATE = 0.05
MIN_REASONABLE_PLAYER_COUNT = 8
MAX_REASONABLE_PLAYER_COUNT = 11


@dataclass(frozen=True)
class SanityInputTables:
    """Container for Layer 2 output tables used by sanity validation."""

    frame_team_shape_metrics: pd.DataFrame
    phase_segment_summaries: pd.DataFrame
    match_tactical_profiles: pd.DataFrame


def load_layer2_feature_outputs(features_dir: Path = FEATURES_DIR) -> SanityInputTables:
    """Load the current Layer 2 v0.1 feature output tables."""
    frame_metrics_path = common.require_file(
        features_dir / "frame_team_shape_metrics.csv",
        label="Layer 2 frame team shape metrics",
    )
    phase_summaries_path = common.require_file(
        features_dir / "phase_segment_summaries.csv",
        label="Layer 2 phase segment summaries",
    )
    match_profiles_path = common.require_file(
        features_dir / "match_tactical_profiles.csv",
        label="Layer 2 match tactical profiles",
    )

    return SanityInputTables(
        frame_team_shape_metrics=common.read_csv(frame_metrics_path),
        phase_segment_summaries=common.read_csv(phase_summaries_path),
        match_tactical_profiles=common.read_csv(match_profiles_path),
    )


def phase_label_for_team(row: pd.Series | dict[str, Any]) -> str | None:
    """Return the tactical phase label that applies to a team-frame row.

    For a team in possession, this returns team_in_possession_phase_type.
    For a team out of possession, this returns team_out_of_possession_phase_type.
    For rows with unknown or missing phase context, this returns None.
    """
    status = row.get("possession_status_for_team")

    if status == "in_possession":
        value = row.get("team_in_possession_phase_type")
    elif status == "out_of_possession":
        value = row.get("team_out_of_possession_phase_type")
    else:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()
    return text or None


def add_team_phase_label(frame_metrics: pd.DataFrame) -> pd.DataFrame:
    """Add validation-only team_phase_label to frame metrics."""
    output = frame_metrics.copy()
    output["team_phase_label"] = output.apply(phase_label_for_team, axis=1)
    return output


def summarize_metric_distributions_by_phase(frame_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize existing shape metrics by team-level phase label."""
    working = add_team_phase_label(frame_metrics)
    working = working[working["team_phase_label"].isin(PHASE_LABELS_OF_INTEREST)].copy()

    rows: list[dict[str, Any]] = []

    for phase_label in PHASE_LABELS_OF_INTEREST:
        phase_df = working[working["team_phase_label"] == phase_label]

        for metric in SHAPE_METRICS:
            if metric not in phase_df.columns:
                rows.append(
                    {
                        "summary_type": "metric_distribution_by_phase",
                        "phase_label": phase_label,
                        "metric": metric,
                        "row_count": 0,
                        "match_count": 0,
                        "team_count": 0,
                        "mean": np.nan,
                        "median": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                        "p05": np.nan,
                        "p95": np.nan,
                    }
                )
                continue

            values = pd.to_numeric(phase_df[metric], errors="coerce").dropna()

            rows.append(
                {
                    "summary_type": "metric_distribution_by_phase",
                    "phase_label": phase_label,
                    "metric": metric,
                    "row_count": int(len(values)),
                    "match_count": int(phase_df["match_id"].nunique()) if not phase_df.empty else 0,
                    "team_count": int(phase_df["team_id"].nunique()) if not phase_df.empty else 0,
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "std": float(values.std(ddof=0)) if not values.empty else np.nan,
                    "min": float(values.min()) if not values.empty else np.nan,
                    "max": float(values.max()) if not values.empty else np.nan,
                    "p05": float(values.quantile(0.05)) if not values.empty else np.nan,
                    "p95": float(values.quantile(0.95)) if not values.empty else np.nan,
                }
            )

    return pd.DataFrame(rows)


def _match_pitch_dimensions(match_id: str | int) -> tuple[float, float]:
    """Return pitch length/width from processed SkillCorner metadata when available."""
    metadata_path = config.PROCESSED_SKILLCORNER_DIR / str(match_id) / "match_metadata.json"

    if metadata_path.exists():
        metadata = common.read_json(metadata_path)
        pitch_length = metadata.get("pitch_length", config.DEFAULT_PITCH_LENGTH)
        pitch_width = metadata.get("pitch_width", config.DEFAULT_PITCH_WIDTH)
        return float(pitch_length), float(pitch_width)

    return float(config.DEFAULT_PITCH_LENGTH), float(config.DEFAULT_PITCH_WIDTH)


def _check_outlier_rate(count: int, total: int) -> str:
    """Classify an outlier rate as pass/warning/fail."""
    if total <= 0:
        return "warning"

    rate = count / total

    if rate > FAIL_OUTLIER_RATE:
        return "fail"

    if rate > WARNING_OUTLIER_RATE:
        return "warning"

    return "pass"


def run_reasonableness_checks(frame_metrics: pd.DataFrame) -> dict[str, Any]:
    """Run simple football reasonableness checks against existing metrics.

    These checks are intentionally conservative. They do not hard-fail every
    outlier, but they do flag rates of implausible values that deserve review.
    """
    if frame_metrics.empty:
        return {
            "status": "fail",
            "checks": {},
            "notes": ["frame_team_shape_metrics.csv is empty."],
        }

    working = frame_metrics.copy()
    working["match_id"] = working["match_id"].astype(str)

    pitch_dims = {
        match_id: _match_pitch_dimensions(match_id)
        for match_id in sorted(working["match_id"].dropna().unique())
    }

    checks: dict[str, Any] = {}

    total_rows = int(len(working))

    width_outlier_count = 0
    depth_outlier_count = 0
    hull_area_outlier_count = 0

    for match_id, match_df in working.groupby("match_id", dropna=False):
        pitch_length, pitch_width = pitch_dims.get(str(match_id), (config.DEFAULT_PITCH_LENGTH, config.DEFAULT_PITCH_WIDTH))
        pitch_area = pitch_length * pitch_width

        width_limit = pitch_width + REASONABLENESS_TOLERANCE_METERS
        depth_limit = pitch_length + REASONABLENESS_TOLERANCE_METERS
        hull_limit = pitch_area * CONVEX_HULL_AREA_TOLERANCE_RATE

        width_values = pd.to_numeric(match_df["team_width"], errors="coerce")
        depth_values = pd.to_numeric(match_df["team_depth"], errors="coerce")
        hull_values = pd.to_numeric(match_df["convex_hull_area"], errors="coerce")

        width_outlier_count += int((width_values > width_limit).sum())
        depth_outlier_count += int((depth_values > depth_limit).sum())
        hull_area_outlier_count += int((hull_values > hull_limit).sum())

    negative_width_count = int((pd.to_numeric(working["team_width"], errors="coerce") < 0).sum())
    negative_depth_count = int((pd.to_numeric(working["team_depth"], errors="coerce") < 0).sum())
    negative_hull_count = int((pd.to_numeric(working["convex_hull_area"], errors="coerce") < 0).sum())

    player_count = pd.to_numeric(working["player_count_used"], errors="coerce")
    low_player_count = int((player_count < MIN_REASONABLE_PLAYER_COUNT).sum())
    high_player_count = int((player_count > MAX_REASONABLE_PLAYER_COUNT).sum())

    detected_count = pd.to_numeric(working["detected_player_count"], errors="coerce")
    extrapolated_count = pd.to_numeric(working["extrapolated_player_count"], errors="coerce")

    linked_phase_rows = int(working["phase_index"].notna().sum()) if "phase_index" in working.columns else 0
    missing_phase_rows = total_rows - linked_phase_rows

    checks["team_width_pitch_bound"] = {
        "status": _check_outlier_rate(width_outlier_count, total_rows),
        "outlier_count": width_outlier_count,
        "row_count": total_rows,
        "outlier_rate": width_outlier_count / total_rows if total_rows else None,
        "rule": f"team_width should usually be <= pitch_width + {REASONABLENESS_TOLERANCE_METERS}m.",
    }
    checks["team_depth_pitch_bound"] = {
        "status": _check_outlier_rate(depth_outlier_count, total_rows),
        "outlier_count": depth_outlier_count,
        "row_count": total_rows,
        "outlier_rate": depth_outlier_count / total_rows if total_rows else None,
        "rule": f"team_depth should usually be <= pitch_length + {REASONABLENESS_TOLERANCE_METERS}m.",
    }
    checks["convex_hull_pitch_area_bound"] = {
        "status": _check_outlier_rate(hull_area_outlier_count, total_rows),
        "outlier_count": hull_area_outlier_count,
        "row_count": total_rows,
        "outlier_rate": hull_area_outlier_count / total_rows if total_rows else None,
        "rule": f"convex_hull_area should usually be <= pitch_area * {CONVEX_HULL_AREA_TOLERANCE_RATE}.",
    }
    checks["negative_metric_values"] = {
        "status": "fail" if any([negative_width_count, negative_depth_count, negative_hull_count]) else "pass",
        "negative_width_count": negative_width_count,
        "negative_depth_count": negative_depth_count,
        "negative_hull_count": negative_hull_count,
    }
    checks["player_count_used"] = {
        "status": _check_outlier_rate(low_player_count + high_player_count, total_rows),
        "low_player_count_rows": low_player_count,
        "high_player_count_rows": high_player_count,
        "row_count": total_rows,
        "expected_range": [MIN_REASONABLE_PLAYER_COUNT, MAX_REASONABLE_PLAYER_COUNT],
    }
    checks["detection_quality"] = {
        "status": "pass",
        "detected_player_count_mean": float(detected_count.mean()) if not detected_count.empty else None,
        "detected_player_count_min": float(detected_count.min()) if not detected_count.empty else None,
        "detected_player_count_max": float(detected_count.max()) if not detected_count.empty else None,
        "extrapolated_player_count_mean": float(extrapolated_count.mean()) if not extrapolated_count.empty else None,
        "extrapolated_player_count_min": float(extrapolated_count.min()) if not extrapolated_count.empty else None,
        "extrapolated_player_count_max": float(extrapolated_count.max()) if not extrapolated_count.empty else None,
    }
    checks["phase_context_presence"] = {
        "status": "pass" if linked_phase_rows > 0 else "fail",
        "row_count": total_rows,
        "linked_phase_rows": linked_phase_rows,
        "missing_phase_rows": missing_phase_rows,
        "linked_phase_row_rate": linked_phase_rows / total_rows if total_rows else None,
        "note": "Missing phase rows can be acceptable when they fall outside actual SkillCorner phase intervals. Use Layer 2 QA for phase-gap-aware diagnostics.",
    }

    statuses = [check["status"] for check in checks.values()]
    overall_status = "pass"
    if "warning" in statuses:
        overall_status = "warning"
    if "fail" in statuses:
        overall_status = "fail"

    notes = [
        "Reasonableness checks are conservative and intended to surface football-sanity issues, not replace visual inspection.",
        "Because attacking-direction normalization is not active, centroid_x should not be used for strong directional tactical claims yet.",
    ]

    return {
        "status": overall_status,
        "checks": checks,
        "notes": notes,
    }


def summarize_phase_comparisons(distribution_summary: pd.DataFrame) -> dict[str, Any]:
    """Create structured phase-comparison notes from distribution summaries."""
    phase_counts = (
        distribution_summary[distribution_summary["metric"] == "team_width"]
        .set_index("phase_label")["row_count"]
        .to_dict()
        if not distribution_summary.empty
        else {}
    )

    def _metric_mean(phase: str, metric: str) -> float | None:
        mask = (distribution_summary["phase_label"] == phase) & (distribution_summary["metric"] == metric)
        if not mask.any():
            return None
        value = distribution_summary.loc[mask, "mean"].iloc[0]
        return None if pd.isna(value) else float(value)

    defensive_phases = ["low_block", "medium_block", "high_block"]
    possession_phases = ["build_up", "create", "finish"]
    transition_phases = ["transition", "quick_break", "defending_transition"]

    missing_phases = [phase for phase in PHASE_LABELS_OF_INTEREST if int(phase_counts.get(phase, 0)) == 0]

    transition_std_values = []
    stable_std_values = []

    for phase in transition_phases:
        for metric in ["team_width", "team_depth", "convex_hull_area", "avg_distance_to_centroid"]:
            mask = (distribution_summary["phase_label"] == phase) & (distribution_summary["metric"] == metric)
            if mask.any() and not pd.isna(distribution_summary.loc[mask, "std"].iloc[0]):
                transition_std_values.append(float(distribution_summary.loc[mask, "std"].iloc[0]))

    for phase in defensive_phases + possession_phases:
        for metric in ["team_width", "team_depth", "convex_hull_area", "avg_distance_to_centroid"]:
            mask = (distribution_summary["phase_label"] == phase) & (distribution_summary["metric"] == metric)
            if mask.any() and not pd.isna(distribution_summary.loc[mask, "std"].iloc[0]):
                stable_std_values.append(float(distribution_summary.loc[mask, "std"].iloc[0]))

    transition_volatility_note = "Not enough data to compare transition volatility."
    if transition_std_values and stable_std_values:
        transition_std_mean = float(np.mean(transition_std_values))
        stable_std_mean = float(np.mean(stable_std_values))
        if transition_std_mean > stable_std_mean:
            transition_volatility_note = (
                "Transition-related phases show higher average metric variability than the selected possession/block phases. "
                "This is a plausible signal, but not yet a transition-stability model."
            )
        else:
            transition_volatility_note = (
                "Transition-related phases do not show higher average metric variability in this coarse check. "
                "This does not prove the metrics are wrong; it means deeper transition validation is needed later."
            )

    notes = {
        "block_phase_behavior": (
            "Block phases are present for validation. Compare low_block, medium_block, and high_block distributions "
            "and plots before making tactical claims."
            if any(int(phase_counts.get(phase, 0)) > 0 for phase in defensive_phases)
            else "Block phase labels were not found in the current Layer 2 output."
        ),
        "possession_phase_behavior": (
            "Possession phases are present for validation. build_up/create/finish spacing behavior should be interpreted "
            "through plots and distribution summaries, not directional centroid_x claims."
            if any(int(phase_counts.get(phase, 0)) > 0 for phase in possession_phases)
            else "Possession phase labels were not found in the current Layer 2 output."
        ),
        "transition_phase_behavior": transition_volatility_note,
        "centroid_caution": (
            "centroid_x and centroid_y can validate coordinate plausibility, but centroid_x should not be used for strong "
            "attacking-direction claims until orientation normalization is validated and activated."
        ),
        "missing_phases": missing_phases,
        "selected_phase_metric_means": {
            phase: {
                metric: _metric_mean(phase, metric)
                for metric in ["team_width", "team_depth", "convex_hull_area", "avg_distance_to_centroid"]
            }
            for phase in PHASE_LABELS_OF_INTEREST
        },
    }

    return notes


def assess_metric_trust(reasonableness: dict[str, Any]) -> dict[str, list[str]]:
    """Return coarse trust/caution lists for current metrics."""
    checks = reasonableness.get("checks", {})

    safe: list[str] = []
    caution: list[str] = []

    if checks.get("team_width_pitch_bound", {}).get("status") == "pass":
        safe.append("team_width")
    else:
        caution.append("team_width")

    if checks.get("team_depth_pitch_bound", {}).get("status") == "pass":
        safe.append("team_depth")
    else:
        caution.append("team_depth")

    if checks.get("convex_hull_pitch_area_bound", {}).get("status") == "pass":
        safe.append("convex_hull_area")
    else:
        caution.append("convex_hull_area")

    safe.append("avg_distance_to_centroid")
    safe.append("player_count_used")
    safe.append("detected_player_count")
    safe.append("extrapolated_player_count")

    caution.extend(["centroid_x", "centroid_y"])

    return {
        "appears_safe_for_descriptive_profiling_after_visual_review": sorted(set(safe)),
        "needs_caution_or_context": sorted(set(caution)),
    }


def build_summary_report(
    tables: SanityInputTables,
    visual_outputs: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build CSV and JSON-ready football sanity summaries."""
    frame_metrics = tables.frame_team_shape_metrics
    distribution_summary = summarize_metric_distributions_by_phase(frame_metrics)
    reasonableness = run_reasonableness_checks(frame_metrics)
    phase_comparison_notes = summarize_phase_comparisons(distribution_summary)
    metric_trust = assess_metric_trust(reasonableness)

    visual_outputs = visual_outputs or []
    visual_warnings = [item for item in visual_outputs if item.get("status") != "created"]

    warnings: list[str] = []
    failures: list[str] = []

    for check_name, check in reasonableness.get("checks", {}).items():
        if check.get("status") == "warning":
            warnings.append(f"Reasonableness warning: {check_name}")
        elif check.get("status") == "fail":
            failures.append(f"Reasonableness failure: {check_name}")

    if visual_warnings:
        warnings.append("Some requested validation plots could not be created. Review visual_checks_created.")

    if phase_comparison_notes.get("missing_phases"):
        warnings.append(f"Some target phases were not present: {phase_comparison_notes['missing_phases']}")

    status = "pass"
    if warnings:
        status = "warning"
    if failures:
        status = "fail"

    matches_inspected = sorted(str(value) for value in frame_metrics["match_id"].dropna().unique())
    phases_inspected = sorted(
        str(value)
        for value in add_team_phase_label(frame_metrics)["team_phase_label"].dropna().unique()
        if str(value) in PHASE_LABELS_OF_INTEREST
    )

    json_report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "matches_inspected": matches_inspected,
        "frame_team_metric_rows_inspected": int(len(frame_metrics)),
        "phase_segment_summary_rows_inspected": int(len(tables.phase_segment_summaries)),
        "match_tactical_profile_rows_inspected": int(len(tables.match_tactical_profiles)),
        "frames_plotted": [item for item in visual_outputs if item.get("plot_type") == "shape_frame" and item.get("status") == "created"],
        "phases_inspected": phases_inspected,
        "metric_distribution_checks": distribution_summary.to_dict(orient="records"),
        "reasonableness_checks": reasonableness,
        "visual_checks_created": visual_outputs,
        "warnings": warnings,
        "failures": failures,
        "interpretation_notes": phase_comparison_notes,
        "metric_trust_assessment": metric_trust,
        "recommended_next_step": (
            "If status is pass or warning with understood limitations, send this football sanity output to the Lead Developer "
            "for review before expanding Layer 2 metrics or moving toward Layer 3 descriptive profiling."
        ),
        "scope_guard": (
            "This validation does not authorize Layer 3 modeling, clustering, AI interpretation, dashboards, reports, "
            "recommendations, advanced compactness interpretation, transition stability modeling, or final tactical claims."
        ),
    }

    return distribution_summary, json_report


def write_football_sanity_outputs(
    distribution_summary: pd.DataFrame,
    json_report: dict[str, Any],
) -> dict[str, Path]:
    """Write Layer 2 v0.2 football sanity outputs."""
    common.ensure_directory(FEATURES_DIR)

    csv_path = common.write_csv(distribution_summary, FOOTBALL_SANITY_CSV_PATH)
    json_path = common.write_json(json_report, FOOTBALL_SANITY_JSON_PATH)

    return {
        "csv": csv_path,
        "json": json_path,
    }


def run_football_sanity_validation(visual_outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run data-only football sanity validation and write outputs."""
    tables = load_layer2_feature_outputs()
    distribution_summary, json_report = build_summary_report(tables, visual_outputs=visual_outputs)
    output_paths = write_football_sanity_outputs(distribution_summary, json_report)
    json_report["output_paths"] = {key: str(path) for key, path in output_paths.items()}
    common.write_json(json_report, FOOTBALL_SANITY_JSON_PATH)
    return json_report
