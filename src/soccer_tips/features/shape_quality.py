"""
Script: shape_quality.py
Project: Soccer TIPS

Overview:
    Builds non-destructive quality flags for Layer 2 frame-level team shape metrics.

Purpose:
    Layer 2 v0.2.3 adds a Layer 3 readiness filter without changing canonical
    Layer 2 schemas or deleting any frame-level metric rows.

Primary input:
    data/processed/features/frame_team_shape_metrics.csv

Primary output:
    data/processed/features/layer2_shape_metric_quality_flags.csv

Usage from project root:
    $env:PYTHONPATH="src"
    python -m soccer_tips.features.shape_quality
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from soccer_tips import config
from soccer_tips.data import common
from soccer_tips.features import build_layer2


FRAME_TEAM_SHAPE_PATH = build_layer2.FRAME_TEAM_SHAPE_OUTPUT_PATH
SHAPE_QUALITY_FLAGS_PATH = build_layer2.FEATURES_DIR / "layer2_shape_metric_quality_flags.csv"

LOW_PLAYER_COUNT_THRESHOLD = 8
HIGH_EXTRAPOLATED_COUNT_THRESHOLD = 6
HIGH_EXTRAPOLATED_RATE_THRESHOLD = 0.50
GEOMETRY_TOLERANCE_MULTIPLIER = 1.05
DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0

JOIN_KEY_COLUMNS = [
    "provider",
    "match_id",
    "frame",
    "period",
    "team_id",
]

OPTIONAL_CONTEXT_COLUMNS = [
    "phase_index",
    "possession_status_for_team",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
]

REFERENCE_COLUMNS = [
    "pitch_length",
    "pitch_width",
    "team_width",
    "team_depth",
    "player_count_used",
    "detected_player_count",
    "extrapolated_player_count",
]

FLAG_COLUMNS = [
    "team_width_strict_bound_flag",
    "team_width_tolerance_fail_flag",
    "team_depth_strict_bound_flag",
    "team_depth_tolerance_fail_flag",
    "low_player_count_flag",
    "high_extrapolated_count_flag",
    "high_extrapolated_rate_flag",
]

OUTPUT_COLUMNS = (
    JOIN_KEY_COLUMNS
    + OPTIONAL_CONTEXT_COLUMNS
    + REFERENCE_COLUMNS
    + FLAG_COLUMNS
    + [
        "extrapolated_player_rate",
        "quality_warning_count",
        "layer3_shape_metric_include",
        "quality_exclusion_reason",
        "quality_warning_notes",
    ]
)


def _load_match_metadata(match_id: str | int) -> dict[str, Any]:
    """Load processed SkillCorner match metadata if available."""
    metadata_path = config.PROCESSED_SKILLCORNER_DIR / str(match_id) / "match_metadata.json"

    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data[0] if data else {}

    if isinstance(data, dict):
        return data

    return {}


def _build_pitch_dimension_lookup(frame_metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build match_id -> pitch length/width lookup from processed metadata."""
    lookup: dict[str, dict[str, float]] = {}

    if "match_id" not in frame_metrics.columns:
        return lookup

    for match_id in sorted(frame_metrics["match_id"].dropna().astype(str).unique()):
        metadata = _load_match_metadata(match_id)
        pitch_length = metadata.get("pitch_length", DEFAULT_PITCH_LENGTH)
        pitch_width = metadata.get("pitch_width", DEFAULT_PITCH_WIDTH)

        try:
            pitch_length = float(pitch_length)
        except (TypeError, ValueError):
            pitch_length = DEFAULT_PITCH_LENGTH

        try:
            pitch_width = float(pitch_width)
        except (TypeError, ValueError):
            pitch_width = DEFAULT_PITCH_WIDTH

        lookup[str(match_id)] = {
            "pitch_length": pitch_length,
            "pitch_width": pitch_width,
        }

    return lookup


def _add_pitch_dimensions(
    frame_metrics: pd.DataFrame,
    pitch_dimension_lookup: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Add pitch_length and pitch_width columns without changing the input dataframe."""
    df = frame_metrics.copy()

    lookup = pitch_dimension_lookup or _build_pitch_dimension_lookup(df)

    if "pitch_length" not in df.columns:
        df["pitch_length"] = df["match_id"].astype(str).map(
            lambda match_id: lookup.get(match_id, {}).get("pitch_length", DEFAULT_PITCH_LENGTH)
        )

    if "pitch_width" not in df.columns:
        df["pitch_width"] = df["match_id"].astype(str).map(
            lambda match_id: lookup.get(match_id, {}).get("pitch_width", DEFAULT_PITCH_WIDTH)
        )

    df["pitch_length"] = pd.to_numeric(df["pitch_length"], errors="coerce").fillna(DEFAULT_PITCH_LENGTH)
    df["pitch_width"] = pd.to_numeric(df["pitch_width"], errors="coerce").fillna(DEFAULT_PITCH_WIDTH)

    return df


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely compute a rate, returning 0 when denominator is missing or <= 0."""
    numerator_numeric = pd.to_numeric(numerator, errors="coerce").fillna(0)
    denominator_numeric = pd.to_numeric(denominator, errors="coerce").fillna(0)

    return np.where(
        denominator_numeric > 0,
        numerator_numeric / denominator_numeric,
        0.0,
    )


def _build_exclusion_reason(row: pd.Series) -> str:
    """Create semicolon-separated hard exclusion reasons for a quality row."""
    reasons: list[str] = []

    if bool(row["team_width_tolerance_fail_flag"]):
        reasons.append("team_width_tolerance_fail")

    if bool(row["team_depth_tolerance_fail_flag"]):
        reasons.append("team_depth_tolerance_fail")

    if bool(row["low_player_count_flag"]):
        reasons.append("low_player_count")

    return "; ".join(reasons)


def _build_warning_notes(row: pd.Series) -> str:
    """Create semicolon-separated warning notes for a quality row."""
    notes: list[str] = []

    if bool(row["team_width_strict_bound_flag"]):
        notes.append("team_width_exceeds_pitch_width")

    if bool(row["team_width_tolerance_fail_flag"]):
        notes.append("team_width_exceeds_5pct_tolerance")

    if bool(row["team_depth_strict_bound_flag"]):
        notes.append("team_depth_exceeds_pitch_length")

    if bool(row["team_depth_tolerance_fail_flag"]):
        notes.append("team_depth_exceeds_5pct_tolerance")

    if bool(row["low_player_count_flag"]):
        notes.append("player_count_below_threshold")

    if bool(row["high_extrapolated_count_flag"]):
        notes.append("high_extrapolated_player_count")

    if bool(row["high_extrapolated_rate_flag"]):
        notes.append("high_extrapolated_player_rate")

    return "; ".join(notes)


def build_shape_metric_quality_flags(
    frame_metrics: pd.DataFrame,
    pitch_dimension_lookup: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Build Layer 2 shape metric quality flags from frame-level team metrics.

    This function is non-destructive. It does not alter Layer 2 canonical output
    schemas and does not remove or modify any rows from frame_team_shape_metrics.
    """
    missing_required_columns = [
        column for column in JOIN_KEY_COLUMNS + [
            "team_width",
            "team_depth",
            "player_count_used",
            "detected_player_count",
            "extrapolated_player_count",
        ]
        if column not in frame_metrics.columns
    ]

    if missing_required_columns:
        raise ValueError(
            "Cannot build shape metric quality flags. Missing required columns: "
            f"{missing_required_columns}"
        )

    df = _add_pitch_dimensions(frame_metrics, pitch_dimension_lookup=pitch_dimension_lookup)

    for column in OPTIONAL_CONTEXT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    numeric_columns = [
        "team_width",
        "team_depth",
        "player_count_used",
        "detected_player_count",
        "extrapolated_player_count",
        "pitch_length",
        "pitch_width",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["extrapolated_player_rate"] = _safe_rate(
        df["extrapolated_player_count"],
        df["player_count_used"],
    )

    df["team_width_strict_bound_flag"] = df["team_width"] > df["pitch_width"]
    df["team_width_tolerance_fail_flag"] = df["team_width"] > (
        df["pitch_width"] * GEOMETRY_TOLERANCE_MULTIPLIER
    )
    df["team_depth_strict_bound_flag"] = df["team_depth"] > df["pitch_length"]
    df["team_depth_tolerance_fail_flag"] = df["team_depth"] > (
        df["pitch_length"] * GEOMETRY_TOLERANCE_MULTIPLIER
    )

    df["low_player_count_flag"] = df["player_count_used"] < LOW_PLAYER_COUNT_THRESHOLD
    df["high_extrapolated_count_flag"] = (
        df["extrapolated_player_count"] >= HIGH_EXTRAPOLATED_COUNT_THRESHOLD
    )
    df["high_extrapolated_rate_flag"] = (
        df["extrapolated_player_rate"] >= HIGH_EXTRAPOLATED_RATE_THRESHOLD
    )

    df["layer3_shape_metric_include"] = ~(
        df["team_width_tolerance_fail_flag"]
        | df["team_depth_tolerance_fail_flag"]
        | df["low_player_count_flag"]
    )

    df["quality_warning_count"] = df[FLAG_COLUMNS].astype(bool).sum(axis=1).astype(int)
    df["quality_exclusion_reason"] = df.apply(_build_exclusion_reason, axis=1)
    df["quality_warning_notes"] = df.apply(_build_warning_notes, axis=1)

    output = df[OUTPUT_COLUMNS].copy()

    bool_columns = FLAG_COLUMNS + ["layer3_shape_metric_include"]
    for column in bool_columns:
        output[column] = output[column].fillna(False).astype(bool)

    return output


def load_frame_team_shape_metrics(path: Path = FRAME_TEAM_SHAPE_PATH) -> pd.DataFrame:
    """Load Layer 2 frame-level team shape metrics."""
    if not path.exists():
        raise FileNotFoundError(f"Missing frame team shape metrics file: {path}")

    return pd.read_csv(path)


def write_shape_metric_quality_flags(
    flags: pd.DataFrame,
    output_path: Path = SHAPE_QUALITY_FLAGS_PATH,
) -> Path:
    """Write the Layer 2 shape metric quality flags CSV."""
    common.ensure_directory(output_path.parent)
    flags.to_csv(output_path, index=False)
    return output_path


def build_and_write_shape_metric_quality_flags(
    frame_metrics_path: Path = FRAME_TEAM_SHAPE_PATH,
    output_path: Path = SHAPE_QUALITY_FLAGS_PATH,
) -> tuple[pd.DataFrame, Path]:
    """Load frame metrics, build quality flags, and write the output CSV."""
    frame_metrics = load_frame_team_shape_metrics(frame_metrics_path)
    flags = build_shape_metric_quality_flags(frame_metrics)
    written_path = write_shape_metric_quality_flags(flags, output_path)
    return flags, written_path


def main() -> None:
    """Command-line entry point."""
    flags, written_path = build_and_write_shape_metric_quality_flags()

    total_rows = len(flags)
    included_rows = int(flags["layer3_shape_metric_include"].sum())
    excluded_rows = total_rows - included_rows

    print("Layer 2 v0.2.3 shape metric quality flags")
    print(f"Rows written: {total_rows:,}")
    print(f"Layer 3 included rows: {included_rows:,}")
    print(f"Layer 3 excluded rows: {excluded_rows:,}")
    print(f"Output path: {written_path}")


if __name__ == "__main__":
    main()
