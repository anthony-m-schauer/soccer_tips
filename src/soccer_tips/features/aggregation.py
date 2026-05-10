"""
Script: aggregation.py
Project: Soccer TIPS

Overview:
    Aggregation utilities for Layer 2 v0.1 feature outputs.

Observation hierarchy:
    frame-level team snapshot -> phase-segment summary -> match-level tactical profile

Scope:
    These aggregations summarize engineered frame metrics. They do not perform
    Layer 3 modeling, clustering, AI interpretation, reporting, or recommendation
    generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from soccer_tips import config
from soccer_tips.features import layer2_schemas


METRIC_COLUMNS = [
    "centroid_x",
    "centroid_y",
    "team_width",
    "team_depth",
    "convex_hull_area",
    "avg_distance_to_centroid",
]

COUNT_MEAN_COLUMNS = [
    "player_count_used",
    "detected_player_count",
    "extrapolated_player_count",
]


def _fps_for_provider(provider: str) -> int:
    """Return provider FPS used for duration estimates."""
    if provider == config.SKILLCORNER_PROVIDER:
        return config.SKILLCORNER_FPS
    if provider == config.METRICA_PROVIDER:
        return config.METRICA_FPS
    return config.SKILLCORNER_FPS


def _safe_std(series: pd.Series) -> float | None:
    """Return std as a float, or None for empty series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.std(ddof=0))


def _summarize_metric_series(series: pd.Series, prefix: str) -> dict[str, float]:
    """Return mean/median/std/min/max summary for one metric series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_std": _safe_std(clean),
        f"{prefix}_min": float(clean.min()),
        f"{prefix}_max": float(clean.max()),
    }


def _warning_frame_count(group_df: pd.DataFrame) -> int:
    """Return count of rows flagged with metric warnings."""
    if "metric_warning_flag" not in group_df.columns:
        return 0

    return int(group_df["metric_warning_flag"].fillna(False).astype(bool).sum())


def aggregate_phase_segment_summaries(frame_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frame-level team metrics into phase-segment summaries."""
    group_columns = [
        "provider",
        "match_id",
        "phase_index",
        "period",
        "team_id",
        "team_in_possession_id",
        "team_in_possession_phase_type",
        "team_out_of_possession_phase_type",
        "possession_status_for_team",
    ]

    missing = [column for column in group_columns if column not in frame_metrics_df.columns]
    if missing:
        raise ValueError(f"Cannot aggregate phase segment summaries; missing columns: {missing}")

    valid_df = frame_metrics_df[frame_metrics_df["phase_index"].notna()].copy()

    if valid_df.empty:
        return layer2_schemas.empty_phase_segment_summaries_df()

    rows: list[dict[str, Any]] = []

    for group_key, group_df in valid_df.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, group_key))
        provider = str(row["provider"])
        frame_count = int(len(group_df))
        warning_frame_count = _warning_frame_count(group_df)

        row.update(
            {
                "frame_count": frame_count,
                "duration_seconds_estimate": frame_count / _fps_for_provider(provider),
                "warning_frame_count": warning_frame_count,
                "warning_frame_rate": warning_frame_count / frame_count if frame_count else np.nan,
            }
        )

        for metric_column in METRIC_COLUMNS:
            row.update(_summarize_metric_series(group_df[metric_column], metric_column))

        for count_column in COUNT_MEAN_COLUMNS:
            row[f"{count_column}_mean"] = float(pd.to_numeric(group_df[count_column], errors="coerce").mean())

        rows.append(row)

    output = pd.DataFrame(rows)
    return layer2_schemas.enforce_phase_segment_summaries_schema(output)


def _profile_row_from_group(
    group_df: pd.DataFrame,
    base_row: dict[str, Any],
    phase_segment_count: int | None = None,
) -> dict[str, Any]:
    """Build one match-profile row from a grouped dataframe."""
    frame_count = int(len(group_df))
    warning_frame_count = _warning_frame_count(group_df)
    provider = str(base_row["provider"])

    row = dict(base_row)
    row.update(
        {
            "frame_count": frame_count,
            "duration_seconds_estimate": frame_count / _fps_for_provider(provider),
            "phase_segment_count": phase_segment_count if phase_segment_count is not None else int(group_df["phase_index"].nunique(dropna=True)),
            "warning_frame_count": warning_frame_count,
            "warning_frame_rate": warning_frame_count / frame_count if frame_count else np.nan,
        }
    )

    for metric_column in METRIC_COLUMNS:
        clean = pd.to_numeric(group_df[metric_column], errors="coerce").dropna()
        row[f"{metric_column}_mean"] = float(clean.mean()) if not clean.empty else np.nan

    for count_column in COUNT_MEAN_COLUMNS:
        row[f"{count_column}_mean"] = float(pd.to_numeric(group_df[count_column], errors="coerce").mean())

    return row


def aggregate_match_tactical_profiles(frame_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frame-level metrics into match-level tactical profile rows.

    The output is intentionally long-form. It includes overall rows,
    possession-status rows, and phase-type rows without creating final cross-match
    tactical fingerprints.
    """
    required_columns = [
        "provider",
        "match_id",
        "team_id",
        "possession_status_for_team",
        "team_in_possession_phase_type",
        "team_out_of_possession_phase_type",
        "phase_index",
    ] + METRIC_COLUMNS + COUNT_MEAN_COLUMNS

    missing = [column for column in required_columns if column not in frame_metrics_df.columns]
    if missing:
        raise ValueError(f"Cannot aggregate match tactical profiles; missing columns: {missing}")

    rows: list[dict[str, Any]] = []

    # Overall rows by team.
    for group_key, group_df in frame_metrics_df.groupby(["provider", "match_id", "team_id"], dropna=False, sort=True):
        provider, match_id, team_id = group_key
        base_row = {
            "provider": provider,
            "match_id": match_id,
            "team_id": team_id,
            "profile_scope": "overall",
            "possession_status_for_team": "all",
            "team_in_possession_phase_type": "all",
            "team_out_of_possession_phase_type": "all",
        }
        rows.append(_profile_row_from_group(group_df, base_row))

    # Possession-status rows by team.
    status_group_columns = ["provider", "match_id", "team_id", "possession_status_for_team"]
    for group_key, group_df in frame_metrics_df.groupby(status_group_columns, dropna=False, sort=True):
        provider, match_id, team_id, possession_status = group_key
        base_row = {
            "provider": provider,
            "match_id": match_id,
            "team_id": team_id,
            "profile_scope": "possession_status",
            "possession_status_for_team": possession_status,
            "team_in_possession_phase_type": "all",
            "team_out_of_possession_phase_type": "all",
        }
        rows.append(_profile_row_from_group(group_df, base_row))

    # In-possession phase type rows.
    in_possession_df = frame_metrics_df[frame_metrics_df["possession_status_for_team"] == "in_possession"].copy()
    in_group_columns = ["provider", "match_id", "team_id", "team_in_possession_phase_type"]
    for group_key, group_df in in_possession_df.groupby(in_group_columns, dropna=False, sort=True):
        provider, match_id, team_id, phase_type = group_key
        base_row = {
            "provider": provider,
            "match_id": match_id,
            "team_id": team_id,
            "profile_scope": "in_possession_phase",
            "possession_status_for_team": "in_possession",
            "team_in_possession_phase_type": phase_type,
            "team_out_of_possession_phase_type": "not_applicable",
        }
        rows.append(_profile_row_from_group(group_df, base_row))

    # Out-of-possession phase type rows.
    out_possession_df = frame_metrics_df[frame_metrics_df["possession_status_for_team"] == "out_of_possession"].copy()
    out_group_columns = ["provider", "match_id", "team_id", "team_out_of_possession_phase_type"]
    for group_key, group_df in out_possession_df.groupby(out_group_columns, dropna=False, sort=True):
        provider, match_id, team_id, phase_type = group_key
        base_row = {
            "provider": provider,
            "match_id": match_id,
            "team_id": team_id,
            "profile_scope": "out_of_possession_phase",
            "possession_status_for_team": "out_of_possession",
            "team_in_possession_phase_type": "not_applicable",
            "team_out_of_possession_phase_type": phase_type,
        }
        rows.append(_profile_row_from_group(group_df, base_row))

    output = pd.DataFrame(rows)
    return layer2_schemas.enforce_match_tactical_profiles_schema(output)
