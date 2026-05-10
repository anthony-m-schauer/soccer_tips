"""
Script: layer2_schemas.py
Project: Soccer TIPS

Overview:
    Schema definitions and dataframe helpers for Layer 2 v0.1 feature outputs.

Scope:
    This module defines feature-engineering output schemas only. It does not
    compute tactical models, make final tactical identity claims, or generate
    dashboards/reports.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


# -----------------------------------------------------------------------------
# Output schemas
# -----------------------------------------------------------------------------

FRAME_TEAM_SHAPE_METRICS_COLUMNS = [
    "provider",
    "match_id",
    "frame",
    "timestamp",
    "period",
    "team_id",
    "phase_index",
    "team_in_possession_id",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
    "possession_status_for_team",
    "centroid_x",
    "centroid_y",
    "team_width",
    "team_depth",
    "convex_hull_area",
    "avg_distance_to_centroid",
    "player_count_used",
    "detected_player_count",
    "extrapolated_player_count",
    "metric_warning_flag",
]


PHASE_SEGMENT_SUMMARY_COLUMNS = [
    "provider",
    "match_id",
    "phase_index",
    "period",
    "team_id",
    "team_in_possession_id",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
    "possession_status_for_team",
    "frame_count",
    "duration_seconds_estimate",
    "warning_frame_count",
    "warning_frame_rate",
    "centroid_x_mean",
    "centroid_x_median",
    "centroid_x_std",
    "centroid_x_min",
    "centroid_x_max",
    "centroid_y_mean",
    "centroid_y_median",
    "centroid_y_std",
    "centroid_y_min",
    "centroid_y_max",
    "team_width_mean",
    "team_width_median",
    "team_width_std",
    "team_width_min",
    "team_width_max",
    "team_depth_mean",
    "team_depth_median",
    "team_depth_std",
    "team_depth_min",
    "team_depth_max",
    "convex_hull_area_mean",
    "convex_hull_area_median",
    "convex_hull_area_std",
    "convex_hull_area_min",
    "convex_hull_area_max",
    "avg_distance_to_centroid_mean",
    "avg_distance_to_centroid_median",
    "avg_distance_to_centroid_std",
    "avg_distance_to_centroid_min",
    "avg_distance_to_centroid_max",
    "player_count_used_mean",
    "detected_player_count_mean",
    "extrapolated_player_count_mean",
]


MATCH_TACTICAL_PROFILE_COLUMNS = [
    "provider",
    "match_id",
    "team_id",
    "profile_scope",
    "possession_status_for_team",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
    "frame_count",
    "duration_seconds_estimate",
    "phase_segment_count",
    "warning_frame_count",
    "warning_frame_rate",
    "centroid_x_mean",
    "centroid_y_mean",
    "team_width_mean",
    "team_depth_mean",
    "convex_hull_area_mean",
    "avg_distance_to_centroid_mean",
    "player_count_used_mean",
    "detected_player_count_mean",
    "extrapolated_player_count_mean",
]


LAYER2_QA_SUMMARY_COLUMNS = [
    "generated_at_utc",
    "features_dir",
    "status",
    "files",
    "schema_checks",
    "metric_checks",
    "phase_linking",
    "warnings",
    "errors",
]


# -----------------------------------------------------------------------------
# Dataframe helpers
# -----------------------------------------------------------------------------

def empty_frame_team_shape_metrics_df() -> pd.DataFrame:
    """Return an empty frame-level team shape metrics dataframe."""
    return pd.DataFrame(columns=FRAME_TEAM_SHAPE_METRICS_COLUMNS)


def empty_phase_segment_summaries_df() -> pd.DataFrame:
    """Return an empty phase segment summary dataframe."""
    return pd.DataFrame(columns=PHASE_SEGMENT_SUMMARY_COLUMNS)


def empty_match_tactical_profiles_df() -> pd.DataFrame:
    """Return an empty match tactical profile dataframe."""
    return pd.DataFrame(columns=MATCH_TACTICAL_PROFILE_COLUMNS)


def missing_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> list[str]:
    """Return expected columns that are missing from a dataframe."""
    return [column for column in expected_columns if column not in df.columns]


def extra_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> list[str]:
    """Return dataframe columns that are not part of the expected schema."""
    expected = set(expected_columns)
    return [column for column in df.columns if column not in expected]


def has_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> bool:
    """Return True when all expected columns are present."""
    return len(missing_columns(df, expected_columns)) == 0


def ensure_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> pd.DataFrame:
    """Return a dataframe with all expected columns present.

    Missing columns are created with pandas.NA. Existing columns are preserved.
    """
    output = df.copy()
    for column in expected_columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output


def enforce_column_order(df: pd.DataFrame, expected_columns: Iterable[str]) -> pd.DataFrame:
    """Return dataframe with expected columns first and extras afterward."""
    expected_columns = list(expected_columns)
    output = ensure_columns(df, expected_columns)
    extras = [column for column in output.columns if column not in expected_columns]
    return output[expected_columns + extras]


def enforce_frame_team_shape_metrics_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe aligned to the frame-team shape metrics schema."""
    return enforce_column_order(df, FRAME_TEAM_SHAPE_METRICS_COLUMNS)


def enforce_phase_segment_summaries_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe aligned to the phase segment summary schema."""
    return enforce_column_order(df, PHASE_SEGMENT_SUMMARY_COLUMNS)


def enforce_match_tactical_profiles_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe aligned to the match tactical profile schema."""
    return enforce_column_order(df, MATCH_TACTICAL_PROFILE_COLUMNS)


def schema_check_from_columns(column_names: Iterable[str], expected_columns: Iterable[str]) -> dict[str, object]:
    """Return schema-compliance details from observed column names."""
    observed = list(column_names)
    expected = list(expected_columns)
    missing = [column for column in expected if column not in observed]
    extras = [column for column in observed if column not in expected]
    ordered_exact_match = observed == expected

    return {
        "expected_columns": expected,
        "observed_columns": observed,
        "missing_columns": missing,
        "extra_columns": extras,
        "has_all_expected_columns": len(missing) == 0,
        "ordered_exact_match": ordered_exact_match,
        "status": "pass" if not missing and not extras and ordered_exact_match else "warning",
    }
