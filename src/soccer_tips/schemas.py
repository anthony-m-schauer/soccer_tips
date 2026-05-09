"""
Script: schemas.py
Project: Soccer TIPS

Overview:
    Canonical schema definitions and simple dataframe schema helpers for Layer 1
    ingestion outputs.

Scope:
    This file defines shared output columns only. It does not parse provider data
    and does not compute tactical metrics.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


# -----------------------------------------------------------------------------
# Canonical processed tracking schemas
# -----------------------------------------------------------------------------

CANONICAL_TRACKING_PLAYER_COLUMNS = [
    "provider",
    "match_id",
    "frame",
    "timestamp",
    "period",
    "team_id",
    "player_id",
    "player_name",
    "position_group",
    "role_name",
    "x_raw",
    "y_raw",
    "x_metric",
    "y_metric",
    "coordinate_system_raw",
    "is_detected",
    "is_extrapolated",
    "source_file",
]

CANONICAL_TRACKING_BALL_COLUMNS = [
    "provider",
    "match_id",
    "frame",
    "timestamp",
    "period",
    "ball_x_raw",
    "ball_y_raw",
    "ball_z_raw",
    "ball_x_metric",
    "ball_y_metric",
    "ball_z_metric",
    "coordinate_system_raw",
    "ball_is_detected",
    "source_file",
]


# -----------------------------------------------------------------------------
# Provider/support schemas
# -----------------------------------------------------------------------------

SKILLCORNER_PLAYER_METADATA_COLUMNS = [
    "match_id",
    "id",
    "team_id",
    "short_name",
    "player_name",
    "number",
    "start_time",
    "end_time",
    "position_group",
    "role_name",
    "trackable_object",
    "team_player_id",
]

VALIDATION_REPORT_COLUMNS = [
    "provider",
    "match_id",
    "status",
    "summary",
    "warnings",
    "errors",
    "checks",
]


# -----------------------------------------------------------------------------
# Dataframe helpers
# -----------------------------------------------------------------------------

def empty_tracking_players_df() -> pd.DataFrame:
    """Return an empty canonical player-tracking dataframe."""
    return pd.DataFrame(columns=CANONICAL_TRACKING_PLAYER_COLUMNS)


def empty_tracking_ball_df() -> pd.DataFrame:
    """Return an empty canonical ball-tracking dataframe."""
    return pd.DataFrame(columns=CANONICAL_TRACKING_BALL_COLUMNS)


def missing_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> list[str]:
    """Return expected columns not present in a dataframe."""
    return [column for column in expected_columns if column not in df.columns]


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
    """Return dataframe with expected columns first, extras afterward."""
    expected_columns = list(expected_columns)
    output = ensure_columns(df, expected_columns)
    extra_columns = [column for column in output.columns if column not in expected_columns]
    return output[expected_columns + extra_columns]


def enforce_tracking_player_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe aligned to the canonical player tracking schema."""
    return enforce_column_order(df, CANONICAL_TRACKING_PLAYER_COLUMNS)


def enforce_tracking_ball_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe aligned to the canonical ball tracking schema."""
    return enforce_column_order(df, CANONICAL_TRACKING_BALL_COLUMNS)
