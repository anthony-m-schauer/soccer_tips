"""
Script: orientation.py
Project: Soccer TIPS

Overview:
    Conservative coordinate preparation utilities for Layer 2 v0.1.

Important boundary:
    Layer 2 v0.1 does not activate attacking-direction normalization. Layer 1
    coordinates are preserved. Analysis-ready coordinates are copied into separate
    columns so future orientation work can be added without destroying provider-
    native or Layer 1 metric coordinates.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


ANALYSIS_X_COLUMN = "x_analysis"
ANALYSIS_Y_COLUMN = "y_analysis"
ORIENTATION_APPLIED_COLUMN = "orientation_normalization_applied"
ORIENTATION_METHOD_COLUMN = "orientation_method"


def add_analysis_coordinates(
    tracking_df: pd.DataFrame,
    x_metric_column: str = "x_metric",
    y_metric_column: str = "y_metric",
) -> pd.DataFrame:
    """Add analysis-ready coordinates while preserving existing Layer 1 coordinates.

    For v0.1, this function deliberately copies x_metric/y_metric into
    x_analysis/y_analysis. It does not flip coordinates by attacking direction.
    """
    required_columns = [x_metric_column, y_metric_column]
    missing = [column for column in required_columns if column not in tracking_df.columns]

    if missing:
        raise ValueError(f"Cannot add analysis coordinates; missing columns: {missing}")

    output = tracking_df.copy()
    output[ANALYSIS_X_COLUMN] = output[x_metric_column]
    output[ANALYSIS_Y_COLUMN] = output[y_metric_column]
    output[ORIENTATION_APPLIED_COLUMN] = False
    output[ORIENTATION_METHOD_COLUMN] = "metric_coordinates_no_attacking_direction_normalization_v0_1"

    return output


def can_normalize_attacking_direction(match_metadata: dict[str, Any] | None) -> bool:
    """Return whether attacking-direction normalization is safe for this version.

    The placeholder is intentionally conservative. Even when match metadata
    contains home_team_side, Layer 2 v0.1 leaves direction normalization inactive
    until the Lead Developer explicitly approves the strategy.
    """
    return False


def normalize_attacking_direction_placeholder(
    tracking_df: pd.DataFrame,
    match_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Placeholder for future attacking-direction normalization.

    This function currently returns analysis-coordinate copies only. It exists so
    future code has a stable extension point without rewriting team shape logic.
    """
    return add_analysis_coordinates(tracking_df)
