"""
Script: normalize.py
Project: Soccer TIPS

Overview:
    Coordinate normalization utilities for Layer 1 ingestion.

Scope boundaries:
    - Preserve provider-native raw coordinates.
    - Generate standardized metric coordinates separately.
    - Do not perform attack-direction or period orientation normalization in v0.1.
"""

from __future__ import annotations

import pandas as pd

from soccer_tips import config


# -----------------------------------------------------------------------------
# Scalar coordinate conversion helpers
# -----------------------------------------------------------------------------

def skillcorner_to_metric(x_raw: float | int | None, y_raw: float | int | None) -> tuple[float | None, float | None]:
    """Return SkillCorner coordinates as metric-centered coordinates.

    SkillCorner coordinates are already metric and centered at midfield, so this
    function intentionally returns the raw values as the metric values.
    """
    return x_raw, y_raw


def metrica_normalized_to_metric(
    x_raw: float | int | None,
    y_raw: float | int | None,
    pitch_length: float = config.DEFAULT_PITCH_LENGTH,
    pitch_width: float = config.DEFAULT_PITCH_WIDTH,
) -> tuple[float | None, float | None]:
    """Convert Metrica normalized [0, 1] coordinates to centered metric coordinates.

    Formula:
        x_metric = x_raw * pitch_length - pitch_length / 2
        y_metric = y_raw * pitch_width - pitch_width / 2

    Notes:
        Metrica can contain small values outside [0, 1]. Those values are not
        clipped here because preserving provider behavior is useful for debugging.
    """
    if pd.isna(x_raw) or pd.isna(y_raw):
        return None, None
    return (float(x_raw) * pitch_length) - (pitch_length / 2), (float(y_raw) * pitch_width) - (pitch_width / 2)


# -----------------------------------------------------------------------------
# Vectorized dataframe helpers
# -----------------------------------------------------------------------------

def add_skillcorner_metric_columns(
    df: pd.DataFrame,
    x_raw_col: str = "x_raw",
    y_raw_col: str = "y_raw",
    x_metric_col: str = "x_metric",
    y_metric_col: str = "y_metric",
) -> pd.DataFrame:
    """Add metric coordinate columns to a SkillCorner dataframe."""
    output = df.copy()
    output[x_metric_col] = output[x_raw_col]
    output[y_metric_col] = output[y_raw_col]
    return output


def add_metrica_metric_columns(
    df: pd.DataFrame,
    x_raw_col: str = "x_raw",
    y_raw_col: str = "y_raw",
    x_metric_col: str = "x_metric",
    y_metric_col: str = "y_metric",
    pitch_length: float = config.DEFAULT_PITCH_LENGTH,
    pitch_width: float = config.DEFAULT_PITCH_WIDTH,
) -> pd.DataFrame:
    """Add metric coordinate columns to a Metrica dataframe."""
    output = df.copy()
    output[x_metric_col] = (output[x_raw_col].astype(float) * pitch_length) - (pitch_length / 2)
    output[y_metric_col] = (output[y_raw_col].astype(float) * pitch_width) - (pitch_width / 2)
    return output
