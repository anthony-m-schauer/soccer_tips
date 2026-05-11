"""
Script: team_shape.py
Project: Soccer TIPS

Overview:
    Frame-level team shape metric utilities for Layer 2 v0.1.

Metrics:
    - centroid_x / centroid_y
    - team_width
    - team_depth
    - convex_hull_area
    - average distance to centroid
    - player count and detection-quality counts

Scope:
    These are simple feature-engineering metrics. They are not final tactical
    identity claims and should not be interpreted as a complete compactness or
    transition-stability model.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from soccer_tips.features import layer2_schemas
from soccer_tips.features import orientation


MIN_PLAYERS_FOR_BASIC_SHAPE = 3
MIN_PLAYERS_FOR_FULL_TEAM_WARNING = 8


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Convert common boolean-like values to a boolean Series."""
    if series.dtype == bool:
        return series.fillna(False)

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _convex_hull_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return convex hull points using Andrew's monotonic chain algorithm."""
    unique_points = sorted(set(points))

    if len(unique_points) <= 1:
        return unique_points

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def convex_hull_area_from_points(points: list[tuple[float, float]]) -> float | None:
    """Return convex hull area for points, or None when fewer than 3 points exist."""
    if len(points) < 3:
        return None

    hull = _convex_hull_points(points)

    if len(hull) < 3:
        return None

    area = 0.0
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        area += point[0] * next_point[1] - next_point[0] * point[1]

    return abs(area) / 2.0


def compute_team_shape_for_group(
    group_df: pd.DataFrame,
    x_column: str = orientation.ANALYSIS_X_COLUMN,
    y_column: str = orientation.ANALYSIS_Y_COLUMN,
) -> dict[str, Any]:
    """Compute team shape metrics for one frame/team group."""
    required_columns = [x_column, y_column, "is_detected", "is_extrapolated"]
    missing = [column for column in required_columns if column not in group_df.columns]

    if missing:
        raise ValueError(f"Cannot compute team shape; missing columns: {missing}")

    valid_points_df = group_df.dropna(subset=[x_column, y_column]).copy()
    player_count_used = int(len(valid_points_df))

    detected_player_count = int(_to_bool_series(valid_points_df["is_detected"]).sum())
    extrapolated_player_count = int(_to_bool_series(valid_points_df["is_extrapolated"]).sum())

    result: dict[str, Any] = {
        "centroid_x": np.nan,
        "centroid_y": np.nan,
        "team_width": np.nan,
        "team_depth": np.nan,
        "convex_hull_area": np.nan,
        "avg_distance_to_centroid": np.nan,
        "player_count_used": player_count_used,
        "detected_player_count": detected_player_count,
        "extrapolated_player_count": extrapolated_player_count,
        "metric_warning_flag": player_count_used < MIN_PLAYERS_FOR_FULL_TEAM_WARNING,
    }

    if player_count_used < MIN_PLAYERS_FOR_BASIC_SHAPE:
        result["metric_warning_flag"] = True
        return result

    x_values = valid_points_df[x_column].astype(float)
    y_values = valid_points_df[y_column].astype(float)

    centroid_x = float(x_values.mean())
    centroid_y = float(y_values.mean())
    # Football convention used by Soccer TIPS:
    # - width is the lateral spread across the pitch: y range
    # - depth is the longitudinal spread along the pitch: x range
    team_width = float(y_values.max() - y_values.min())
    team_depth = float(x_values.max() - x_values.min())

    distances = [
        sqrt((float(x) - centroid_x) ** 2 + (float(y) - centroid_y) ** 2)
        for x, y in zip(x_values, y_values)
    ]

    points = list(zip(x_values.tolist(), y_values.tolist()))
    hull_area = convex_hull_area_from_points(points)

    result.update(
        {
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "team_width": team_width,
            "team_depth": team_depth,
            "convex_hull_area": np.nan if hull_area is None else float(hull_area),
            "avg_distance_to_centroid": float(np.mean(distances)),
        }
    )

    return result


def compute_frame_team_shape_metrics(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Compute one row of team shape metrics per provider/match/frame/team."""
    if orientation.ANALYSIS_X_COLUMN not in tracking_df.columns or orientation.ANALYSIS_Y_COLUMN not in tracking_df.columns:
        working_df = orientation.add_analysis_coordinates(tracking_df)
    else:
        working_df = tracking_df.copy()

    group_columns = ["provider", "match_id", "frame", "timestamp", "period", "team_id"]
    missing_group_columns = [column for column in group_columns if column not in working_df.columns]

    if missing_group_columns:
        raise ValueError(f"Cannot compute frame-team metrics; missing columns: {missing_group_columns}")

    rows: list[dict[str, Any]] = []

    for group_key, group_df in working_df.groupby(group_columns, dropna=False, sort=True):
        base_row = dict(zip(group_columns, group_key))
        metrics = compute_team_shape_for_group(group_df)
        base_row.update(metrics)
        rows.append(base_row)

    metrics_df = pd.DataFrame(rows)

    for context_column in [
        "phase_index",
        "team_in_possession_id",
        "team_in_possession_phase_type",
        "team_out_of_possession_phase_type",
        "possession_status_for_team",
    ]:
        if context_column not in metrics_df.columns:
            metrics_df[context_column] = pd.NA

    return layer2_schemas.enforce_frame_team_shape_metrics_schema(metrics_df)
