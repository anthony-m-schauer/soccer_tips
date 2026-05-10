"""
Script: test_layer2_features.py
Project: Soccer TIPS

Overview:
    Lightweight sanity tests for Layer 2 v0.1 feature-engineering foundation.

Usage from project root:
    $env:PYTHONPATH="src"
    python tests/test_layer2_features.py
"""

from __future__ import annotations

import math

import pandas as pd

from soccer_tips.features import layer2_schemas
from soccer_tips.features import orientation
from soccer_tips.features import phase_context
from soccer_tips.features import team_shape


def test_layer2_schema_definitions_exist() -> None:
    """Confirm expected Layer 2 schema definitions are available."""
    assert "provider" in layer2_schemas.FRAME_TEAM_SHAPE_METRICS_COLUMNS
    assert "centroid_x" in layer2_schemas.FRAME_TEAM_SHAPE_METRICS_COLUMNS
    assert "team_width" in layer2_schemas.FRAME_TEAM_SHAPE_METRICS_COLUMNS
    assert "convex_hull_area" in layer2_schemas.FRAME_TEAM_SHAPE_METRICS_COLUMNS
    assert "frame_count" in layer2_schemas.PHASE_SEGMENT_SUMMARY_COLUMNS
    assert "profile_scope" in layer2_schemas.MATCH_TACTICAL_PROFILE_COLUMNS


def test_centroid_width_depth_and_hull_calculation() -> None:
    """Confirm team shape metrics work on a simple square."""
    df = pd.DataFrame(
        {
            "x_analysis": [0.0, 2.0, 2.0, 0.0],
            "y_analysis": [0.0, 0.0, 2.0, 2.0],
            "is_detected": [True, True, True, False],
            "is_extrapolated": [False, False, False, True],
        }
    )

    metrics = team_shape.compute_team_shape_for_group(df)

    assert metrics["centroid_x"] == 1.0
    assert metrics["centroid_y"] == 1.0
    assert metrics["team_width"] == 2.0
    assert metrics["team_depth"] == 2.0
    assert metrics["convex_hull_area"] == 4.0
    assert metrics["player_count_used"] == 4
    assert metrics["detected_player_count"] == 3
    assert metrics["extrapolated_player_count"] == 1


def test_convex_hull_handles_fewer_than_three_points_safely() -> None:
    """Confirm hull area is not computed for fewer than 3 points."""
    assert team_shape.convex_hull_area_from_points([]) is None
    assert team_shape.convex_hull_area_from_points([(0.0, 0.0)]) is None
    assert team_shape.convex_hull_area_from_points([(0.0, 0.0), (1.0, 1.0)]) is None


def test_frame_team_shape_metrics_grouping() -> None:
    """Confirm frame/team grouping returns one row per team-frame."""
    df = pd.DataFrame(
        {
            "provider": ["skillcorner"] * 6,
            "match_id": ["test_match"] * 6,
            "frame": [1] * 6,
            "timestamp": [0.1] * 6,
            "period": [1] * 6,
            "team_id": [10, 10, 10, 20, 20, 20],
            "x_metric": [0.0, 1.0, 0.0, 10.0, 11.0, 10.0],
            "y_metric": [0.0, 0.0, 1.0, 10.0, 10.0, 11.0],
            "is_detected": [True] * 6,
            "is_extrapolated": [False] * 6,
        }
    )

    metrics = team_shape.compute_frame_team_shape_metrics(df)

    assert len(metrics) == 2
    assert set(metrics["team_id"]) == {10, 20}
    assert list(metrics.columns[: len(layer2_schemas.FRAME_TEAM_SHAPE_METRICS_COLUMNS)]) == layer2_schemas.FRAME_TEAM_SHAPE_METRICS_COLUMNS


def test_phase_linking_handles_simple_frame_ranges() -> None:
    """Confirm SkillCorner phase linking uses frame_start/frame_end by period."""
    tracking = pd.DataFrame(
        {
            "provider": ["skillcorner", "skillcorner", "skillcorner", "skillcorner"],
            "match_id": ["test_match"] * 4,
            "frame": [1, 1, 5, 5],
            "timestamp": [0.1, 0.1, 0.5, 0.5],
            "period": [1, 1, 1, 1],
            "team_id": [100, 200, 100, 200],
        }
    )

    phases = pd.DataFrame(
        {
            "frame_start": [0, 4],
            "frame_end": [3, 7],
            "period": [1, 1],
            "team_in_possession_id": [100, 200],
            "team_in_possession_phase_type": ["build_up", "create"],
            "team_out_of_possession_phase_type": ["medium_block", "low_block"],
        }
    )

    linked = phase_context.link_skillcorner_phase_context(tracking, phases)

    frame_one_team_100 = linked[(linked["frame"] == 1) & (linked["team_id"] == 100)].iloc[0]
    frame_five_team_100 = linked[(linked["frame"] == 5) & (linked["team_id"] == 100)].iloc[0]

    assert frame_one_team_100["possession_status_for_team"] == "in_possession"
    assert frame_one_team_100["team_in_possession_phase_type"] == "build_up"
    assert frame_five_team_100["possession_status_for_team"] == "out_of_possession"
    assert frame_five_team_100["team_out_of_possession_phase_type"] == "low_block"


def test_orientation_adds_analysis_coordinates_without_destroying_metric_coordinates() -> None:
    """Confirm orientation module copies metric coordinates separately."""
    df = pd.DataFrame({"x_metric": [1.0], "y_metric": [2.0]})
    output = orientation.add_analysis_coordinates(df)

    assert output.loc[0, "x_metric"] == 1.0
    assert output.loc[0, "y_metric"] == 2.0
    assert output.loc[0, "x_analysis"] == 1.0
    assert output.loc[0, "y_analysis"] == 2.0
    assert output.loc[0, "orientation_normalization_applied"] is False or output.loc[0, "orientation_normalization_applied"] == False


def run_all_tests() -> None:
    """Run all Layer 2 sanity tests."""
    test_layer2_schema_definitions_exist()
    test_centroid_width_depth_and_hull_calculation()
    test_convex_hull_handles_fewer_than_three_points_safely()
    test_frame_team_shape_metrics_grouping()
    test_phase_linking_handles_simple_frame_ranges()
    test_orientation_adds_analysis_coordinates_without_destroying_metric_coordinates()
    print("All Layer 2 feature sanity checks passed.")


if __name__ == "__main__":
    run_all_tests()
