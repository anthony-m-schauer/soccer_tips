"""
Lightweight sanity tests for Layer 2 v0.2 football sanity validation helpers.

Run from project root:
    $env:PYTHONPATH="src"
    python tests/test_football_sanity.py
"""

from __future__ import annotations

import pandas as pd

from soccer_tips.features import football_sanity
from soccer_tips.features import visualize_layer2


def test_phase_label_for_team_returns_in_possession_label() -> None:
    row = {
        "possession_status_for_team": "in_possession",
        "team_in_possession_phase_type": "build_up",
        "team_out_of_possession_phase_type": "high_block",
    }

    assert football_sanity.phase_label_for_team(row) == "build_up"


def test_phase_label_for_team_returns_out_of_possession_label() -> None:
    row = {
        "possession_status_for_team": "out_of_possession",
        "team_in_possession_phase_type": "build_up",
        "team_out_of_possession_phase_type": "low_block",
    }

    assert football_sanity.phase_label_for_team(row) == "low_block"


def test_distribution_summary_contains_requested_phase_and_metric() -> None:
    df = pd.DataFrame(
        [
            {
                "provider": "skillcorner",
                "match_id": "1",
                "frame": 1,
                "period": 1,
                "team_id": 10,
                "possession_status_for_team": "in_possession",
                "team_in_possession_phase_type": "build_up",
                "team_out_of_possession_phase_type": "medium_block",
                "team_width": 40.0,
                "team_depth": 30.0,
                "convex_hull_area": 600.0,
                "avg_distance_to_centroid": 12.0,
                "centroid_x": 1.0,
                "centroid_y": 2.0,
            }
        ]
    )

    summary = football_sanity.summarize_metric_distributions_by_phase(df)
    row = summary[(summary["phase_label"] == "build_up") & (summary["metric"] == "team_width")].iloc[0]

    assert row["row_count"] == 1
    assert row["mean"] == 40.0


def test_reasonableness_check_warns_or_fails_large_width() -> None:
    df = pd.DataFrame(
        [
            {
                "match_id": "fake_match",
                "team_width": 90.0,
                "team_depth": 40.0,
                "convex_hull_area": 1000.0,
                "player_count_used": 11,
                "detected_player_count": 11,
                "extrapolated_player_count": 0,
                "phase_index": 1,
            }
            for _ in range(10)
        ]
    )

    result = football_sanity.run_reasonableness_checks(df)

    assert result["checks"]["team_width_pitch_bound"]["status"] in {"warning", "fail"}


def test_convex_hull_points_handles_fewer_than_three_points() -> None:
    points = [(0.0, 0.0), (1.0, 1.0)]
    hull = visualize_layer2.convex_hull_points(points)

    assert hull == points


def run_all_tests() -> None:
    """Run all test functions without requiring pytest."""
    test_phase_label_for_team_returns_in_possession_label()
    test_phase_label_for_team_returns_out_of_possession_label()
    test_distribution_summary_contains_requested_phase_and_metric()
    test_reasonableness_check_warns_or_fails_large_width()
    test_convex_hull_points_handles_fewer_than_three_points()
    print("All Layer 2 football sanity tests passed.")


if __name__ == "__main__":
    run_all_tests()
