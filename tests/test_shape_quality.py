"""
Script: test_shape_quality.py
Project: Soccer TIPS

Overview:
    Lightweight tests for Layer 2 v0.2.3 shape metric quality flags.

Usage from project root:
    $env:PYTHONPATH="src"
    python tests/test_shape_quality.py
"""

from __future__ import annotations

import pandas as pd

from soccer_tips.features import qa_shape_quality
from soccer_tips.features import shape_quality


def _fake_frame_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "provider": "skillcorner",
                "match_id": "fake_match",
                "frame": 1,
                "period": 1,
                "team_id": "A",
                "phase_index": 10,
                "possession_status_for_team": "in_possession",
                "team_in_possession_phase_type": "build_up",
                "team_out_of_possession_phase_type": "medium_block",
                "team_width": 60.0,
                "team_depth": 80.0,
                "player_count_used": 11,
                "detected_player_count": 10,
                "extrapolated_player_count": 1,
            },
            {
                "provider": "skillcorner",
                "match_id": "fake_match",
                "frame": 2,
                "period": 1,
                "team_id": "A",
                "phase_index": 10,
                "possession_status_for_team": "in_possession",
                "team_in_possession_phase_type": "build_up",
                "team_out_of_possession_phase_type": "medium_block",
                "team_width": 69.0,
                "team_depth": 80.0,
                "player_count_used": 11,
                "detected_player_count": 5,
                "extrapolated_player_count": 6,
            },
            {
                "provider": "skillcorner",
                "match_id": "fake_match",
                "frame": 3,
                "period": 1,
                "team_id": "A",
                "phase_index": 10,
                "possession_status_for_team": "in_possession",
                "team_in_possession_phase_type": "build_up",
                "team_out_of_possession_phase_type": "medium_block",
                "team_width": 72.0,
                "team_depth": 80.0,
                "player_count_used": 11,
                "detected_player_count": 11,
                "extrapolated_player_count": 0,
            },
            {
                "provider": "skillcorner",
                "match_id": "fake_match",
                "frame": 4,
                "period": 1,
                "team_id": "A",
                "phase_index": 10,
                "possession_status_for_team": "in_possession",
                "team_in_possession_phase_type": "build_up",
                "team_out_of_possession_phase_type": "medium_block",
                "team_width": 50.0,
                "team_depth": 80.0,
                "player_count_used": 7,
                "detected_player_count": 7,
                "extrapolated_player_count": 0,
            },
        ]
    )


def test_quality_flags_strict_warning_does_not_exclude() -> None:
    flags = shape_quality.build_shape_metric_quality_flags(
        _fake_frame_metrics(),
        pitch_dimension_lookup={"fake_match": {"pitch_length": 105.0, "pitch_width": 68.0}},
    )

    row = flags.loc[flags["frame"] == 2].iloc[0]
    assert bool(row["team_width_strict_bound_flag"]) is True
    assert bool(row["team_width_tolerance_fail_flag"]) is False
    assert bool(row["high_extrapolated_count_flag"]) is True
    assert bool(row["high_extrapolated_rate_flag"]) is True
    assert bool(row["layer3_shape_metric_include"]) is True


def test_quality_flags_tolerance_failure_excludes() -> None:
    flags = shape_quality.build_shape_metric_quality_flags(
        _fake_frame_metrics(),
        pitch_dimension_lookup={"fake_match": {"pitch_length": 105.0, "pitch_width": 68.0}},
    )

    row = flags.loc[flags["frame"] == 3].iloc[0]
    assert bool(row["team_width_strict_bound_flag"]) is True
    assert bool(row["team_width_tolerance_fail_flag"]) is True
    assert bool(row["layer3_shape_metric_include"]) is False
    assert "team_width_tolerance_fail" in row["quality_exclusion_reason"]


def test_quality_flags_low_player_count_excludes() -> None:
    flags = shape_quality.build_shape_metric_quality_flags(
        _fake_frame_metrics(),
        pitch_dimension_lookup={"fake_match": {"pitch_length": 105.0, "pitch_width": 68.0}},
    )

    row = flags.loc[flags["frame"] == 4].iloc[0]
    assert bool(row["low_player_count_flag"]) is True
    assert bool(row["layer3_shape_metric_include"]) is False
    assert "low_player_count" in row["quality_exclusion_reason"]


def test_shape_quality_summary_status_warning_for_rare_exclusions() -> None:
    flags = shape_quality.build_shape_metric_quality_flags(
        _fake_frame_metrics(),
        pitch_dimension_lookup={"fake_match": {"pitch_length": 105.0, "pitch_width": 68.0}},
    )

    summary = qa_shape_quality.build_shape_quality_summary(flags)

    assert summary["total_rows_checked"] == 4
    assert summary["rows_with_strict_width_warning"] == 2
    assert summary["rows_with_width_tolerance_failure"] == 1
    assert summary["low_player_count_rows"] == 1
    assert summary["rows_included_for_layer3"] == 2
    assert summary["rows_excluded_from_layer3"] == 2
    assert summary["status"] == "fail"


def run_all_tests() -> None:
    test_quality_flags_strict_warning_does_not_exclude()
    test_quality_flags_tolerance_failure_excludes()
    test_quality_flags_low_player_count_excludes()
    test_shape_quality_summary_status_warning_for_rare_exclusions()
    print("All Layer 2 shape quality tests passed.")


if __name__ == "__main__":
    run_all_tests()
