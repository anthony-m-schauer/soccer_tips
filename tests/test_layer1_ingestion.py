"""
Script: test_layer1_ingestion.py
Project: Soccer TIPS

Overview:
    Lightweight sanity checks for Layer 1 ingestion infrastructure.

Run from project root:
    python tests/test_layer1_ingestion.py

Notes:
    These tests avoid external datasets. They check schemas, config paths,
    common I/O helpers, coordinate conversion, and validation report structure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soccer_tips import config  # noqa: E402
from soccer_tips import schemas  # noqa: E402
from soccer_tips.data import common  # noqa: E402
from soccer_tips.data import normalize  # noqa: E402
from soccer_tips.data import validation  # noqa: E402
from soccer_tips.data import build_layer1  # noqa: E402
from soccer_tips.data import qa_layer1_outputs  # noqa: E402


def test_canonical_tracking_player_schema_contains_expected_columns() -> None:
    expected = {
        "provider",
        "match_id",
        "frame",
        "timestamp",
        "period",
        "team_id",
        "player_id",
        "player_name",
        "x_raw",
        "y_raw",
        "x_metric",
        "y_metric",
        "coordinate_system_raw",
        "is_detected",
        "is_extrapolated",
        "source_file",
    }
    assert expected.issubset(set(schemas.CANONICAL_TRACKING_PLAYER_COLUMNS))


def test_canonical_tracking_ball_schema_contains_expected_columns() -> None:
    expected = {
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
    }
    assert expected.issubset(set(schemas.CANONICAL_TRACKING_BALL_COLUMNS))


def test_config_paths_resolve_to_project_structure() -> None:
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.SRC_DIR == config.PROJECT_ROOT / "src"
    assert config.EXTERNAL_DIR == config.PROJECT_ROOT / "external"
    assert config.PROCESSED_DIR == config.DATA_DIR / "processed"
    assert config.SKILLCORNER_MATCHES_INDEX_PATH.name == "matches.json"


def test_common_json_and_csv_helpers_work() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        json_path = temp_path / "nested" / "test.json"
        common.write_json({"a": 1, "b": [1, 2]}, json_path)
        assert common.read_json(json_path) == {"a": 1, "b": [1, 2]}

        csv_path = temp_path / "nested" / "test.csv"
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        common.write_csv(df, csv_path)
        loaded = common.read_csv(csv_path)
        assert loaded.shape == (2, 2)
        assert loaded["x"].tolist() == [1, 2]


def test_skillcorner_coordinate_conversion_preserves_metric_centered_values() -> None:
    x_metric, y_metric = normalize.skillcorner_to_metric(-10.0, 5.0)
    assert x_metric == -10.0
    assert y_metric == 5.0


def test_metrica_coordinate_conversion_generates_centered_metric_values() -> None:
    x_metric, y_metric = normalize.metrica_normalized_to_metric(0.5, 0.5, pitch_length=105, pitch_width=68)
    assert x_metric == 0.0
    assert y_metric == 0.0

    x_metric, y_metric = normalize.metrica_normalized_to_metric(0.0, 0.0, pitch_length=105, pitch_width=68)
    assert x_metric == -52.5
    assert y_metric == -34.0


def test_validation_report_structure_is_json_ready() -> None:
    report = validation.create_validation_report(
        provider="test_provider",
        match_id="test_match",
        status="pass",
        summary={"rows": 10},
        checks={"file_check": {"ok": True}},
        warnings=[],
        errors=[],
    )
    assert set(report.keys()) == {"provider", "match_id", "status", "summary", "warnings", "errors", "checks"}
    assert report["provider"] == "test_provider"
    assert report["status"] == "pass"


def test_frame_continuity_detects_gaps() -> None:
    result = validation.check_frame_continuity([1, 2, 4, 5])
    assert result["has_frame_gaps"] is True
    assert result["missing_frame_count"] == 1
    assert result["sample_missing_frames"] == [3]


def test_batch_scope_excludes_metrica_sample_game_3() -> None:
    assert build_layer1.V01_METRICA_BATCH_GAMES == ["Sample_Game_1", "Sample_Game_2"]
    assert "Sample_Game_3" not in build_layer1.V01_METRICA_BATCH_GAMES


def test_qa_schema_check_detects_extra_columns() -> None:
    result = qa_layer1_outputs._schema_check_from_columns(
        ["provider", "match_id", "extra_col"],
        ["provider", "match_id"],
    )
    assert result["missing_columns"] == []
    assert result["extra_columns"] == ["extra_col"]
    assert result["status"] == "warning"


def run_all_tests() -> None:
    test_canonical_tracking_player_schema_contains_expected_columns()
    test_canonical_tracking_ball_schema_contains_expected_columns()
    test_config_paths_resolve_to_project_structure()
    test_common_json_and_csv_helpers_work()
    test_skillcorner_coordinate_conversion_preserves_metric_centered_values()
    test_metrica_coordinate_conversion_generates_centered_metric_values()
    test_validation_report_structure_is_json_ready()
    test_frame_continuity_detects_gaps()
    test_batch_scope_excludes_metrica_sample_game_3()
    test_qa_schema_check_detects_extra_columns()
    print("All Layer 1 ingestion sanity checks passed.")


if __name__ == "__main__":
    run_all_tests()
