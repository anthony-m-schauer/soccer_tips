"""
Script: build_layer1.py
Project: Soccer TIPS

Overview:
    Layer 1 build runner for reusable ingestion infrastructure.

Usage examples from project root:
    python -m soccer_tips.data.build_layer1 --provider skillcorner --skillcorner-match-id 2017461
    python -m soccer_tips.data.build_layer1 --provider metrica --metrica-game Sample_Game_1
    python -m soccer_tips.data.build_layer1 --provider all --skillcorner-match-id 2017461 --metrica-game Sample_Game_1
    python -m soccer_tips.data.build_layer1 --provider all --batch

Scope:
    This runner writes processed Layer 1 outputs only. It does not compute
    tactical features, build dashboards, or run AI interpretation.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from soccer_tips import config
from soccer_tips.data import common
from soccer_tips.data import metrica_loader
from soccer_tips.data import skillcorner_loader
from soccer_tips.data import validation


# -----------------------------------------------------------------------------
# Controlled v0.1 batch scope
# -----------------------------------------------------------------------------

V01_METRICA_BATCH_GAMES = ["Sample_Game_1", "Sample_Game_2"]


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def _write_metadata_json(metadata: dict[str, Any], output_path: Path) -> Path:
    """Write match metadata JSON."""
    return common.write_json(metadata, output_path)


def _write_optional_dataframe(df: pd.DataFrame | None, output_path: Path) -> Path | None:
    """Write dataframe if present."""
    if df is None:
        return None
    return common.write_csv(df, output_path)


# -----------------------------------------------------------------------------
# SkillCorner processing
# -----------------------------------------------------------------------------

def choose_first_skillcorner_match_id() -> str:
    """Return the first match id from SkillCorner matches.json."""
    matches_df = skillcorner_loader.load_matches_index()
    if matches_df.empty or "id" not in matches_df.columns:
        raise ValueError("Could not choose first SkillCorner match id. matches.json is empty or missing id column.")
    return str(matches_df.iloc[0]["id"])


def list_skillcorner_match_ids() -> list[str]:
    """Return all SkillCorner match ids discovered from matches.json."""
    matches_df = skillcorner_loader.load_matches_index()
    if matches_df.empty or "id" not in matches_df.columns:
        raise ValueError("Could not list SkillCorner match ids. matches.json is empty or missing id column.")
    return [str(match_id) for match_id in matches_df["id"].dropna().tolist()]


def process_skillcorner_match(match_id: int | str) -> dict[str, Any]:
    """Process one SkillCorner match into Layer 1 canonical outputs."""
    config.ensure_processed_directories()
    match_id_text = str(match_id)
    output_dir = config.PROCESSED_SKILLCORNER_DIR / match_id_text
    common.ensure_directory(output_dir)

    bundle = skillcorner_loader.load_skillcorner_match(match_id_text)
    report = validation.validate_skillcorner_loaded_match(bundle)

    output_paths = {
        "tracking_players": common.write_csv(
            bundle["tracking_players"], output_dir / "canonical_tracking_players.csv"
        ),
        "tracking_ball": common.write_csv(bundle["tracking_ball"], output_dir / "canonical_tracking_ball.csv"),
        "match_metadata": _write_metadata_json(bundle["metadata"], output_dir / "match_metadata.json"),
        "players_metadata": common.write_csv(bundle["players_metadata"], output_dir / "players_metadata.csv"),
        "phases": common.write_csv(bundle["phases"], output_dir / "phases_of_play.csv"),
        "dynamic_events": _write_optional_dataframe(bundle["dynamic_events"], output_dir / "dynamic_events.csv"),
        "validation_report": validation.write_validation_report(
            report, config.VALIDATION_REPORTS_DIR / f"skillcorner_{match_id_text}_validation.json"
        ),
    }

    return {
        "provider": config.SKILLCORNER_PROVIDER,
        "match_id": match_id_text,
        "output_dir": output_dir,
        "output_paths": output_paths,
        "validation_report": report,
    }


# -----------------------------------------------------------------------------
# Metrica processing
# -----------------------------------------------------------------------------

def process_metrica_game(sample_game: int | str) -> dict[str, Any]:
    """Process one Metrica sample game into Layer 1 canonical outputs."""
    config.ensure_processed_directories()
    sample_game_name = metrica_loader.normalize_sample_game_name(sample_game)
    output_dir = config.PROCESSED_METRICA_DIR / sample_game_name
    common.ensure_directory(output_dir)

    bundle = metrica_loader.load_metrica_game(sample_game_name)
    report = validation.validate_metrica_loaded_game(bundle)

    output_paths = {
        "tracking_players": common.write_csv(
            bundle["tracking_players"], output_dir / "canonical_tracking_players.csv"
        ),
        "tracking_ball": common.write_csv(bundle["tracking_ball"], output_dir / "canonical_tracking_ball.csv"),
        "events": common.write_csv(bundle["events"], output_dir / "events.csv"),
        "validation_report": validation.write_validation_report(
            report, config.VALIDATION_REPORTS_DIR / f"metrica_{sample_game_name}_validation.json"
        ),
    }

    return {
        "provider": config.METRICA_PROVIDER,
        "match_id": sample_game_name,
        "output_dir": output_dir,
        "output_paths": output_paths,
        "validation_report": report,
    }


# -----------------------------------------------------------------------------
# Batch processing
# -----------------------------------------------------------------------------

def _result_from_exception(provider: str, match_id: str, exc: Exception) -> dict[str, Any]:
    """Return a runner-style result for an unexpected fatal processing error."""
    return {
        "provider": provider,
        "match_id": match_id,
        "output_dir": None,
        "output_paths": {},
        "validation_report": {
            "provider": provider,
            "match_id": match_id,
            "status": "error",
            "summary": {},
            "warnings": [],
            "errors": [str(exc)],
            "checks": {"traceback": traceback.format_exc()},
        },
    }


def _safe_process_skillcorner_match(match_id: str) -> dict[str, Any]:
    """Process one SkillCorner match and return an error result instead of stopping the batch."""
    try:
        return process_skillcorner_match(match_id)
    except Exception as exc:  # pragma: no cover - safety path for real data batch runs
        return _result_from_exception(config.SKILLCORNER_PROVIDER, match_id, exc)


def _safe_process_metrica_game(sample_game: str) -> dict[str, Any]:
    """Process one Metrica game and return an error result instead of stopping the batch."""
    try:
        return process_metrica_game(sample_game)
    except Exception as exc:  # pragma: no cover - safety path for real data batch runs
        return _result_from_exception(config.METRICA_PROVIDER, sample_game, exc)


def process_batch(provider: str = "all") -> list[dict[str, Any]]:
    """Run the controlled Layer 1 v0.1 batch.

    Batch scope:
        - all SkillCorner matches discovered from matches.json
        - Metrica Sample_Game_1
        - Metrica Sample_Game_2

    Metrica Sample_Game_3 remains explicitly out of scope for v0.1.
    """
    results: list[dict[str, Any]] = []

    if provider in {"skillcorner", "all"}:
        for match_id in list_skillcorner_match_ids():
            results.append(_safe_process_skillcorner_match(match_id))

    if provider in {"metrica", "all"}:
        for sample_game in V01_METRICA_BATCH_GAMES:
            results.append(_safe_process_metrica_game(sample_game))

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for Layer 1 runner."""
    parser = argparse.ArgumentParser(description="Run Soccer TIPS Layer 1 ingestion processing.")
    parser.add_argument(
        "--provider",
        choices=["skillcorner", "metrica", "all"],
        default="all",
        help="Provider to process.",
    )
    parser.add_argument(
        "--skillcorner-match-id",
        default=None,
        help="SkillCorner match id to process. If omitted, first match from matches.json is used.",
    )
    parser.add_argument(
        "--metrica-game",
        default="Sample_Game_1",
        help="Metrica sample game to process. Only Sample_Game_1 and Sample_Game_2 are supported in v0.1.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Run controlled v0.1 batch mode. Processes all SkillCorner matches from matches.json "
            "and Metrica Sample_Game_1/Sample_Game_2. Sample_Game_3 remains out of scope."
        ),
    )
    return parser


def main() -> None:
    """Run Layer 1 ingestion processing from CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.batch:
        results = process_batch(args.provider)
    else:
        results: list[dict[str, Any]] = []

        if args.provider in {"skillcorner", "all"}:
            match_id = args.skillcorner_match_id or choose_first_skillcorner_match_id()
            results.append(process_skillcorner_match(match_id))

        if args.provider in {"metrica", "all"}:
            results.append(process_metrica_game(args.metrica_game))

    print("Layer 1 ingestion processing complete.")
    for result in results:
        output_dir = result["output_dir"] or "not written"
        print(f"- {result['provider']} {result['match_id']}: {output_dir}")
        print(f"  validation status: {result['validation_report']['status']}")

    error_count = sum(1 for result in results if result["validation_report"].get("status") == "error")
    if error_count:
        print(f"WARNING: {error_count} run(s) ended with fatal errors. Check printed status and logs.")


if __name__ == "__main__":
    main()
