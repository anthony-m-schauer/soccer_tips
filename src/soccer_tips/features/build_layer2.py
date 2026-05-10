"""
Script: build_layer2.py
Project: Soccer TIPS

Overview:
    Layer 2 v0.1 feature-engineering build runner.

Purpose:
    Consume Layer 1 canonical SkillCorner outputs and produce reusable feature
    tables for frame-level team shape metrics, phase-segment summaries, and
    match-level tactical profiles.

Usage from project root:
    $env:PYTHONPATH="src"
    python -m soccer_tips.features.build_layer2 --provider skillcorner --skillcorner-match-id 2017461
    python -m soccer_tips.features.build_layer2 --provider skillcorner --batch

Scope:
    This runner does not build Layer 3 modeling, clustering, AI interpretation,
    dashboards, recommendation systems, reports, advanced compactness models, or
    transition-stability interpretations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from soccer_tips import config
from soccer_tips.data import common
from soccer_tips.features import aggregation
from soccer_tips.features import layer2_schemas
from soccer_tips.features import orientation
from soccer_tips.features import phase_context
from soccer_tips.features import team_shape


FEATURES_DIR = config.PROCESSED_DIR / "features"
FRAME_TEAM_SHAPE_OUTPUT_PATH = FEATURES_DIR / "frame_team_shape_metrics.csv"
PHASE_SEGMENT_SUMMARIES_OUTPUT_PATH = FEATURES_DIR / "phase_segment_summaries.csv"
MATCH_TACTICAL_PROFILES_OUTPUT_PATH = FEATURES_DIR / "match_tactical_profiles.csv"


def ensure_features_directory() -> Path:
    """Create and return the Layer 2 processed features directory."""
    return common.ensure_directory(FEATURES_DIR)


def _skillcorner_processed_match_dir(match_id: int | str) -> Path:
    """Return processed SkillCorner match directory from Layer 1 outputs."""
    return config.PROCESSED_SKILLCORNER_DIR / str(match_id)


def discover_processed_skillcorner_match_ids() -> list[str]:
    """Return processed SkillCorner match ids that contain Layer 1 player tracking."""
    if not config.PROCESSED_SKILLCORNER_DIR.exists():
        return []

    match_ids: list[str] = []

    for path in sorted(config.PROCESSED_SKILLCORNER_DIR.iterdir()):
        if not path.is_dir():
            continue

        player_path = path / "canonical_tracking_players.csv"
        phases_path = path / "phases_of_play.csv"

        if player_path.exists() and phases_path.exists():
            match_ids.append(path.name)

    return match_ids


def load_layer1_skillcorner_outputs(match_id: int | str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    """Load Layer 1 processed SkillCorner player tracking, phases, and metadata."""
    match_dir = _skillcorner_processed_match_dir(match_id)

    player_path = common.require_file(
        match_dir / "canonical_tracking_players.csv",
        label=f"SkillCorner {match_id} canonical player tracking",
    )
    phases_path = common.require_file(
        match_dir / "phases_of_play.csv",
        label=f"SkillCorner {match_id} phases of play",
    )
    metadata_path = match_dir / "match_metadata.json"

    tracking_players = common.read_csv(player_path)
    phases = common.read_csv(phases_path)
    metadata = common.read_json(metadata_path) if metadata_path.exists() else None

    return tracking_players, phases, metadata


def build_skillcorner_match_features(match_id: int | str) -> dict[str, pd.DataFrame]:
    """Build Layer 2 v0.1 feature tables for one SkillCorner match."""
    tracking_players, phases, metadata = load_layer1_skillcorner_outputs(match_id)

    analysis_tracking = orientation.normalize_attacking_direction_placeholder(tracking_players, metadata)
    frame_team_metrics = team_shape.compute_frame_team_shape_metrics(analysis_tracking)
    frame_team_context = phase_context.link_skillcorner_phase_context(analysis_tracking, phases)

    frame_team_metrics = phase_context.merge_phase_context_into_metrics(frame_team_metrics, frame_team_context)
    frame_team_metrics = layer2_schemas.enforce_frame_team_shape_metrics_schema(frame_team_metrics)

    phase_segment_summaries = aggregation.aggregate_phase_segment_summaries(frame_team_metrics)
    match_tactical_profiles = aggregation.aggregate_match_tactical_profiles(frame_team_metrics)

    return {
        "frame_team_shape_metrics": frame_team_metrics,
        "phase_segment_summaries": phase_segment_summaries,
        "match_tactical_profiles": match_tactical_profiles,
    }


def _combine_feature_parts(feature_parts: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Combine per-match feature outputs into one set of output tables."""
    if not feature_parts:
        return {
            "frame_team_shape_metrics": layer2_schemas.empty_frame_team_shape_metrics_df(),
            "phase_segment_summaries": layer2_schemas.empty_phase_segment_summaries_df(),
            "match_tactical_profiles": layer2_schemas.empty_match_tactical_profiles_df(),
        }

    return {
        "frame_team_shape_metrics": layer2_schemas.enforce_frame_team_shape_metrics_schema(
            pd.concat([part["frame_team_shape_metrics"] for part in feature_parts], ignore_index=True)
        ),
        "phase_segment_summaries": layer2_schemas.enforce_phase_segment_summaries_schema(
            pd.concat([part["phase_segment_summaries"] for part in feature_parts], ignore_index=True)
        ),
        "match_tactical_profiles": layer2_schemas.enforce_match_tactical_profiles_schema(
            pd.concat([part["match_tactical_profiles"] for part in feature_parts], ignore_index=True)
        ),
    }


def build_skillcorner_layer2_features(match_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Build Layer 2 features for one or more processed SkillCorner matches."""
    feature_parts: list[dict[str, pd.DataFrame]] = []

    for match_id in match_ids:
        print(f"Building Layer 2 features for SkillCorner match {match_id}...")
        feature_parts.append(build_skillcorner_match_features(match_id))

    return _combine_feature_parts(feature_parts)


def write_layer2_feature_outputs(features: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Write Layer 2 feature output tables to data/processed/features/."""
    ensure_features_directory()

    output_paths = {
        "frame_team_shape_metrics": common.write_csv(
            features["frame_team_shape_metrics"],
            FRAME_TEAM_SHAPE_OUTPUT_PATH,
        ),
        "phase_segment_summaries": common.write_csv(
            features["phase_segment_summaries"],
            PHASE_SEGMENT_SUMMARIES_OUTPUT_PATH,
        ),
        "match_tactical_profiles": common.write_csv(
            features["match_tactical_profiles"],
            MATCH_TACTICAL_PROFILES_OUTPUT_PATH,
        ),
    }

    return output_paths


def run_layer2_build(provider: str, skillcorner_match_id: str | None, batch: bool) -> dict[str, Path]:
    """Run Layer 2 v0.1 build for the requested provider/match selection."""
    if provider != config.SKILLCORNER_PROVIDER:
        raise ValueError("Layer 2 v0.1 only supports provider='skillcorner'. Metrica is intentionally secondary for now.")

    if batch:
        match_ids = discover_processed_skillcorner_match_ids()
        if not match_ids:
            raise FileNotFoundError("No processed SkillCorner matches found under data/processed/skillcorner/.")
    else:
        match_ids = [skillcorner_match_id or "2017461"]

    features = build_skillcorner_layer2_features(match_ids)
    output_paths = write_layer2_feature_outputs(features)

    print("Layer 2 v0.1 feature build complete.")
    print(f"- provider: {provider}")
    print(f"- matches processed: {', '.join(match_ids)}")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")

    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build Soccer TIPS Layer 2 v0.1 feature tables.")
    parser.add_argument(
        "--provider",
        default=config.SKILLCORNER_PROVIDER,
        choices=[config.SKILLCORNER_PROVIDER],
        help="Provider to process. Layer 2 v0.1 supports SkillCorner only.",
    )
    parser.add_argument(
        "--skillcorner-match-id",
        default="2017461",
        help="Processed SkillCorner match id to build when not using --batch.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all processed SkillCorner matches found under data/processed/skillcorner/.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_layer2_build(
        provider=args.provider,
        skillcorner_match_id=args.skillcorner_match_id,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
