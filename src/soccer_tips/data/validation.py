"""
Script: validation.py
Project: Soccer TIPS

Overview:
    Validation utilities for Layer 1 ingestion outputs and raw-file checks.

Scope:
    These checks are intentionally structural. They do not compute tactical
    metrics or make Layer 2 analytical claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from soccer_tips import config
from soccer_tips.data import common
from soccer_tips.data import skillcorner_loader
from soccer_tips.data import metrica_loader


# -----------------------------------------------------------------------------
# Generic checks
# -----------------------------------------------------------------------------

def check_files_exist(paths: dict[str, Path | str]) -> dict[str, Any]:
    """Check whether named files exist."""
    checked = {name: str(path) for name, path in paths.items()}
    exists = {name: Path(path).exists() for name, path in paths.items()}
    missing = [name for name, path_exists in exists.items() if not path_exists]
    return {
        "checked_paths": checked,
        "exists": exists,
        "missing": missing,
        "all_required_exist": len(missing) == 0,
    }


def check_duplicate_frames(frames: Iterable[Any]) -> dict[str, Any]:
    """Check duplicate frame ids from any frame iterable."""
    frame_series = pd.Series(list(frames), name="frame").dropna()
    duplicate_count = int(frame_series.duplicated().sum())
    return {
        "total_frame_values": int(len(frame_series)),
        "unique_frames": int(frame_series.nunique()),
        "duplicate_frame_values": duplicate_count,
        "has_duplicates": duplicate_count > 0,
    }


def check_frame_continuity(frames: Iterable[Any]) -> dict[str, Any]:
    """Check frame continuity over sorted unique integer-like frame ids."""
    frame_series = pd.Series(list(frames), name="frame").dropna()
    if frame_series.empty:
        return {
            "min_frame": None,
            "max_frame": None,
            "unique_frames": 0,
            "missing_frame_count": 0,
            "has_frame_gaps": False,
            "sample_missing_frames": [],
        }

    unique_frames = sorted(pd.to_numeric(frame_series, errors="coerce").dropna().astype(int).unique().tolist())
    if not unique_frames:
        return {
            "min_frame": None,
            "max_frame": None,
            "unique_frames": 0,
            "missing_frame_count": 0,
            "has_frame_gaps": False,
            "sample_missing_frames": [],
        }

    expected = set(range(unique_frames[0], unique_frames[-1] + 1))
    actual = set(unique_frames)
    missing = sorted(expected - actual)

    return {
        "min_frame": int(unique_frames[0]),
        "max_frame": int(unique_frames[-1]),
        "unique_frames": int(len(unique_frames)),
        "missing_frame_count": int(len(missing)),
        "has_frame_gaps": len(missing) > 0,
        "sample_missing_frames": missing[:25],
    }


def check_player_counts(tracking_players_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize number of player rows per frame."""
    if tracking_players_df.empty or "frame" not in tracking_players_df.columns:
        return {
            "frames_with_players": 0,
            "min_players_per_frame": None,
            "max_players_per_frame": None,
            "player_count_distribution": {},
        }

    counts = tracking_players_df.groupby("frame")["player_id"].count()
    distribution = counts.value_counts().sort_index().to_dict()
    return {
        "frames_with_players": int(counts.shape[0]),
        "min_players_per_frame": int(counts.min()) if not counts.empty else None,
        "max_players_per_frame": int(counts.max()) if not counts.empty else None,
        "player_count_distribution": {str(int(key)): int(value) for key, value in distribution.items()},
    }


def create_validation_report(
    provider: str,
    match_id: str,
    status: str,
    summary: dict[str, Any] | None = None,
    checks: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Create a standardized, JSON-writeable validation report dictionary."""
    return {
        "provider": provider,
        "match_id": str(match_id),
        "status": status,
        "summary": summary or {},
        "warnings": warnings or [],
        "errors": errors or [],
        "checks": checks or {},
    }


# -----------------------------------------------------------------------------
# SkillCorner validation
# -----------------------------------------------------------------------------

def check_skillcorner_expected_files(match_id: int | str) -> dict[str, Any]:
    """Check expected SkillCorner files, treating dynamic events as optional."""
    paths = skillcorner_loader.get_match_file_paths(match_id)
    required = {
        "metadata": paths["metadata"],
        "tracking": paths["tracking"],
        "phases": paths["phases"],
    }
    optional = {"dynamic_events": paths["dynamic_events"]}

    required_check = check_files_exist(required)
    optional_check = check_files_exist(optional)

    warnings: list[str] = []
    if not optional_check["exists"].get("dynamic_events", False):
        warnings.append(f"Missing optional SkillCorner dynamic events file for match {match_id}.")

    return {
        "required": required_check,
        "optional": optional_check,
        "warnings": warnings,
    }


def check_skillcorner_phase_alignment(phases_df: pd.DataFrame, tracking_frames: Iterable[Any]) -> dict[str, Any]:
    """Check SkillCorner phase windows against available tracking frames."""
    frame_series = pd.Series(list(tracking_frames)).dropna()
    if frame_series.empty or phases_df.empty:
        return {
            "phase_rows": int(len(phases_df)),
            "phase_start_before_tracking": None,
            "phase_end_after_tracking": None,
            "invalid_phase_ranges": None,
        }

    min_frame = int(pd.to_numeric(frame_series, errors="coerce").min())
    max_frame = int(pd.to_numeric(frame_series, errors="coerce").max())

    frame_start = pd.to_numeric(phases_df["frame_start"], errors="coerce")
    frame_end = pd.to_numeric(phases_df["frame_end"], errors="coerce")

    return {
        "phase_rows": int(len(phases_df)),
        "tracking_min_frame": min_frame,
        "tracking_max_frame": max_frame,
        "phase_start_before_tracking": int((frame_start < min_frame).sum()),
        "phase_end_after_tracking": int((frame_end > max_frame).sum()),
        "invalid_phase_ranges": int((frame_end < frame_start).sum()),
    }


def check_event_alignment(
    events_df: pd.DataFrame | None,
    tracking_frames: Iterable[Any],
    start_col: str,
    end_col: str,
) -> dict[str, Any]:
    """Check event frame windows against available tracking frames."""
    if events_df is None:
        return {
            "event_rows": None,
            "events_before_tracking": None,
            "events_after_tracking": None,
            "invalid_event_ranges": None,
            "events_available": False,
        }

    frame_series = pd.Series(list(tracking_frames)).dropna()
    if frame_series.empty or events_df.empty:
        return {
            "event_rows": int(len(events_df)),
            "events_before_tracking": None,
            "events_after_tracking": None,
            "invalid_event_ranges": None,
            "events_available": True,
        }

    min_frame = int(pd.to_numeric(frame_series, errors="coerce").min())
    max_frame = int(pd.to_numeric(frame_series, errors="coerce").max())
    start_frames = pd.to_numeric(events_df[start_col], errors="coerce")
    end_frames = pd.to_numeric(events_df[end_col], errors="coerce")

    return {
        "event_rows": int(len(events_df)),
        "tracking_min_frame": min_frame,
        "tracking_max_frame": max_frame,
        "events_before_tracking": int((start_frames < min_frame).sum()),
        "events_after_tracking": int((end_frames > max_frame).sum()),
        "invalid_event_ranges": int((end_frames < start_frames).sum()),
        "events_available": True,
    }


def validate_skillcorner_loaded_match(bundle: dict[str, Any]) -> dict[str, Any]:
    """Build validation report for a loaded SkillCorner match bundle."""
    match_id = str(bundle["match_id"])
    tracking_players = bundle["tracking_players"]
    tracking_ball = bundle["tracking_ball"]
    phases = bundle["phases"]
    dynamic_events = bundle["dynamic_events"]
    tracking_frames_raw = bundle["tracking_frames"]
    frame_ids = [frame.get("frame") for frame in tracking_frames_raw]

    file_check = check_skillcorner_expected_files(match_id)
    warnings = list(file_check.get("warnings", []))

    checks = {
        "files": file_check,
        "tracking_frame_continuity": check_frame_continuity(frame_ids),
        "tracking_frame_duplicates": check_duplicate_frames(frame_ids),
        "player_counts": check_player_counts(tracking_players),
        "phase_alignment": check_skillcorner_phase_alignment(phases, frame_ids),
        "dynamic_event_alignment": check_event_alignment(dynamic_events, frame_ids, "frame_start", "frame_end")
        if dynamic_events is not None and {"frame_start", "frame_end"}.issubset(dynamic_events.columns)
        else {
            "events_available": dynamic_events is not None,
            "event_rows": None if dynamic_events is None else int(len(dynamic_events)),
            "note": "Dynamic events missing or expected frame_start/frame_end columns not found.",
        },
    }

    summary = {
        "tracking_total_frames": int(len(tracking_frames_raw)),
        "tracking_player_rows": int(len(tracking_players)),
        "tracking_ball_rows": int(len(tracking_ball)),
        "phase_rows": int(len(phases)),
        "event_rows": None if dynamic_events is None else int(len(dynamic_events)),
        "dynamic_events_available": dynamic_events is not None,
    }

    errors = []
    required_missing = file_check["required"]["missing"]
    if required_missing:
        errors.append(f"Missing required SkillCorner files: {required_missing}")

    status = "pass_with_warnings" if warnings and not errors else "pass"
    if errors:
        status = "fail"

    return create_validation_report(
        provider=config.SKILLCORNER_PROVIDER,
        match_id=match_id,
        status=status,
        summary=summary,
        checks=checks,
        warnings=warnings,
        errors=errors,
    )


# -----------------------------------------------------------------------------
# Metrica validation
# -----------------------------------------------------------------------------

def check_metrica_expected_files(sample_game: int | str) -> dict[str, Any]:
    """Check expected Metrica files for Sample_Game_1 or Sample_Game_2."""
    try:
        paths = metrica_loader.get_metrica_file_paths(sample_game)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "required": {
                "checked_paths": {},
                "exists": {},
                "missing": [],
                "all_required_exist": False,
            },
            "warnings": [],
            "errors": [str(exc)],
        }

    required = {
        "home_tracking": paths["home_tracking"],
        "away_tracking": paths["away_tracking"],
        "events": paths["events"],
    }
    return {
        "required": check_files_exist(required),
        "warnings": [],
        "errors": [],
    }


def check_metrica_invalid_event_ranges(events_df: pd.DataFrame) -> dict[str, Any]:
    """Check Metrica invalid event ranges.

    One kickoff edge case may appear in the sample data; this is reported as a
    warning-level structural finding, not meaningful corruption.
    """
    required = {"Start Frame", "End Frame"}
    if not required.issubset(events_df.columns):
        return {
            "event_rows": int(len(events_df)),
            "invalid_event_ranges": None,
            "note": "Start Frame / End Frame columns not found.",
        }

    start_frames = pd.to_numeric(events_df["Start Frame"], errors="coerce")
    end_frames = pd.to_numeric(events_df["End Frame"], errors="coerce")
    invalid_mask = end_frames < start_frames
    invalid_rows = events_df.loc[invalid_mask].head(10).to_dict(orient="records")

    return {
        "event_rows": int(len(events_df)),
        "invalid_event_ranges": int(invalid_mask.sum()),
        "sample_invalid_event_rows": invalid_rows,
    }


def validate_metrica_loaded_game(bundle: dict[str, Any]) -> dict[str, Any]:
    """Build validation report for a loaded Metrica sample game bundle."""
    match_id = str(bundle["match_id"])
    tracking_players = bundle["tracking_players"]
    tracking_ball = bundle["tracking_ball"]
    home_raw = bundle["home_raw_tracking"]
    away_raw = bundle["away_raw_tracking"]
    events = bundle["events"]

    file_check = check_metrica_expected_files(match_id)
    warnings = list(file_check.get("warnings", []))
    errors = list(file_check.get("errors", []))

    home_frames = home_raw["Frame"].tolist() if "Frame" in home_raw.columns else []
    away_frames = away_raw["Frame"].tolist() if "Frame" in away_raw.columns else []

    checks = {
        "files": file_check,
        "home_frame_continuity": check_frame_continuity(home_frames),
        "away_frame_continuity": check_frame_continuity(away_frames),
        "home_frame_duplicates": check_duplicate_frames(home_frames),
        "away_frame_duplicates": check_duplicate_frames(away_frames),
        "player_counts": check_player_counts(tracking_players),
        "event_alignment": check_event_alignment(events, home_frames, "Start Frame", "End Frame"),
        "invalid_event_ranges": check_metrica_invalid_event_ranges(events),
    }

    invalid_ranges = checks["invalid_event_ranges"].get("invalid_event_ranges")
    if invalid_ranges:
        warnings.append(
            f"Metrica contains {invalid_ranges} invalid event range(s); known kickoff initialization edge cases may occur."
        )

    summary = {
        "home_tracking_rows": int(len(home_raw)),
        "away_tracking_rows": int(len(away_raw)),
        "tracking_player_rows": int(len(tracking_players)),
        "tracking_ball_rows": int(len(tracking_ball)),
        "event_rows": int(len(events)),
        "fps": config.METRICA_FPS,
    }

    required_missing = file_check.get("required", {}).get("missing", [])
    if required_missing:
        errors.append(f"Missing required Metrica files: {required_missing}")

    status = "pass_with_warnings" if warnings and not errors else "pass"
    if errors:
        status = "fail"

    return create_validation_report(
        provider=config.METRICA_PROVIDER,
        match_id=match_id,
        status=status,
        summary=summary,
        checks=checks,
        warnings=warnings,
        errors=errors,
    )


def write_validation_report(report: dict[str, Any], path: Path | str) -> Path:
    """Write a validation report to JSON."""
    return common.write_json(report, path)
