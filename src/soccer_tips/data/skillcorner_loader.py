"""
Script: skillcorner_loader.py
Project: Soccer TIPS

Overview:
    Provider-specific SkillCorner Open Data loader for Layer 1 ingestion.

Core rules implemented:
    - matches.json is the match index.
    - Tracking files are JSONL, one frame per line.
    - tracking player_data.player_id joins to players metadata id.
    - dynamic_events.csv is optional and should warn, not crash.
    - raw SkillCorner coordinates are preserved.
    - metric coordinates are generated separately and match raw values in v0.1.
    - is_detected is preserved; is_extrapolated is derived when possible.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from soccer_tips import config
from soccer_tips import schemas
from soccer_tips.data import common


# -----------------------------------------------------------------------------
# File path helpers
# -----------------------------------------------------------------------------

def get_match_dir(match_id: int | str) -> Path:
    """Return the SkillCorner raw match directory."""
    return config.get_skillcorner_match_dir(match_id)


def get_match_file_paths(match_id: int | str) -> dict[str, Path]:
    """Return expected SkillCorner file paths for one match."""
    match_id_text = str(match_id)
    match_dir = get_match_dir(match_id_text)
    return {
        "match_dir": match_dir,
        "metadata": match_dir / f"{match_id_text}_match.json",
        "tracking": match_dir / f"{match_id_text}_tracking_extrapolated.jsonl",
        "phases": match_dir / f"{match_id_text}_phases_of_play.csv",
        "dynamic_events": match_dir / f"{match_id_text}_dynamic_events.csv",
    }


# -----------------------------------------------------------------------------
# Core raw loading functions
# -----------------------------------------------------------------------------

def load_matches_index(path: Path | str = config.SKILLCORNER_MATCHES_INDEX_PATH) -> pd.DataFrame:
    """Load the SkillCorner matches index into a dataframe."""
    data = common.read_json(path)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if "matches" in data and isinstance(data["matches"], list):
            return pd.DataFrame(data["matches"])
        return pd.DataFrame([data])
    raise ValueError(f"Unsupported matches index JSON structure: {type(data).__name__}")


def load_match_metadata(match_id: int | str) -> dict[str, Any]:
    """Load one SkillCorner match metadata JSON file."""
    paths = get_match_file_paths(match_id)
    metadata = common.read_json(paths["metadata"])
    return metadata


def _extract_role_fields(player: dict[str, Any]) -> tuple[Any, Any]:
    """Extract position group and role name from SkillCorner player metadata."""
    role = player.get("player_role") or {}
    if not isinstance(role, dict):
        return None, None
    return role.get("position_group"), role.get("name")


def load_players(match_id: int | str) -> pd.DataFrame:
    """Load and flatten SkillCorner player metadata.

    Important join rule:
        tracking player_data.player_id -> players_df["id"]
    """
    metadata = load_match_metadata(match_id)
    players = metadata.get("players", [])

    rows: list[dict[str, Any]] = []
    for player in players:
        position_group, role_name = _extract_role_fields(player)
        rows.append(
            {
                "match_id": str(match_id),
                "id": player.get("id"),
                "team_id": player.get("team_id"),
                "short_name": player.get("short_name"),
                "player_name": player.get("short_name") or player.get("name"),
                "number": player.get("number"),
                "start_time": player.get("start_time"),
                "end_time": player.get("end_time"),
                "position_group": position_group,
                "role_name": role_name,
                "trackable_object": player.get("trackable_object"),
                "team_player_id": player.get("team_player_id"),
            }
        )

    players_df = pd.DataFrame(rows)
    return schemas.enforce_column_order(players_df, schemas.SKILLCORNER_PLAYER_METADATA_COLUMNS)


def iter_tracking_frames(match_id: int | str) -> Iterator[dict[str, Any]]:
    """Yield SkillCorner tracking frames from JSONL, one line per frame."""
    paths = get_match_file_paths(match_id)
    tracking_path = common.require_file(paths["tracking"], label="SkillCorner tracking JSONL")

    with tracking_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {tracking_path}, line {line_number}") from exc


def load_tracking_frames(match_id: int | str) -> list[dict[str, Any]]:
    """Load SkillCorner tracking frames into memory.

    This is acceptable for v0.1/open-data scale. If future files become much
    larger, downstream code should switch to chunked writing.
    """
    return list(iter_tracking_frames(match_id))


def load_phases(match_id: int | str) -> pd.DataFrame:
    """Load SkillCorner phases of play."""
    paths = get_match_file_paths(match_id)
    return common.read_csv(paths["phases"])


def load_dynamic_events(match_id: int | str) -> pd.DataFrame | None:
    """Load SkillCorner dynamic events if present.

    Dynamic events are optional because match_id 1953632 is known to be missing
    this file in the open dataset.
    """
    paths = get_match_file_paths(match_id)
    events_path = paths["dynamic_events"]
    if not events_path.exists():
        warnings.warn(
            f"Missing optional SkillCorner dynamic events file for match {match_id}: {events_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return pd.read_csv(events_path, low_memory=False)


# -----------------------------------------------------------------------------
# Canonical conversion helpers
# -----------------------------------------------------------------------------

def _build_player_lookup(players_df: pd.DataFrame) -> dict[Any, dict[str, Any]]:
    """Build lookup keyed by SkillCorner player metadata id."""
    lookup: dict[Any, dict[str, Any]] = {}
    if players_df.empty or "id" not in players_df.columns:
        return lookup
    for _, row in players_df.iterrows():
        lookup[row["id"]] = row.to_dict()
    return lookup


def _as_bool_or_none(value: Any) -> bool | None:
    """Return bool when value is boolean-like, otherwise None."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    return bool(value)


def tracking_frames_to_canonical_tables(
    match_id: int | str,
    frames: list[dict[str, Any]],
    players_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert SkillCorner tracking frames into canonical player and ball tables."""
    paths = get_match_file_paths(match_id)
    source_file = str(paths["tracking"])
    player_lookup = _build_player_lookup(players_df)

    player_rows: list[dict[str, Any]] = []
    ball_rows: list[dict[str, Any]] = []

    for frame_record in frames:
        frame = frame_record.get("frame")
        timestamp = frame_record.get("timestamp")
        period = frame_record.get("period")

        for player_data in frame_record.get("player_data", []) or []:
            player_id = player_data.get("player_id")
            player_meta = player_lookup.get(player_id, {})
            is_detected = _as_bool_or_none(player_data.get("is_detected"))
            is_extrapolated = None if is_detected is None else not is_detected
            x_raw = player_data.get("x")
            y_raw = player_data.get("y")

            player_rows.append(
                {
                    "provider": config.SKILLCORNER_PROVIDER,
                    "match_id": str(match_id),
                    "frame": frame,
                    "timestamp": timestamp,
                    "period": period,
                    "team_id": player_meta.get("team_id"),
                    "player_id": player_id,
                    "player_name": player_meta.get("player_name") or player_meta.get("short_name"),
                    "position_group": player_meta.get("position_group"),
                    "role_name": player_meta.get("role_name"),
                    "x_raw": x_raw,
                    "y_raw": y_raw,
                    "x_metric": x_raw,
                    "y_metric": y_raw,
                    "coordinate_system_raw": config.SKILLCORNER_COORDINATE_SYSTEM,
                    "is_detected": is_detected,
                    "is_extrapolated": is_extrapolated,
                    "source_file": source_file,
                }
            )

        ball_data = frame_record.get("ball_data") or {}
        ball_x_raw = ball_data.get("x")
        ball_y_raw = ball_data.get("y")
        ball_z_raw = ball_data.get("z")

        ball_rows.append(
            {
                "provider": config.SKILLCORNER_PROVIDER,
                "match_id": str(match_id),
                "frame": frame,
                "timestamp": timestamp,
                "period": period,
                "ball_x_raw": ball_x_raw,
                "ball_y_raw": ball_y_raw,
                "ball_z_raw": ball_z_raw,
                "ball_x_metric": ball_x_raw,
                "ball_y_metric": ball_y_raw,
                "ball_z_metric": ball_z_raw,
                "coordinate_system_raw": config.SKILLCORNER_COORDINATE_SYSTEM,
                "ball_is_detected": _as_bool_or_none(ball_data.get("is_detected")),
                "source_file": source_file,
            }
        )

    players_out = schemas.enforce_tracking_player_schema(pd.DataFrame(player_rows))
    ball_out = schemas.enforce_tracking_ball_schema(pd.DataFrame(ball_rows))
    return players_out, ball_out


def load_skillcorner_match(match_id: int | str) -> dict[str, Any]:
    """Load one SkillCorner match and convert tracking to canonical tables."""
    metadata = load_match_metadata(match_id)
    players_metadata = load_players(match_id)
    frames = load_tracking_frames(match_id)
    phases = load_phases(match_id)
    dynamic_events = load_dynamic_events(match_id)
    tracking_players, tracking_ball = tracking_frames_to_canonical_tables(match_id, frames, players_metadata)

    return {
        "provider": config.SKILLCORNER_PROVIDER,
        "match_id": str(match_id),
        "metadata": metadata,
        "players_metadata": players_metadata,
        "tracking_frames": frames,
        "tracking_players": tracking_players,
        "tracking_ball": tracking_ball,
        "phases": phases,
        "dynamic_events": dynamic_events,
    }
