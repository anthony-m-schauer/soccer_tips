"""
Script: metrica_loader.py
Project: Soccer TIPS

Overview:
    Provider-specific Metrica Sports sample data loader for Layer 1 ingestion.

Scope:
    - Supports Sample_Game_1 and Sample_Game_2 only.
    - Leaves Sample_Game_3 out of scope because it uses a different internal
      structure.

Core rules implemented:
    - Tracking CSVs are loaded with skiprows=2.
    - Tracking is wide format: one row = one frame.
    - Player coordinates are paired x/y columns.
    - Ball coordinates are extracted from the Ball x/y pair.
    - Raw normalized coordinates are preserved.
    - Metric coordinates are generated separately.
    - Metrica timing is 25 FPS / 0.04 seconds per frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from soccer_tips import config
from soccer_tips import schemas
from soccer_tips.data import common
from soccer_tips.data import normalize


SUPPORTED_SAMPLE_GAMES = {"Sample_Game_1", "Sample_Game_2"}


# -----------------------------------------------------------------------------
# File path helpers
# -----------------------------------------------------------------------------

def normalize_sample_game_name(sample_game: int | str) -> str:
    """Normalize sample game input to Sample_Game_N format."""
    sample_game_text = str(sample_game)
    if sample_game_text.startswith("Sample_Game_"):
        return sample_game_text
    return f"Sample_Game_{sample_game_text}"


def get_sample_game_dir(sample_game: int | str) -> Path:
    """Return the Metrica sample game folder."""
    return config.get_metrica_sample_game_dir(sample_game)


def _find_file(folder: Path, candidate_names: list[str], glob_patterns: list[str]) -> Path:
    """Find a file using exact candidates first, then glob patterns."""
    for name in candidate_names:
        path = folder / name
        if path.exists():
            return path

    matches: list[Path] = []
    for pattern in glob_patterns:
        matches.extend(sorted(folder.glob(pattern)))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple files matched in {folder}: {[str(path) for path in matches]}")

    raise FileNotFoundError(f"No matching file found in {folder}. Tried: {candidate_names + glob_patterns}")


def get_metrica_file_paths(sample_game: int | str) -> dict[str, Path]:
    """Return key Metrica file paths for Sample_Game_1 or Sample_Game_2."""
    game_name = normalize_sample_game_name(sample_game)
    if game_name not in SUPPORTED_SAMPLE_GAMES:
        raise ValueError(
            f"{game_name} is not supported in Layer 1 v0.1. "
            "Only Sample_Game_1 and Sample_Game_2 are in scope."
        )

    folder = get_sample_game_dir(game_name)
    return {
        "sample_game_dir": folder,
        "home_tracking": _find_file(
            folder,
            ["RawTrackingData_Home_Team.csv", f"{game_name}_RawTrackingData_Home_Team.csv"],
            ["*RawTrackingData_Home_Team*.csv", "*Home*Tracking*.csv"],
        ),
        "away_tracking": _find_file(
            folder,
            ["RawTrackingData_Away_Team.csv", f"{game_name}_RawTrackingData_Away_Team.csv"],
            ["*RawTrackingData_Away_Team*.csv", "*Away*Tracking*.csv"],
        ),
        "events": _find_file(
            folder,
            ["RawEventsData.csv", f"{game_name}_RawEventsData.csv"],
            ["*RawEventsData*.csv", "*Events*.csv"],
        ),
    }


# -----------------------------------------------------------------------------
# Raw loading functions
# -----------------------------------------------------------------------------

def load_tracking_csv(path: Path | str) -> pd.DataFrame:
    """Load a Metrica tracking CSV using the validated skiprows=2 parser."""
    return common.read_csv(path, skiprows=2)


def load_home_tracking(sample_game: int | str) -> pd.DataFrame:
    """Load home-team tracking for a supported Metrica sample game."""
    paths = get_metrica_file_paths(sample_game)
    return load_tracking_csv(paths["home_tracking"])


def load_away_tracking(sample_game: int | str) -> pd.DataFrame:
    """Load away-team tracking for a supported Metrica sample game."""
    paths = get_metrica_file_paths(sample_game)
    return load_tracking_csv(paths["away_tracking"])


def load_events(sample_game: int | str) -> pd.DataFrame:
    """Load Metrica events for a supported sample game."""
    paths = get_metrica_file_paths(sample_game)
    return common.read_csv(paths["events"])


# -----------------------------------------------------------------------------
# Wide-to-long tracking conversion
# -----------------------------------------------------------------------------

def _is_unnamed_column(column_name: str) -> bool:
    return str(column_name).lower().startswith("unnamed")


def get_coordinate_pairs(raw_tracking_df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Identify x/y coordinate pairs from a Metrica raw tracking dataframe.

    Returns tuples of:
        (entity_label, x_column, y_column)
    """
    columns = list(raw_tracking_df.columns)
    pairs: list[tuple[str, str, str]] = []

    # First three columns are Period, Frame, Time [s]. Remaining columns are x/y
    # pairs where the x column has the entity label and the y column is unnamed.
    index = 3
    while index < len(columns) - 1:
        x_col = columns[index]
        y_col = columns[index + 1]
        if not _is_unnamed_column(str(x_col)):
            pairs.append((str(x_col), str(x_col), str(y_col)))
            index += 2
        else:
            index += 1

    return pairs


def _core_tracking_values(row: pd.Series) -> tuple[Any, Any, Any]:
    """Extract period, frame, and timestamp from a raw Metrica tracking row."""
    period = row.get("Period")
    frame = row.get("Frame")
    timestamp = row.get("Time [s]")
    return period, frame, timestamp


def raw_tracking_to_canonical_players(
    raw_tracking_df: pd.DataFrame,
    sample_game: int | str,
    team_label: str,
    source_file: Path | str,
    pitch_length: float = config.DEFAULT_PITCH_LENGTH,
    pitch_width: float = config.DEFAULT_PITCH_WIDTH,
) -> pd.DataFrame:
    """Convert one Metrica raw tracking dataframe to canonical player rows."""
    sample_game_name = normalize_sample_game_name(sample_game)
    coordinate_pairs = get_coordinate_pairs(raw_tracking_df)
    player_pairs = [pair for pair in coordinate_pairs if pair[0].lower() != "ball"]

    rows: list[dict[str, Any]] = []
    for _, row in raw_tracking_df.iterrows():
        period, frame, timestamp = _core_tracking_values(row)

        for player_label, x_col, y_col in player_pairs:
            x_raw = row.get(x_col)
            y_raw = row.get(y_col)
            if pd.isna(x_raw) and pd.isna(y_raw):
                continue

            x_metric, y_metric = normalize.metrica_normalized_to_metric(
                x_raw,
                y_raw,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )
            player_id = f"{team_label}_{player_label}"

            rows.append(
                {
                    "provider": config.METRICA_PROVIDER,
                    "match_id": sample_game_name,
                    "frame": frame,
                    "timestamp": timestamp,
                    "period": period,
                    "team_id": team_label,
                    "player_id": player_id,
                    "player_name": player_label,
                    "position_group": None,
                    "role_name": None,
                    "x_raw": x_raw,
                    "y_raw": y_raw,
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "coordinate_system_raw": config.METRICA_COORDINATE_SYSTEM,
                    "is_detected": None,
                    "is_extrapolated": None,
                    "source_file": str(source_file),
                }
            )

    return schemas.enforce_tracking_player_schema(pd.DataFrame(rows))


def raw_tracking_to_canonical_ball(
    raw_tracking_df: pd.DataFrame,
    sample_game: int | str,
    source_file: Path | str,
    pitch_length: float = config.DEFAULT_PITCH_LENGTH,
    pitch_width: float = config.DEFAULT_PITCH_WIDTH,
) -> pd.DataFrame:
    """Extract canonical ball rows from one Metrica raw tracking dataframe."""
    sample_game_name = normalize_sample_game_name(sample_game)
    coordinate_pairs = get_coordinate_pairs(raw_tracking_df)
    ball_pair = next((pair for pair in coordinate_pairs if pair[0].lower() == "ball"), None)
    if ball_pair is None:
        return schemas.empty_tracking_ball_df()

    _, x_col, y_col = ball_pair
    rows: list[dict[str, Any]] = []

    for _, row in raw_tracking_df.iterrows():
        period, frame, timestamp = _core_tracking_values(row)
        x_raw = row.get(x_col)
        y_raw = row.get(y_col)
        if pd.isna(x_raw) and pd.isna(y_raw):
            continue

        x_metric, y_metric = normalize.metrica_normalized_to_metric(
            x_raw,
            y_raw,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

        rows.append(
            {
                "provider": config.METRICA_PROVIDER,
                "match_id": sample_game_name,
                "frame": frame,
                "timestamp": timestamp,
                "period": period,
                "ball_x_raw": x_raw,
                "ball_y_raw": y_raw,
                "ball_z_raw": None,
                "ball_x_metric": x_metric,
                "ball_y_metric": y_metric,
                "ball_z_metric": None,
                "coordinate_system_raw": config.METRICA_COORDINATE_SYSTEM,
                "ball_is_detected": None,
                "source_file": str(source_file),
            }
        )

    return schemas.enforce_tracking_ball_schema(pd.DataFrame(rows))


def load_metrica_game(sample_game: int | str) -> dict[str, Any]:
    """Load one supported Metrica sample game and convert to canonical tables."""
    game_name = normalize_sample_game_name(sample_game)
    paths = get_metrica_file_paths(game_name)

    home_raw = load_tracking_csv(paths["home_tracking"])
    away_raw = load_tracking_csv(paths["away_tracking"])
    events = load_events(game_name)

    home_players = raw_tracking_to_canonical_players(home_raw, game_name, "Home", paths["home_tracking"])
    away_players = raw_tracking_to_canonical_players(away_raw, game_name, "Away", paths["away_tracking"])
    tracking_players = schemas.enforce_tracking_player_schema(pd.concat([home_players, away_players], ignore_index=True))

    # Ball tracking is duplicated in the home/away files in the standard Metrica
    # sample format. Use home tracking as the canonical ball source for v0.1.
    tracking_ball = raw_tracking_to_canonical_ball(home_raw, game_name, paths["home_tracking"])

    return {
        "provider": config.METRICA_PROVIDER,
        "match_id": game_name,
        "sample_game": game_name,
        "home_raw_tracking": home_raw,
        "away_raw_tracking": away_raw,
        "tracking_players": tracking_players,
        "tracking_ball": tracking_ball,
        "events": events,
        "file_paths": paths,
    }
