"""
Script: phase_context.py
Project: Soccer TIPS

Overview:
    SkillCorner phase-context linking utilities for Layer 2 v0.1.

Purpose:
    Convert Layer 1 canonical tracking rows into a frame/team phase-context table
    using SkillCorner phases_of_play.csv.

Important boundary:
    The raw tracking possession field is not used here. Possession status is
    derived from SkillCorner phase labels and team_in_possession_id.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


PHASE_CONTEXT_COLUMNS = [
    "provider",
    "match_id",
    "frame",
    "timestamp",
    "period",
    "team_id",
    "phase_index",
    "team_in_possession_id",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
    "possession_status_for_team",
]


REQUIRED_PHASE_COLUMNS = [
    "frame_start",
    "frame_end",
    "period",
    "team_in_possession_id",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
]


REQUIRED_TRACKING_COLUMNS = [
    "provider",
    "match_id",
    "frame",
    "timestamp",
    "period",
    "team_id",
]


def _normalize_team_id(value: Any) -> str | None:
    """Normalize team ids for robust comparison across int/float/string values."""
    if pd.isna(value):
        return None

    text = str(value).strip()

    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass

    return text


def determine_possession_status_for_team(team_id: Any, team_in_possession_id: Any) -> str:
    """Return in_possession/out_of_possession/unknown for one team-frame."""
    normalized_team_id = _normalize_team_id(team_id)
    normalized_possession_team_id = _normalize_team_id(team_in_possession_id)

    if normalized_team_id is None or normalized_possession_team_id is None:
        return "unknown"

    if normalized_team_id == normalized_possession_team_id:
        return "in_possession"

    return "out_of_possession"


def _prepare_phases(phases_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare SkillCorner phase rows for frame linking."""
    missing = [column for column in REQUIRED_PHASE_COLUMNS if column not in phases_df.columns]

    if missing:
        raise ValueError(f"Cannot link SkillCorner phases; missing columns: {missing}")

    phases = phases_df.copy()

    if "phase_index" not in phases.columns:
        phases = phases.reset_index(drop=True)
        phases["phase_index"] = phases.index

    phases["frame_start"] = pd.to_numeric(phases["frame_start"], errors="coerce")
    phases["frame_end"] = pd.to_numeric(phases["frame_end"], errors="coerce")
    phases["period"] = pd.to_numeric(phases["period"], errors="coerce")
    phases = phases.dropna(subset=["frame_start", "frame_end", "period"])

    return phases.sort_values(["period", "frame_start", "frame_end"]).reset_index(drop=True)


def build_frame_team_base(tracking_players_df: pd.DataFrame) -> pd.DataFrame:
    """Return unique provider/match/frame/team rows from canonical player tracking."""
    missing = [column for column in REQUIRED_TRACKING_COLUMNS if column not in tracking_players_df.columns]

    if missing:
        raise ValueError(f"Cannot build frame/team base; missing columns: {missing}")

    base = (
        tracking_players_df[REQUIRED_TRACKING_COLUMNS]
        .drop_duplicates()
        .sort_values(["provider", "match_id", "period", "frame", "team_id"])
        .reset_index(drop=True)
    )

    base["frame"] = pd.to_numeric(base["frame"], errors="coerce")
    base["period"] = pd.to_numeric(base["period"], errors="coerce")

    return base


def link_skillcorner_phase_context(
    tracking_players_df: pd.DataFrame,
    phases_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach SkillCorner phase labels to frame/team rows.

    Uses frame_start <= frame <= frame_end within each period. The raw tracking
    possession field is not used.
    """
    base = build_frame_team_base(tracking_players_df)
    phases = _prepare_phases(phases_df)

    linked_parts: list[pd.DataFrame] = []

    for period, base_period_df in base.groupby("period", dropna=False, sort=True):
        phase_period_df = phases[phases["period"] == period].copy()
        base_period_df = base_period_df.sort_values("frame").copy()

        if phase_period_df.empty:
            base_period_df["phase_index"] = pd.NA
            base_period_df["team_in_possession_id"] = pd.NA
            base_period_df["team_in_possession_phase_type"] = pd.NA
            base_period_df["team_out_of_possession_phase_type"] = pd.NA
            linked_parts.append(base_period_df)
            continue

        phase_columns = [
            "phase_index",
            "period",
            "frame_start",
            "frame_end",
            "team_in_possession_id",
            "team_in_possession_phase_type",
            "team_out_of_possession_phase_type",
        ]
        phase_period_df = phase_period_df[phase_columns].sort_values("frame_start")

        merged = pd.merge_asof(
            base_period_df,
            phase_period_df,
            left_on="frame",
            right_on="frame_start",
            by="period",
            direction="backward",
            allow_exact_matches=True,
        )

        valid_phase_mask = merged["frame_end"].notna() & (merged["frame"] <= merged["frame_end"])

        for column in [
            "phase_index",
            "team_in_possession_id",
            "team_in_possession_phase_type",
            "team_out_of_possession_phase_type",
        ]:
            merged.loc[~valid_phase_mask, column] = pd.NA

        linked_parts.append(merged)

    if not linked_parts:
        output = pd.DataFrame(columns=PHASE_CONTEXT_COLUMNS)
        return output

    linked = pd.concat(linked_parts, ignore_index=True)

    linked["possession_status_for_team"] = linked.apply(
        lambda row: determine_possession_status_for_team(row["team_id"], row["team_in_possession_id"]),
        axis=1,
    )

    output = linked[PHASE_CONTEXT_COLUMNS].sort_values(
        ["provider", "match_id", "period", "frame", "team_id"]
    )

    return output.reset_index(drop=True)


def merge_phase_context_into_metrics(
    frame_team_metrics_df: pd.DataFrame,
    phase_context_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge phase context into frame/team metrics."""
    merge_keys = ["provider", "match_id", "frame", "timestamp", "period", "team_id"]
    context_columns = [column for column in PHASE_CONTEXT_COLUMNS if column not in merge_keys]

    output = frame_team_metrics_df.drop(columns=context_columns, errors="ignore").merge(
        phase_context_df[merge_keys + context_columns],
        on=merge_keys,
        how="left",
    )

    if "possession_status_for_team" in output.columns:
        output["possession_status_for_team"] = output["possession_status_for_team"].fillna("unknown")

    return output
