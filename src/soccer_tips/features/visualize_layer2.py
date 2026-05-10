"""
Script: visualize_layer2.py
Project: Soccer TIPS

Overview:
    Validation-focused plotting helpers for Layer 2 v0.2 football sanity checks.

Purpose:
    Create selected pitch snapshots and phase-segment time-series plots from
    accepted Layer 1 and Layer 2 outputs. These are analytical validation visuals,
    not dashboards or final reports.

Scope guard:
    This module does not add new metrics, perform Layer 3 modeling, generate
    AI interpretation, create dashboards, or produce recommendation/reporting
    outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soccer_tips import config
from soccer_tips.data import common
from soccer_tips.features import build_layer2
from soccer_tips.features import football_sanity


FIGURE_OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "figures" / "layer2_sanity"

SHAPE_PLOT_TARGETS = [
    {"target_name": "build_up", "phase_options": ["build_up"]},
    {"target_name": "low_block", "phase_options": ["low_block"]},
    {"target_name": "medium_or_high_block", "phase_options": ["medium_block", "high_block"]},
    {"target_name": "transition", "phase_options": ["transition"]},
    {"target_name": "defending_transition", "phase_options": ["defending_transition"]},
]

TIME_SERIES_TARGETS = [
    "build_up",
    "low_block",
    "medium_block",
    "high_block",
    "transition",
    "defending_transition",
]

TIME_SERIES_METRICS = [
    "team_width",
    "team_depth",
    "convex_hull_area",
    "avg_distance_to_centroid",
]


def ensure_figure_output_dir() -> Path:
    """Create and return the Layer 2 sanity figure directory."""
    return common.ensure_directory(FIGURE_OUTPUT_DIR)


def sanitize_filename(value: Any) -> str:
    """Return a filesystem-safe token."""
    text = str(value)
    for char in ["/", "\\", ":", " ", "|", "[", "]", "(", ")"]:
        text = text.replace(char, "_")
    return text


def convex_hull_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return convex hull points using Andrew's monotonic chain algorithm."""
    unique_points = sorted(set(points))

    if len(unique_points) <= 1:
        return unique_points

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _load_match_tracking(match_id: str | int) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any] | None]:
    """Load Layer 1 processed player/ball tracking and metadata for a match."""
    match_dir = config.PROCESSED_SKILLCORNER_DIR / str(match_id)
    players_path = common.require_file(match_dir / "canonical_tracking_players.csv", label=f"SkillCorner {match_id} players")
    ball_path = match_dir / "canonical_tracking_ball.csv"
    metadata_path = match_dir / "match_metadata.json"

    players = common.read_csv(players_path)
    ball = common.read_optional_csv(ball_path)
    metadata = common.read_json(metadata_path) if metadata_path.exists() else None

    return players, ball, metadata


def _pitch_dimensions(metadata: dict[str, Any] | None) -> tuple[float, float]:
    """Return pitch length and width from metadata or defaults."""
    if metadata is None:
        return float(config.DEFAULT_PITCH_LENGTH), float(config.DEFAULT_PITCH_WIDTH)

    return (
        float(metadata.get("pitch_length", config.DEFAULT_PITCH_LENGTH)),
        float(metadata.get("pitch_width", config.DEFAULT_PITCH_WIDTH)),
    )


def _draw_pitch(ax: Any, pitch_length: float, pitch_width: float) -> None:
    """Draw a simple centered pitch outline."""
    x_min, x_max = -pitch_length / 2, pitch_length / 2
    y_min, y_max = -pitch_width / 2, pitch_width / 2

    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min])
    ax.axvline(0, linestyle="--", linewidth=1)

    center_circle = plt.Circle((0, 0), 9.15, fill=False, linewidth=1)
    ax.add_patch(center_circle)

    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_min - 5, y_max + 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x_metric")
    ax.set_ylabel("y_metric")


def _plot_team_hull_and_centroid(ax: Any, team_df: pd.DataFrame, team_id: Any) -> None:
    """Plot one team's player points, centroid, and convex hull when valid."""
    clean = team_df.dropna(subset=["x_metric", "y_metric"]).copy()

    if clean.empty:
        return

    ax.scatter(clean["x_metric"], clean["y_metric"], label=f"team {team_id} players", s=28)

    centroid_x = float(clean["x_metric"].mean())
    centroid_y = float(clean["y_metric"].mean())
    ax.scatter([centroid_x], [centroid_y], marker="x", s=100, label=f"team {team_id} centroid")

    points = list(zip(clean["x_metric"].astype(float), clean["y_metric"].astype(float)))
    hull = convex_hull_points(points)

    if len(hull) >= 3:
        hull_closed = hull + [hull[0]]
        x_values = [point[0] for point in hull_closed]
        y_values = [point[1] for point in hull_closed]
        ax.plot(x_values, y_values, linewidth=1.5, label=f"team {team_id} hull")


def plot_shape_frame(
    match_id: str | int,
    frame: int,
    period: int,
    phase_label: str,
    team_id: Any,
    team_in_possession_id: Any,
    output_path: Path,
) -> Path:
    """Create one pitch snapshot for a selected match/frame/team example."""
    players, ball, metadata = _load_match_tracking(match_id)
    pitch_length, pitch_width = _pitch_dimensions(metadata)

    frame_players = players[pd.to_numeric(players["frame"], errors="coerce") == int(frame)].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_pitch(ax, pitch_length, pitch_width)

    for current_team_id, team_df in frame_players.groupby("team_id", dropna=False, sort=True):
        _plot_team_hull_and_centroid(ax, team_df, current_team_id)

    if ball is not None and not ball.empty:
        frame_ball = ball[pd.to_numeric(ball["frame"], errors="coerce") == int(frame)].copy()
        if not frame_ball.empty and {"ball_x_metric", "ball_y_metric"}.issubset(frame_ball.columns):
            ball_row = frame_ball.iloc[0]
            if not pd.isna(ball_row["ball_x_metric"]) and not pd.isna(ball_row["ball_y_metric"]):
                ax.scatter(
                    [float(ball_row["ball_x_metric"])],
                    [float(ball_row["ball_y_metric"])],
                    marker="*",
                    s=150,
                    label="ball",
                )

    ax.set_title(
        f"Layer 2 sanity shape plot | match={match_id} frame={frame} period={period}\n"
        f"target_phase={phase_label} target_team={team_id} team_in_possession_id={team_in_possession_id}"
    )
    ax.legend(loc="upper right", fontsize="small")

    common.ensure_directory(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def _select_shape_examples(frame_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Select one frame/team example for each required phase target."""
    working = football_sanity.add_team_phase_label(frame_metrics)
    selected: list[dict[str, Any]] = []

    for target in SHAPE_PLOT_TARGETS:
        target_name = target["target_name"]
        phase_options = target["phase_options"]
        target_df = working[working["team_phase_label"].isin(phase_options)].copy()
        target_df = target_df[target_df["player_count_used"] >= football_sanity.MIN_REASONABLE_PLAYER_COUNT]

        if target_df.empty:
            selected.append(
                {
                    "plot_type": "shape_frame",
                    "target_name": target_name,
                    "phase_options": phase_options,
                    "status": "missing_phase_example",
                    "message": "No eligible frame/team row found for requested phase example.",
                }
            )
            continue

        # Pick a stable deterministic example near the middle of the available rows.
        target_df = target_df.sort_values(["match_id", "period", "frame", "team_id"]).reset_index(drop=True)
        row = target_df.iloc[len(target_df) // 2]

        selected.append(
            {
                "plot_type": "shape_frame",
                "target_name": target_name,
                "phase_label": row["team_phase_label"],
                "match_id": str(row["match_id"]),
                "frame": int(row["frame"]),
                "period": int(row["period"]),
                "team_id": row["team_id"],
                "team_in_possession_id": row["team_in_possession_id"],
                "status": "selected",
            }
        )

    return selected


def create_shape_validation_plots(frame_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Create required pitch-shape validation plots."""
    output_dir = ensure_figure_output_dir()
    selections = _select_shape_examples(frame_metrics)
    outputs: list[dict[str, Any]] = []

    for selection in selections:
        if selection.get("status") != "selected":
            outputs.append(selection)
            continue

        filename = (
            f"shape_{sanitize_filename(selection['target_name'])}_"
            f"{sanitize_filename(selection['phase_label'])}_"
            f"match_{sanitize_filename(selection['match_id'])}_"
            f"frame_{selection['frame']}_"
            f"team_{sanitize_filename(selection['team_id'])}.png"
        )
        output_path = output_dir / filename

        try:
            plot_shape_frame(
                match_id=selection["match_id"],
                frame=selection["frame"],
                period=selection["period"],
                phase_label=selection["phase_label"],
                team_id=selection["team_id"],
                team_in_possession_id=selection["team_in_possession_id"],
                output_path=output_path,
            )
            created = dict(selection)
            created["status"] = "created"
            created["path"] = str(output_path)
            outputs.append(created)
        except Exception as exc:  # pragma: no cover - defensive plot failure reporting
            failed = dict(selection)
            failed["status"] = "plot_failed"
            failed["error"] = str(exc)
            outputs.append(failed)

    return outputs


def _select_time_series_segments(phase_summaries: pd.DataFrame, max_segments: int = 6) -> list[dict[str, Any]]:
    """Select phase segments for time-series validation plots."""
    selected: list[dict[str, Any]] = []

    working = phase_summaries.copy()

    for phase_label in TIME_SERIES_TARGETS:
        if phase_label in ["build_up", "transition", "quick_break"]:
            phase_df = working[working["team_in_possession_phase_type"] == phase_label]
        else:
            phase_df = working[working["team_out_of_possession_phase_type"] == phase_label]

        if phase_df.empty:
            continue

        phase_df = phase_df.sort_values("frame_count", ascending=False).reset_index(drop=True)
        row = phase_df.iloc[0]

        selected.append(
            {
                "plot_type": "phase_segment_time_series",
                "phase_label": phase_label,
                "match_id": str(row["match_id"]),
                "phase_index": row["phase_index"],
                "period": int(row["period"]),
                "team_id": row["team_id"],
                "status": "selected",
            }
        )

        if len(selected) >= max_segments:
            break

    return selected


def plot_phase_segment_time_series(
    frame_metrics: pd.DataFrame,
    match_id: str | int,
    phase_index: Any,
    team_id: Any,
    phase_label: str,
    output_path: Path,
) -> Path:
    """Plot key shape metrics over time for one phase/team segment."""
    segment_df = frame_metrics[
        (frame_metrics["match_id"].astype(str) == str(match_id))
        & (frame_metrics["phase_index"].astype(str) == str(phase_index))
        & (frame_metrics["team_id"].astype(str) == str(team_id))
    ].copy()

    if segment_df.empty:
        raise ValueError(f"No frame metrics found for match={match_id}, phase_index={phase_index}, team_id={team_id}.")

    segment_df = segment_df.sort_values("frame")
    x_values = segment_df["timestamp"] if "timestamp" in segment_df.columns else segment_df["frame"]

    fig, axes = plt.subplots(len(TIME_SERIES_METRICS), 1, figsize=(11, 8), sharex=True)

    for axis, metric in zip(axes, TIME_SERIES_METRICS):
        axis.plot(x_values, pd.to_numeric(segment_df[metric], errors="coerce"))
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestamp" if "timestamp" in segment_df.columns else "frame")
    fig.suptitle(
        f"Layer 2 sanity time series | match={match_id} phase_index={phase_index} "
        f"team={team_id} phase={phase_label}"
    )

    common.ensure_directory(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def create_phase_segment_time_series_plots(
    frame_metrics: pd.DataFrame,
    phase_summaries: pd.DataFrame,
    max_segments: int = 6,
) -> list[dict[str, Any]]:
    """Create selected phase-segment time-series validation plots."""
    output_dir = ensure_figure_output_dir()
    selections = _select_time_series_segments(phase_summaries, max_segments=max_segments)
    outputs: list[dict[str, Any]] = []

    for selection in selections:
        filename = (
            f"timeseries_{sanitize_filename(selection['phase_label'])}_"
            f"match_{sanitize_filename(selection['match_id'])}_"
            f"phase_{sanitize_filename(selection['phase_index'])}_"
            f"team_{sanitize_filename(selection['team_id'])}.png"
        )
        output_path = output_dir / filename

        try:
            plot_phase_segment_time_series(
                frame_metrics=frame_metrics,
                match_id=selection["match_id"],
                phase_index=selection["phase_index"],
                team_id=selection["team_id"],
                phase_label=selection["phase_label"],
                output_path=output_path,
            )
            created = dict(selection)
            created["status"] = "created"
            created["path"] = str(output_path)
            outputs.append(created)
        except Exception as exc:  # pragma: no cover - defensive plot failure reporting
            failed = dict(selection)
            failed["status"] = "plot_failed"
            failed["error"] = str(exc)
            outputs.append(failed)

    return outputs


def create_all_validation_plots(
    frame_metrics: pd.DataFrame,
    phase_summaries: pd.DataFrame,
    max_time_series_segments: int = 6,
) -> list[dict[str, Any]]:
    """Create all Layer 2 v0.2 football sanity validation plots."""
    outputs = []
    outputs.extend(create_shape_validation_plots(frame_metrics))
    outputs.extend(create_phase_segment_time_series_plots(frame_metrics, phase_summaries, max_segments=max_time_series_segments))
    return outputs


def main() -> None:
    """Command-line entry point for plot-only generation."""
    tables = football_sanity.load_layer2_feature_outputs()
    outputs = create_all_validation_plots(
        frame_metrics=tables.frame_team_shape_metrics,
        phase_summaries=tables.phase_segment_summaries,
    )

    created_count = sum(1 for item in outputs if item.get("status") == "created")
    print("Layer 2 v0.2 sanity validation plots complete.")
    print(f"- figures directory: {FIGURE_OUTPUT_DIR}")
    print(f"- plots created: {created_count}")
    for item in outputs:
        if item.get("status") == "created":
            print(f"  - {item['path']}")
        else:
            print(f"  - {item.get('target_name', item.get('phase_label', 'unknown'))}: {item.get('status')}")


if __name__ == "__main__":
    main()
