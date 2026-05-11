"""
Layer 2 v0.2.1 — team_width Warning Review

Place this file at:
    src/soccer_tips/features/review_team_width_v021.py

Run from the Soccer TIPS project root with:
    $env:PYTHONPATH="src"
    python -m soccer_tips.features.review_team_width_v021

This script inspects the actual local Layer 2 generated outputs, identifies
team_width pitch-bound violations, inspects raw player coordinates for the
worst examples, creates exact validation plots, and writes a review package.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.cwd()
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
SKILLCORNER_DIR = PROJECT_ROOT / "data" / "processed" / "skillcorner"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "layer2_sanity"
REVIEW_FIGURES_DIR = FIGURES_DIR / "team_width_review"

FRAME_METRICS_PATH = FEATURES_DIR / "frame_team_shape_metrics.csv"
PHASE_SEGMENTS_PATH = FEATURES_DIR / "phase_segment_summaries.csv"
FOOTBALL_SANITY_JSON_PATH = FEATURES_DIR / "layer2_v02_football_sanity_summary.json"
FOOTBALL_SANITY_CSV_PATH = FEATURES_DIR / "layer2_v02_football_sanity_summary.csv"

REVIEW_JSON_PATH = FEATURES_DIR / "layer2_v021_team_width_warning_review.json"
REVIEW_CSV_PATH = FEATURES_DIR / "layer2_v021_team_width_worst_examples.csv"
ADDENDUM_TXT_PATH = FEATURES_DIR / "layer2_v021_team_width_handoff_addendum.txt"

DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0
DEFAULT_NEIGHBOR_WINDOW = 10
DEFAULT_WORST_N = 10
DEFAULT_PLOT_N = 5


@dataclass
class PitchDimensions:
    length: float = DEFAULT_PITCH_LENGTH
    width: float = DEFAULT_PITCH_WIDTH

    @property
    def x_min(self) -> float:
        return -self.length / 2.0

    @property
    def x_max(self) -> float:
        return self.length / 2.0

    @property
    def y_min(self) -> float:
        return -self.width / 2.0

    @property
    def y_max(self) -> float:
        return self.width / 2.0


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def recursive_numeric_candidates(obj: Any, key_tokens: Sequence[str]) -> List[Tuple[str, float]]:
    """Find numeric values whose key path contains any/all useful tokens."""
    results: List[Tuple[str, float]] = []

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                new_path = f"{path}.{k}" if path else str(k)
                walk(v, new_path)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        else:
            number = safe_float(value)
            if number is not None:
                lowered = path.lower()
                if all(token.lower() in lowered for token in key_tokens):
                    results.append((path, number))
    walk(obj)
    return results


def infer_pitch_bound(sanity_json: Dict[str, Any], metadata_bounds: Dict[str, PitchDimensions]) -> float:
    """Infer the team_width pitch-width bound used by prior sanity QA.

    The football sanity JSON structure may evolve, so this searches for an
    explicit team_width bound/threshold first. If no explicit value is found,
    it falls back to the max SkillCorner pitch width in match metadata, usually 68.
    """
    candidates = []
    for tokens in (["team_width", "bound"], ["team_width", "threshold"], ["pitch", "width", "bound"]):
        candidates.extend(recursive_numeric_candidates(sanity_json, tokens))

    # Prefer plausible pitch-width-scale values, not counts.
    plausible = [(path, val) for path, val in candidates if 40.0 <= val <= 90.0]
    if plausible:
        # If several are present, use the largest plausible bound as conservative.
        return float(max(val for _, val in plausible))

    if metadata_bounds:
        return float(max(dim.width for dim in metadata_bounds.values()))

    return DEFAULT_PITCH_WIDTH


def load_pitch_dimensions(match_id: str) -> PitchDimensions:
    metadata_path = SKILLCORNER_DIR / str(match_id) / "match_metadata.json"
    if not metadata_path.exists():
        return PitchDimensions()
    metadata = read_json(metadata_path)

    # Current Layer 1 metadata JSON may be a flat dict or list-like record.
    if isinstance(metadata, list) and metadata:
        metadata = metadata[0]
    if isinstance(metadata, dict) and "0" in metadata and isinstance(metadata["0"], dict):
        metadata = metadata["0"]

    length = safe_float(metadata.get("pitch_length") if isinstance(metadata, dict) else None, DEFAULT_PITCH_LENGTH)
    width = safe_float(metadata.get("pitch_width") if isinstance(metadata, dict) else None, DEFAULT_PITCH_WIDTH)
    return PitchDimensions(length=length or DEFAULT_PITCH_LENGTH, width=width or DEFAULT_PITCH_WIDTH)


# ---------------------------------------------------------------------------
# Convex hull + plotting utilities
# ---------------------------------------------------------------------------

def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Monotonic-chain convex hull. Returns hull points without requiring scipy."""
    pts = sorted(set((float(x), float(y)) for x, y in points if pd.notna(x) and pd.notna(y)))
    if len(pts) <= 1:
        return pts

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def draw_pitch(ax: Any, dimensions: PitchDimensions) -> None:
    x_min, x_max = dimensions.x_min, dimensions.x_max
    y_min, y_max = dimensions.y_min, dimensions.y_max

    # Outer pitch and halfway line.
    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min])
    ax.plot([0, 0], [y_min, y_max])

    # Center circle, approximate.
    circle = plt.Circle((0, 0), 9.15, fill=False)
    ax.add_patch(circle)

    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_min - 5, y_max + 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x_metric")
    ax.set_ylabel("y_metric")


def create_outlier_plot(
    match_id: str,
    frame: int,
    period: int,
    target_team_id: Any,
    frame_metrics_row: pd.Series,
    tracking_players: pd.DataFrame,
    dimensions: PitchDimensions,
    output_path: Path,
) -> None:
    frame_players = tracking_players[(tracking_players["frame"] == frame) & (tracking_players["period"] == period)].copy()
    if frame_players.empty:
        frame_players = tracking_players[tracking_players["frame"] == frame].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    draw_pitch(ax, dimensions)

    markers = ["o", "s", "^", "D"]
    for i, (team_id, team_df) in enumerate(frame_players.groupby("team_id")):
        marker = markers[i % len(markers)]
        label = f"team {team_id}"
        ax.scatter(team_df["x_metric"], team_df["y_metric"], marker=marker, label=label)

        valid_points = team_df[["x_metric", "y_metric"]].dropna()
        if len(valid_points) >= 3:
            hull = convex_hull(list(valid_points.itertuples(index=False, name=None)))
            if len(hull) >= 3:
                closed = hull + [hull[0]]
                xs = [p[0] for p in closed]
                ys = [p[1] for p in closed]
                ax.plot(xs, ys, linestyle="--", linewidth=1)

        centroid_x = team_df["x_metric"].mean()
        centroid_y = team_df["y_metric"].mean()
        ax.scatter([centroid_x], [centroid_y], marker="x", s=100)

    title = (
        f"team_width review | match={match_id} frame={frame} period={period} team={target_team_id}\n"
        f"team_width={frame_metrics_row.get('team_width', np.nan):.3f} | "
        f"phase={frame_metrics_row.get('team_in_possession_phase_type', None)} / "
        f"{frame_metrics_row.get('team_out_of_possession_phase_type', None)} | "
        f"possession={frame_metrics_row.get('possession_status_for_team', None)}"
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Review logic
# ---------------------------------------------------------------------------

def inspect_raw_coordinates(
    row: pd.Series,
    dimensions: PitchDimensions,
) -> Dict[str, Any]:
    match_id = str(row["match_id"])
    frame = safe_int(row["frame"])
    period = safe_int(row["period"])
    team_id = row["team_id"]

    tracking_path = SKILLCORNER_DIR / match_id / "canonical_tracking_players.csv"
    if not tracking_path.exists():
        return {
            "match_id": match_id,
            "frame": frame,
            "period": period,
            "team_id": str(team_id),
            "error": f"Missing tracking file: {tracking_path}",
        }

    usecols = None
    tracking = pd.read_csv(tracking_path, usecols=usecols)
    mask = (tracking["frame"] == frame) & (tracking["period"] == period) & (tracking["team_id"].astype(str) == str(team_id))
    team_rows = tracking.loc[mask].copy()

    if team_rows.empty:
        return {
            "match_id": match_id,
            "frame": frame,
            "period": period,
            "team_id": str(team_id),
            "error": "No player rows found for match/frame/period/team.",
        }

    team_rows = team_rows.dropna(subset=["y_metric"])
    if team_rows.empty:
        return {
            "match_id": match_id,
            "frame": frame,
            "period": period,
            "team_id": str(team_id),
            "error": "No non-null y_metric rows found.",
        }

    min_idx = team_rows["y_metric"].idxmin()
    max_idx = team_rows["y_metric"].idxmax()
    min_row = team_rows.loc[min_idx]
    max_row = team_rows.loc[max_idx]

    sorted_y = team_rows["y_metric"].sort_values().to_numpy()
    width = float(sorted_y[-1] - sorted_y[0]) if len(sorted_y) else np.nan
    width_without_min = float(sorted_y[-1] - sorted_y[1]) if len(sorted_y) >= 2 else np.nan
    width_without_max = float(sorted_y[-2] - sorted_y[0]) if len(sorted_y) >= 2 else np.nan

    pitch_y_min = dimensions.y_min
    pitch_y_max = dimensions.y_max
    outside_rows = team_rows[(team_rows["y_metric"] < pitch_y_min) | (team_rows["y_metric"] > pitch_y_max)]

    one_player_driving = bool(width_without_min <= dimensions.width or width_without_max <= dimensions.width)

    def player_info(player_row: pd.Series) -> Dict[str, Any]:
        return {
            "player_id": _json_default(player_row.get("player_id")),
            "short_name": _json_default(player_row.get("short_name")),
            "x_metric": safe_float(player_row.get("x_metric")),
            "y_metric": safe_float(player_row.get("y_metric")),
            "is_detected": _json_default(player_row.get("is_detected")),
            "position_group": _json_default(player_row.get("position_group")),
            "role_name": _json_default(player_row.get("role_name")),
        }

    return {
        "match_id": match_id,
        "frame": frame,
        "period": period,
        "team_id": str(team_id),
        "pitch_width": dimensions.width,
        "expected_y_bounds": [pitch_y_min, pitch_y_max],
        "player_rows_found": int(len(team_rows)),
        "min_y_metric": safe_float(team_rows["y_metric"].min()),
        "max_y_metric": safe_float(team_rows["y_metric"].max()),
        "raw_team_width_from_players": safe_float(width),
        "min_y_player": player_info(min_row),
        "max_y_player": player_info(max_row),
        "outside_expected_pitch_y_count": int(len(outside_rows)),
        "outside_expected_pitch_y_players": [player_info(r) for _, r in outside_rows.iterrows()],
        "width_without_min_y_player": safe_float(width_without_min),
        "width_without_max_y_player": safe_float(width_without_max),
        "one_player_driving_warning": one_player_driving,
        "min_max_detection_status": {
            "min_y_is_detected": _json_default(min_row.get("is_detected")),
            "max_y_is_detected": _json_default(max_row.get("is_detected")),
        },
    }


def inspect_time_continuity(
    metrics: pd.DataFrame,
    row: pd.Series,
    bound: float,
    window: int = DEFAULT_NEIGHBOR_WINDOW,
) -> Dict[str, Any]:
    match_id = str(row["match_id"])
    frame = safe_int(row["frame"])
    period = safe_int(row["period"])
    team_id = row["team_id"]

    sub = metrics[
        (metrics["match_id"].astype(str) == match_id)
        & (metrics["team_id"].astype(str) == str(team_id))
        & (metrics["period"] == period)
        & (metrics["frame"].between(frame - window, frame + window))
    ].copy()
    sub = sub.sort_values("frame")

    if sub.empty:
        return {
            "match_id": match_id,
            "frame": frame,
            "period": period,
            "team_id": str(team_id),
            "error": "No neighboring frame rows found.",
        }

    current_width = safe_float(row.get("team_width"), np.nan)
    violation_frames = sub.loc[sub["team_width"] > bound, "frame"].tolist()
    one_frame_spike = bool(len(violation_frames) == 1 and frame in violation_frames)
    persists = bool(len(violation_frames) >= 3)

    hull_median = safe_float(sub["convex_hull_area"].median(), np.nan) if "convex_hull_area" in sub else np.nan
    avg_dist_median = safe_float(sub["avg_distance_to_centroid"].median(), np.nan) if "avg_distance_to_centroid" in sub else np.nan
    current_hull = safe_float(row.get("convex_hull_area"), np.nan)
    current_avg_dist = safe_float(row.get("avg_distance_to_centroid"), np.nan)

    related_metrics_spike = False
    if hull_median and current_hull and hull_median > 0 and current_hull > hull_median * 1.5:
        related_metrics_spike = True
    if avg_dist_median and current_avg_dist and avg_dist_median > 0 and current_avg_dist > avg_dist_median * 1.5:
        related_metrics_spike = True

    return {
        "match_id": match_id,
        "frame": frame,
        "period": period,
        "team_id": str(team_id),
        "window_frames_checked": [int(sub["frame"].min()), int(sub["frame"].max())],
        "neighbor_row_count": int(len(sub)),
        "violation_frame_count_in_window": int(len(violation_frames)),
        "violation_frames_in_window": [int(v) for v in violation_frames],
        "one_frame_spike": one_frame_spike,
        "persists_across_several_frames": persists,
        "current_team_width": safe_float(current_width),
        "median_team_width_in_window": safe_float(sub["team_width"].median()),
        "max_team_width_in_window": safe_float(sub["team_width"].max()),
        "current_convex_hull_area": current_hull,
        "median_convex_hull_area_in_window": hull_median,
        "current_avg_distance_to_centroid": current_avg_dist,
        "median_avg_distance_to_centroid_in_window": avg_dist_median,
        "related_metrics_also_spike": related_metrics_spike,
    }


def summarize_affected(values: pd.Series, limit: int = 20) -> List[Any]:
    vals = sorted([v for v in values.dropna().unique().tolist()], key=lambda x: str(x))
    return vals[:limit]


def summarize_phases(violations: pd.DataFrame) -> Dict[str, List[Any]]:
    result: Dict[str, List[Any]] = {}
    for col in ["team_in_possession_phase_type", "team_out_of_possession_phase_type", "possession_status_for_team"]:
        if col in violations.columns:
            result[col] = summarize_affected(violations[col])
    return result


def check_phase_segment_distortion(phase_segments: pd.DataFrame, violations: pd.DataFrame) -> Dict[str, Any]:
    if phase_segments.empty or violations.empty:
        return {
            "available": bool(not phase_segments.empty),
            "flagged_phase_segments_found": 0,
            "notes": "No phase-segment distortion check performed because phase segments or violations are empty.",
        }

    needed = ["provider", "match_id", "period", "team_id", "phase_index"]
    if not all(col in phase_segments.columns for col in needed) or not all(col in violations.columns for col in needed):
        return {
            "available": False,
            "notes": "Required phase segment key columns missing.",
        }

    keys = violations[needed].drop_duplicates()
    merged = phase_segments.merge(keys, on=needed, how="inner")

    summary_cols = [c for c in phase_segments.columns if "team_width" in c]
    top_rows = merged.sort_values(summary_cols[0], ascending=False).head(10) if summary_cols else merged.head(10)

    return {
        "available": True,
        "flagged_phase_segments_found": int(len(merged)),
        "team_width_summary_columns_found": summary_cols,
        "top_flagged_phase_segments": top_rows[needed + summary_cols].to_dict(orient="records") if summary_cols else top_rows[needed].to_dict(orient="records"),
        "notes": "Flagged frame rows were matched to containing phase-segment summary keys when phase_index was available.",
    }


def classify_review(
    total_rows: int,
    violations: pd.DataFrame,
    bound: float,
    raw_inspections: List[Dict[str, Any]],
    continuity_inspections: List[Dict[str, Any]],
) -> Tuple[str, str, str, str]:
    if violations.empty:
        return "valid", "true edge/outlier behavior", "allow team_width normally", "no code change needed"

    violation_pct = len(violations) / total_rows * 100.0 if total_rows else 0.0
    max_width = float(violations["team_width"].max())
    max_ratio = max_width / bound if bound else np.inf
    outside_count = sum(int(item.get("outside_expected_pitch_y_count", 0) or 0) for item in raw_inspections)
    one_player_count = sum(bool(item.get("one_player_driving_warning", False)) for item in raw_inspections)
    one_frame_spikes = sum(bool(item.get("one_frame_spike", False)) for item in continuity_inspections)
    persistent = sum(bool(item.get("persists_across_several_frames", False)) for item in continuity_inspections)

    # Conservative classification rules.
    if max_ratio >= 1.25 or violation_pct > 1.0:
        return "unsafe for Layer 3", "unresolved", "exclude from Layer 3 v0.1 tactical claims", "keep diagnostic only"

    if outside_count > 0 or one_player_count > 0 or one_frame_spikes > 0:
        return (
            "valid with filtering/tolerance rules",
            "broadcast/extrapolated tracking artifact" if outside_count > 0 else "tolerance issue",
            "allow only after filtering/tolerance rule",
            "add filtering/tolerance rule",
        )

    if violation_pct <= 0.1 and max_ratio <= 1.05:
        return "valid with outlier flags", "tolerance issue", "allow with flags", "add warning flag"

    if persistent > 0 and max_ratio <= 1.10:
        return "valid with outlier flags", "true edge/outlier behavior", "allow with flags", "add warning flag"

    return "valid with outlier flags", "unresolved", "allow with flags", "add warning flag"


def create_handoff_addendum(review: Dict[str, Any]) -> str:
    summary = review["warning_summary"]
    return f"""Layer 2 v0.2.1 — team_width Warning Review Result

Status decision: {review['status_decision']}

Evidence summary:
- Total frame-team rows reviewed: {summary['total_frame_team_rows']:,}
- Violating rows: {summary['violating_rows']:,}
- Violating row percentage: {summary['violating_row_percentage']:.6f}%
- Pitch-width bound used: {summary['pitch_width_bound_used']:.3f}
- Max team_width: {summary['max_team_width']:.3f}
- Median team_width: {summary['median_team_width']:.3f}
- Affected matches: {summary['affected_matches']}
- Affected teams: {summary['affected_teams']}
- Affected periods: {summary['affected_periods']}
- Affected phases: {summary['affected_phases']}

Cause assessment: {review['cause_assessment']}

Layer 3 recommendation: {review['layer3_recommendation']}

Implementation recommendation: {review['implementation_recommendation']}

Visual evidence:
- Existing layer2_sanity plots found: {len(review['visual_evidence']['existing_layer2_sanity_plots'])}
- Exact team_width review plots created: {len(review['visual_evidence']['created_exact_outlier_plots'])}
- Plot output folder: {review['visual_evidence']['review_plot_folder']}

Code or documentation changes made:
- No metric definition was changed by this review script.
- No Layer 3 outputs were created.
- The review package was written to {REVIEW_JSON_PATH.as_posix()} and {REVIEW_CSV_PATH.as_posix()}.

Remaining limitations:
- Human visual review of the generated exact outlier plots is still recommended before final Lead Developer acceptance.
- If the Lead Developer decides a filtering/tolerance rule is needed, that rule should be implemented deliberately and tested before Layer 3 uses team_width.
"""


def run_review() -> Dict[str, Any]:
    require_file(FRAME_METRICS_PATH)
    require_file(PHASE_SEGMENTS_PATH)
    require_file(FOOTBALL_SANITY_JSON_PATH)
    require_file(FOOTBALL_SANITY_CSV_PATH)

    REVIEW_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(FRAME_METRICS_PATH)
    phase_segments = pd.read_csv(PHASE_SEGMENTS_PATH)
    sanity_json = read_json(FOOTBALL_SANITY_JSON_PATH)
    sanity_csv = pd.read_csv(FOOTBALL_SANITY_CSV_PATH)

    match_ids = [str(m) for m in metrics["match_id"].dropna().unique().tolist()]
    pitch_dimensions = {match_id: load_pitch_dimensions(match_id) for match_id in match_ids}
    bound = infer_pitch_bound(sanity_json, pitch_dimensions)

    violations = metrics[metrics["team_width"] > bound].copy()
    worst = metrics.sort_values("team_width", ascending=False).head(DEFAULT_WORST_N).copy()
    worst_violations = violations.sort_values("team_width", ascending=False).head(DEFAULT_WORST_N).copy()
    worst_examples = worst_violations if not worst_violations.empty else worst

    total_rows = int(len(metrics))
    violation_count = int(len(violations))
    violation_pct = float(violation_count / total_rows * 100.0) if total_rows else 0.0

    raw_inspections: List[Dict[str, Any]] = []
    continuity_inspections: List[Dict[str, Any]] = []
    created_plots: List[str] = []

    for _, row in worst_examples.iterrows():
        match_id = str(row["match_id"])
        dimensions = pitch_dimensions.get(match_id, PitchDimensions())
        raw = inspect_raw_coordinates(row, dimensions)
        raw_inspections.append(raw)
        continuity = inspect_time_continuity(metrics, row, bound=bound)
        continuity_inspections.append(continuity)

    # Create exact validation plots for top flagged examples only.
    plot_rows = worst_violations.head(DEFAULT_PLOT_N) if not worst_violations.empty else worst.head(DEFAULT_PLOT_N)
    tracking_cache: Dict[str, pd.DataFrame] = {}
    for _, row in plot_rows.iterrows():
        match_id = str(row["match_id"])
        tracking_path = SKILLCORNER_DIR / match_id / "canonical_tracking_players.csv"
        if not tracking_path.exists():
            continue
        if match_id not in tracking_cache:
            tracking_cache[match_id] = pd.read_csv(tracking_path)
        frame = safe_int(row["frame"])
        period = safe_int(row["period"])
        team_id = row["team_id"]
        dimensions = pitch_dimensions.get(match_id, PitchDimensions())
        output_path = REVIEW_FIGURES_DIR / f"team_width_review_match_{match_id}_frame_{frame}_team_{team_id}.png"
        create_outlier_plot(
            match_id=match_id,
            frame=frame,
            period=period,
            target_team_id=team_id,
            frame_metrics_row=row,
            tracking_players=tracking_cache[match_id],
            dimensions=dimensions,
            output_path=output_path,
        )
        created_plots.append(str(output_path))

    existing_plots = sorted(str(p) for p in FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []

    status_decision, cause, layer3_rec, implementation_rec = classify_review(
        total_rows=total_rows,
        violations=violations,
        bound=bound,
        raw_inspections=raw_inspections,
        continuity_inspections=continuity_inspections,
    )

    worst_columns = [
        "match_id",
        "frame",
        "period",
        "team_id",
        "phase_index",
        "team_in_possession_phase_type",
        "team_out_of_possession_phase_type",
        "possession_status_for_team",
        "team_width",
        "team_depth",
        "convex_hull_area",
        "avg_distance_to_centroid",
        "player_count_used",
        "detected_player_count",
        "extrapolated_player_count",
        "metric_warning_flag",
    ]
    available_worst_columns = [c for c in worst_columns if c in worst_examples.columns]
    worst_examples_output = worst_examples[available_worst_columns].copy()
    worst_examples_output["notes"] = np.where(
        worst_examples_output["team_width"] > bound,
        "team_width exceeds pitch-width bound",
        "top team_width row but does not exceed inferred bound",
    )
    worst_examples_output.to_csv(REVIEW_CSV_PATH, index=False)

    phase_distortion = check_phase_segment_distortion(phase_segments, violations)

    warning_summary = {
        "total_frame_team_rows": total_rows,
        "violating_rows": violation_count,
        "violating_row_percentage": violation_pct,
        "max_team_width": safe_float(metrics["team_width"].max()),
        "median_team_width": safe_float(metrics["team_width"].median()),
        "pitch_width_bound_used": safe_float(bound),
        "affected_matches": summarize_affected(violations["match_id"]) if not violations.empty else [],
        "affected_teams": summarize_affected(violations["team_id"]) if not violations.empty else [],
        "affected_periods": summarize_affected(violations["period"]) if not violations.empty else [],
        "affected_phases": summarize_phases(violations) if not violations.empty else {},
    }

    review = {
        "review_title": "Layer 2 v0.2.1 — team_width Warning Review",
        "status_decision": status_decision,
        "warning_summary": warning_summary,
        "worst_examples_table_path": str(REVIEW_CSV_PATH),
        "worst_examples": worst_examples_output.to_dict(orient="records"),
        "raw_coordinate_inspection": raw_inspections,
        "time_continuity_inspection": continuity_inspections,
        "phase_segment_distortion_check": phase_distortion,
        "visual_evidence": {
            "existing_layer2_sanity_plots": existing_plots,
            "created_exact_outlier_plots": created_plots,
            "review_plot_folder": str(REVIEW_FIGURES_DIR),
            "note": "Exact outlier plots were generated for local visual review. Human/Lead Developer visual confirmation is still recommended.",
        },
        "cause_assessment": cause,
        "layer3_recommendation": layer3_rec,
        "implementation_recommendation": implementation_rec,
        "source_files_used": {
            "football_sanity_json": str(FOOTBALL_SANITY_JSON_PATH),
            "football_sanity_csv": str(FOOTBALL_SANITY_CSV_PATH),
            "frame_team_shape_metrics": str(FRAME_METRICS_PATH),
            "phase_segment_summaries": str(PHASE_SEGMENTS_PATH),
            "skillcorner_tracking_root": str(SKILLCORNER_DIR),
            "layer2_sanity_figures": str(FIGURES_DIR),
        },
        "scope_guard": {
            "layer3_started": False,
            "tactical_identity_profiles_created": False,
            "metric_definition_changed": False,
            "team_width_used_for_tactical_claims": False,
        },
    }

    addendum = create_handoff_addendum(review)
    review["handoff_addendum_text"] = addendum

    write_json(review, REVIEW_JSON_PATH)
    ADDENDUM_TXT_PATH.write_text(addendum, encoding="utf-8")

    return review


def print_summary(review: Dict[str, Any]) -> None:
    summary = review["warning_summary"]
    print("Layer 2 v0.2.1 team_width Warning Review")
    print(f"Status decision: {review['status_decision']}")
    print(f"Cause assessment: {review['cause_assessment']}")
    print(f"Layer 3 recommendation: {review['layer3_recommendation']}")
    print(f"Implementation recommendation: {review['implementation_recommendation']}")
    print("")
    print("Warning summary:")
    print(f"- total frame-team rows: {summary['total_frame_team_rows']:,}")
    print(f"- violating rows: {summary['violating_rows']:,}")
    print(f"- violating percentage: {summary['violating_row_percentage']:.6f}%")
    print(f"- max team_width: {summary['max_team_width']:.3f}")
    print(f"- median team_width: {summary['median_team_width']:.3f}")
    print(f"- pitch-width bound used: {summary['pitch_width_bound_used']:.3f}")
    print(f"- affected matches: {summary['affected_matches']}")
    print(f"- affected teams: {summary['affected_teams']}")
    print(f"- affected periods: {summary['affected_periods']}")
    print(f"- affected phases: {summary['affected_phases']}")
    print("")
    print("Outputs written:")
    print(f"- review JSON: {REVIEW_JSON_PATH}")
    print(f"- worst examples CSV: {REVIEW_CSV_PATH}")
    print(f"- handoff addendum TXT: {ADDENDUM_TXT_PATH}")
    print(f"- exact review plots folder: {REVIEW_FIGURES_DIR}")


if __name__ == "__main__":
    review_package = run_review()
    print_summary(review_package)
