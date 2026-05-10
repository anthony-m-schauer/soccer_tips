"""
Script: qa_football_sanity.py
Project: Soccer TIPS

Overview:
    Command-line runner for Layer 2 v0.2 football sanity validation.

Usage from project root:
    $env:PYTHONPATH="src"
    python -m soccer_tips.features.qa_football_sanity

Outputs:
    data/processed/features/layer2_v02_football_sanity_summary.csv
    data/processed/features/layer2_v02_football_sanity_summary.json
    outputs/figures/layer2_sanity/*.png

Scope guard:
    This is a validation checkpoint. It does not add new tactical metrics,
    Layer 3 modeling, clustering, AI interpretation, dashboards, reports,
    recommendations, compactness interpretation, or transition-stability modeling.
"""

from __future__ import annotations

import argparse
from typing import Any

from soccer_tips.features import football_sanity
from soccer_tips.features import visualize_layer2


def run_football_sanity_qa(create_plots: bool = True, max_time_series_segments: int = 6) -> dict[str, Any]:
    """Run Layer 2 v0.2 football sanity QA and write outputs."""
    tables = football_sanity.load_layer2_feature_outputs()

    visual_outputs: list[dict[str, Any]] = []
    if create_plots:
        visual_outputs = visualize_layer2.create_all_validation_plots(
            frame_metrics=tables.frame_team_shape_metrics,
            phase_summaries=tables.phase_segment_summaries,
            max_time_series_segments=max_time_series_segments,
        )

    distribution_summary, json_report = football_sanity.build_summary_report(
        tables,
        visual_outputs=visual_outputs,
    )
    output_paths = football_sanity.write_football_sanity_outputs(distribution_summary, json_report)
    json_report["output_paths"] = {key: str(path) for key, path in output_paths.items()}
    football_sanity.write_football_sanity_outputs(distribution_summary, json_report)

    return json_report


def print_report_summary(report: dict[str, Any]) -> None:
    """Print a concise summary for terminal review."""
    print("Layer 2 v0.2 Football Sanity QA")
    print(f"Status: {report['status']}")
    print(f"Matches inspected: {len(report['matches_inspected'])}")
    print(f"Frame-team rows inspected: {report['frame_team_metric_rows_inspected']}")
    print(f"Phase segment rows inspected: {report['phase_segment_summary_rows_inspected']}")
    print(f"Match profile rows inspected: {report['match_tactical_profile_rows_inspected']}")
    print(f"Phases inspected: {', '.join(report['phases_inspected'])}")

    created_plots = [item for item in report["visual_checks_created"] if item.get("status") == "created"]
    print(f"Validation plots created: {len(created_plots)}")

    print("Reasonableness check statuses:")
    for check_name, check in report["reasonableness_checks"]["checks"].items():
        print(f"- {check_name}: {check.get('status')}")

    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")

    if report["failures"]:
        print("Failures:")
        for failure in report["failures"]:
            print(f"- {failure}")

    print("Output files:")
    for label, path in report.get("output_paths", {}).items():
        print(f"- {label}: {path}")
    print(f"- figures: {visualize_layer2.FIGURE_OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Soccer TIPS Layer 2 v0.2 football sanity validation.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run data-only football sanity checks without creating validation plots.",
    )
    parser.add_argument(
        "--max-time-series-segments",
        type=int,
        default=6,
        help="Maximum number of phase-segment time-series plots to create.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    report = run_football_sanity_qa(
        create_plots=not args.no_plots,
        max_time_series_segments=args.max_time_series_segments,
    )
    print_report_summary(report)


if __name__ == "__main__":
    main()
