"""
Script: qa_layer1_outputs.py
Project: Soccer TIPS

Overview:
    Local QA utility for Layer 1 v0.1 processed outputs.

Purpose:
    - Inspect processed output files.
    - Report table shapes without loading full large CSVs into memory.
    - Confirm canonical player and ball tracking schema compliance.
    - Summarize validation report statuses and warnings.
    - Write a structured QA report JSON.

Usage from project root:
    $env:PYTHONPATH="src"
    python -m soccer_tips.data.qa_layer1_outputs

Scope:
    This utility performs Layer 1 QA only. It does not compute tactical metrics,
    compactness, transition stability, tactical identity models, dashboards, AI
    interpretation, or reporting-layer outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from soccer_tips import config
from soccer_tips import schemas
from soccer_tips.data import common


# -----------------------------------------------------------------------------
# Expected processed output names
# -----------------------------------------------------------------------------

SKILLCORNER_EXPECTED_OUTPUTS = {
    "tracking_players": "canonical_tracking_players.csv",
    "tracking_ball": "canonical_tracking_ball.csv",
    "match_metadata": "match_metadata.json",
    "players_metadata": "players_metadata.csv",
    "phases": "phases_of_play.csv",
    "dynamic_events": "dynamic_events.csv",  # optional; missing is allowed for known matches such as 1953632
}

METRICA_EXPECTED_OUTPUTS = {
    "tracking_players": "canonical_tracking_players.csv",
    "tracking_ball": "canonical_tracking_ball.csv",
    "events": "events.csv",
}

VALIDATION_REPORT_EXPECTED_KEYS = schemas.VALIDATION_REPORT_COLUMNS


@dataclass(frozen=True)
class CsvInspection:
    """Lightweight CSV inspection result."""

    rows: int
    columns: int
    column_names: list[str]


# -----------------------------------------------------------------------------
# Lightweight file inspection helpers
# -----------------------------------------------------------------------------

def _count_csv_data_rows(path: Path) -> int:
    """Count CSV data rows without loading the full file into memory."""
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8", errors="replace") as file:
        line_count = sum(1 for _ in file)

    return max(line_count - 1, 0)


def inspect_csv(path: Path) -> CsvInspection | None:
    """Return row/column counts and column names for a CSV, or None if missing."""
    if not path.exists():
        return None

    header_df = pd.read_csv(path, nrows=0)

    return CsvInspection(
        rows=_count_csv_data_rows(path),
        columns=len(header_df.columns),
        column_names=list(header_df.columns),
    )


def inspect_json(path: Path) -> dict[str, Any] | None:
    """Return basic JSON shape/key information, or None if missing."""
    if not path.exists():
        return None

    data = common.read_json(path)

    if isinstance(data, dict):
        return {
            "json_type": "dict",
            "shape": [1, len(data.keys())],
            "keys": list(data.keys()),
        }

    if isinstance(data, list):
        first_keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
        return {
            "json_type": "list",
            "shape": [len(data), len(first_keys)],
            "first_item_keys": first_keys,
        }

    return {
        "json_type": type(data).__name__,
        "shape": None,
        "keys": [],
    }


def _schema_check_from_columns(column_names: list[str], expected_columns: list[str]) -> dict[str, Any]:
    """Return schema compliance details from observed columns."""
    observed = list(column_names)
    expected = list(expected_columns)

    missing = [column for column in expected if column not in observed]
    extra = [column for column in observed if column not in expected]
    ordered_match = observed == expected

    return {
        "expected_columns": expected,
        "observed_columns": observed,
        "missing_columns": missing,
        "extra_columns": extra,
        "has_all_expected_columns": len(missing) == 0,
        "ordered_exact_match": ordered_match,
        "status": "pass" if len(missing) == 0 and len(extra) == 0 and ordered_match else "warning",
    }


def _schema_check_with_helper(path: Path, expected_columns: list[str]) -> dict[str, Any] | None:
    """Use schema helper functions against a header-only dataframe."""
    if not path.exists():
        return None

    header_df = pd.read_csv(path, nrows=0)

    missing = schemas.missing_columns(header_df, expected_columns)
    extra = [column for column in header_df.columns if column not in expected_columns]
    ordered_exact_match = list(header_df.columns) == expected_columns

    return {
        "expected_columns": list(expected_columns),
        "observed_columns": list(header_df.columns),
        "missing_columns": missing,
        "extra_columns": extra,
        "has_all_expected_columns": schemas.has_columns(header_df, expected_columns),
        "ordered_exact_match": ordered_exact_match,
        "status": "pass" if not missing and not extra and ordered_exact_match else "warning",
    }


# -----------------------------------------------------------------------------
# Provider/run inspection
# -----------------------------------------------------------------------------

def _inspect_table(path: Path, expected_columns: list[str] | None = None) -> dict[str, Any]:
    """Inspect a CSV table and optionally check it against a schema."""
    inspection = inspect_csv(path)

    if inspection is None:
        return {
            "exists": False,
            "path": str(path),
            "shape": None,
            "columns": [],
            "schema_check": None,
        }

    schema_check = None

    if expected_columns is not None:
        schema_check = _schema_check_with_helper(path, expected_columns)

    return {
        "exists": True,
        "path": str(path),
        "shape": [inspection.rows, inspection.columns],
        "columns": inspection.column_names,
        "schema_check": schema_check,
    }


def _read_validation_report(provider: str, match_id: str) -> dict[str, Any] | None:
    """Read one validation report if present."""
    report_path = config.VALIDATION_REPORTS_DIR / f"{provider}_{match_id}_validation.json"

    if not report_path.exists():
        return None

    return common.read_json(report_path)


def _inspect_validation_report(provider: str, match_id: str) -> dict[str, Any]:
    """Inspect validation report and top-level schema keys."""
    report_path = config.VALIDATION_REPORTS_DIR / f"{provider}_{match_id}_validation.json"
    json_info = inspect_json(report_path)
    report = _read_validation_report(provider, match_id)

    if report is None:
        return {
            "exists": False,
            "path": str(report_path),
            "shape": None,
            "status": None,
            "warnings": [],
            "errors": [],
            "schema_check": None,
        }

    observed_keys = list(report.keys())
    schema_check = _schema_check_from_columns(observed_keys, VALIDATION_REPORT_EXPECTED_KEYS)

    return {
        "exists": True,
        "path": str(report_path),
        "shape": json_info["shape"] if json_info else None,
        "status": report.get("status"),
        "warnings": report.get("warnings", []),
        "errors": report.get("errors", []),
        "schema_check": schema_check,
    }


def inspect_skillcorner_run(match_dir: Path) -> dict[str, Any]:
    """Inspect one processed SkillCorner match output directory."""
    match_id = match_dir.name
    output_files = sorted(path.name for path in match_dir.iterdir() if path.is_file())

    paths = {
        label: match_dir / filename
        for label, filename in SKILLCORNER_EXPECTED_OUTPUTS.items()
    }

    tables = {
        "tracking_players": _inspect_table(
            paths["tracking_players"],
            schemas.CANONICAL_TRACKING_PLAYER_COLUMNS,
        ),
        "tracking_ball": _inspect_table(
            paths["tracking_ball"],
            schemas.CANONICAL_TRACKING_BALL_COLUMNS,
        ),
        "players_metadata": _inspect_table(paths["players_metadata"]),
        "phases": _inspect_table(paths["phases"]),
        "dynamic_events": _inspect_table(paths["dynamic_events"]),
    }

    metadata_info = inspect_json(paths["match_metadata"])
    validation_report = _inspect_validation_report(config.SKILLCORNER_PROVIDER, match_id)

    return {
        "provider": config.SKILLCORNER_PROVIDER,
        "match_id": match_id,
        "output_dir": str(match_dir),
        "output_files": output_files,
        "tables": tables,
        "metadata": {
            "exists": paths["match_metadata"].exists(),
            "path": str(paths["match_metadata"]),
            **(metadata_info or {"shape": None, "keys": []}),
        },
        "validation_report": validation_report,
    }


def inspect_metrica_run(game_dir: Path) -> dict[str, Any]:
    """Inspect one processed Metrica sample-game output directory."""
    match_id = game_dir.name
    output_files = sorted(path.name for path in game_dir.iterdir() if path.is_file())

    paths = {
        label: game_dir / filename
        for label, filename in METRICA_EXPECTED_OUTPUTS.items()
    }

    tables = {
        "tracking_players": _inspect_table(
            paths["tracking_players"],
            schemas.CANONICAL_TRACKING_PLAYER_COLUMNS,
        ),
        "tracking_ball": _inspect_table(
            paths["tracking_ball"],
            schemas.CANONICAL_TRACKING_BALL_COLUMNS,
        ),
        "events": _inspect_table(paths["events"]),
    }

    validation_report = _inspect_validation_report(config.METRICA_PROVIDER, match_id)

    return {
        "provider": config.METRICA_PROVIDER,
        "match_id": match_id,
        "output_dir": str(game_dir),
        "output_files": output_files,
        "tables": tables,
        "validation_report": validation_report,
    }


# -----------------------------------------------------------------------------
# Summary helpers
# -----------------------------------------------------------------------------

def _iter_schema_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect all schema check dictionaries from the QA report."""
    checks: list[dict[str, Any]] = []

    for provider_runs in report["providers"].values():
        for run in provider_runs.values():
            for table in run.get("tables", {}).values():
                schema_check = table.get("schema_check")

                if schema_check is not None:
                    checks.append(schema_check)

            validation_schema = run.get("validation_report", {}).get("schema_check")

            if validation_schema is not None:
                checks.append(validation_schema)

    return checks


def _build_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Build compact QA summary from detailed provider/run report."""
    validation_statuses: dict[str, str | None] = {}
    warnings: dict[str, list[str]] = {}
    errors: dict[str, list[str]] = {}

    for provider_name, provider_runs in report["providers"].items():
        for match_id, run in provider_runs.items():
            key = f"{provider_name}:{match_id}"
            validation_report = run.get("validation_report", {})

            validation_statuses[key] = validation_report.get("status")
            warnings[key] = validation_report.get("warnings", [])
            errors[key] = validation_report.get("errors", [])

    schema_checks = _iter_schema_checks(report)
    schema_warnings = [
        check
        for check in schema_checks
        if check.get("status") != "pass"
    ]

    return {
        "run_count": sum(len(provider_runs) for provider_runs in report["providers"].values()),
        "validation_statuses": validation_statuses,
        "warnings": {
            key: value
            for key, value in warnings.items()
            if value
        },
        "errors": {
            key: value
            for key, value in errors.items()
            if value
        },
        "schema_compliance": {
            "checks_run": len(schema_checks),
            "warning_count": len(schema_warnings),
            "status": "pass" if not schema_warnings else "warning",
            "warnings": schema_warnings,
        },
    }


def build_layer1_qa_report() -> dict[str, Any]:
    """Inspect all currently available processed Layer 1 outputs."""
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_root": str(config.PROCESSED_DIR),
        "providers": {
            config.SKILLCORNER_PROVIDER: {},
            config.METRICA_PROVIDER: {},
        },
        "summary": {},
    }

    if config.PROCESSED_SKILLCORNER_DIR.exists():
        skillcorner_dirs = sorted(
            path
            for path in config.PROCESSED_SKILLCORNER_DIR.iterdir()
            if path.is_dir()
        )

        for match_dir in skillcorner_dirs:
            report["providers"][config.SKILLCORNER_PROVIDER][match_dir.name] = inspect_skillcorner_run(match_dir)

    if config.PROCESSED_METRICA_DIR.exists():
        metrica_dirs = sorted(
            path
            for path in config.PROCESSED_METRICA_DIR.iterdir()
            if path.is_dir()
        )

        for game_dir in metrica_dirs:
            report["providers"][config.METRICA_PROVIDER][game_dir.name] = inspect_metrica_run(game_dir)

    report["summary"] = _build_summary(report)

    return report


def write_layer1_qa_report(report: dict[str, Any]) -> Path:
    """Write structured Layer 1 QA report JSON."""
    output_path = config.VALIDATION_REPORTS_DIR / "layer1_v01_qa_summary.json"

    return common.write_json(report, output_path)


def print_layer1_qa_report(report: dict[str, Any]) -> None:
    """Print a concise, human-readable QA summary."""
    print("Layer 1 v0.1 QA report")
    print(f"Processed root: {report['processed_root']}")
    print(f"Runs inspected: {report['summary']['run_count']}")
    print(f"Schema compliance status: {report['summary']['schema_compliance']['status']}")
    print("")

    for provider_name, provider_runs in report["providers"].items():
        if not provider_runs:
            continue

        print(f"{provider_name}")

        for match_id, run in provider_runs.items():
            validation_status = run["validation_report"].get("status")

            print(f"- {match_id}: validation={validation_status}")
            print(f"  files: {', '.join(run['output_files'])}")

            for table_name, table in run.get("tables", {}).items():
                if not table.get("exists"):
                    print(f"  {table_name}: missing")
                    continue

                schema_status = None

                if table.get("schema_check"):
                    schema_status = table["schema_check"]["status"]

                schema_text = f", schema={schema_status}" if schema_status else ""

                print(f"  {table_name}: shape={tuple(table['shape'])}{schema_text}")

            if "metadata" in run:
                metadata = run["metadata"]
                print(f"  metadata: exists={metadata.get('exists')}, shape={metadata.get('shape')}")

            warnings = run["validation_report"].get("warnings", [])

            if warnings:
                print(f"  warnings: {warnings}")

        print("")

    output_path = config.VALIDATION_REPORTS_DIR / "layer1_v01_qa_summary.json"
    print(f"Structured QA report path: {output_path}")


def main() -> None:
    """Build, write, and print the Layer 1 v0.1 QA report."""
    config.ensure_processed_directories()

    report = build_layer1_qa_report()
    write_layer1_qa_report(report)
    print_layer1_qa_report(report)


if __name__ == "__main__":
    main()