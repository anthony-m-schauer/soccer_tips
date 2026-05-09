"""
Script: common.py
Project: Soccer TIPS

Overview:
    Shared lightweight I/O helpers for Layer 1 ingestion modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

def ensure_directory(path: Path | str) -> Path:
    """Create a directory if needed and return it as a Path."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def require_file(path: Path | str, label: str | None = None) -> Path:
    """Return path if it exists, otherwise raise FileNotFoundError."""
    file_path = Path(path)
    if not file_path.exists():
        label_text = f" for {label}" if label else ""
        raise FileNotFoundError(f"Missing required file{label_text}: {file_path}")
    return file_path


def file_exists(path: Path | str) -> bool:
    """Return True if path exists."""
    return Path(path).exists()


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------

def _json_default(value: Any) -> Any:
    """JSON serializer fallback for pathlib, numpy, and pandas objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path | str) -> Any:
    """Read JSON from disk."""
    file_path = require_file(path)
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Any, path: Path | str, indent: int = 2) -> Path:
    """Write JSON to disk and return the output path."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, default=_json_default)
    return output_path


# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------

def read_csv(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Read CSV from disk after checking file existence."""
    file_path = require_file(path)
    return pd.read_csv(file_path, **kwargs)


def write_csv(df: pd.DataFrame, path: Path | str, index: bool = False, **kwargs: Any) -> Path:
    """Write dataframe to CSV and return the output path."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=index, **kwargs)
    return output_path


def read_optional_csv(path: Path | str, **kwargs: Any) -> pd.DataFrame | None:
    """Read a CSV if it exists; otherwise return None."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    return pd.read_csv(file_path, **kwargs)
