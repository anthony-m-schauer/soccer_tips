"""
Script: config.py
Project: Soccer TIPS

Overview:
    Central path configuration for Layer 1 ingestion infrastructure.

Design notes:
    - Uses pathlib.Path only.
    - Avoids user-specific absolute paths.
    - Assumes the package lives under: <project_root>/src/soccer_tips/.
    - Raw provider datasets stay under external/.
    - Processed outputs are written under data/processed/.
"""

from pathlib import Path


# -----------------------------------------------------------------------------
# Project root
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Top-level project directories
# -----------------------------------------------------------------------------

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = PROJECT_ROOT / "external"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"


# -----------------------------------------------------------------------------
# Raw/external provider data locations
# -----------------------------------------------------------------------------

SKILLCORNER_DIR = EXTERNAL_DIR / "skillcorner"
SKILLCORNER_DATA_DIR = SKILLCORNER_DIR / "data"
SKILLCORNER_MATCHES_INDEX_PATH = SKILLCORNER_DATA_DIR / "matches.json"
SKILLCORNER_MATCHES_DIR = SKILLCORNER_DATA_DIR / "matches"

METRICA_DIR = EXTERNAL_DIR / "metrica"
METRICA_DATA_DIR = METRICA_DIR / "data"


# -----------------------------------------------------------------------------
# Processed output locations
# -----------------------------------------------------------------------------

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_SKILLCORNER_DIR = PROCESSED_DIR / "skillcorner"
PROCESSED_METRICA_DIR = PROCESSED_DIR / "metrica"
VALIDATION_REPORTS_DIR = PROCESSED_DIR / "validation_reports"


# -----------------------------------------------------------------------------
# Default pitch dimensions
# -----------------------------------------------------------------------------

# These are fallback values only. SkillCorner match metadata should be treated as
# authoritative when available.
DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0


# -----------------------------------------------------------------------------
# Provider constants
# -----------------------------------------------------------------------------

SKILLCORNER_PROVIDER = "skillcorner"
METRICA_PROVIDER = "metrica"

SKILLCORNER_COORDINATE_SYSTEM = "metric_centered"
METRICA_COORDINATE_SYSTEM = "normalized_0_1"

SKILLCORNER_FPS = 10
METRICA_FPS = 25


# -----------------------------------------------------------------------------
# Directory helpers
# -----------------------------------------------------------------------------

def get_skillcorner_match_dir(match_id: int | str) -> Path:
    """Return the raw SkillCorner match folder for a match id."""
    return SKILLCORNER_MATCHES_DIR / str(match_id)


def get_metrica_sample_game_dir(sample_game: int | str) -> Path:
    """Return the raw Metrica sample game folder.

    Accepts either:
        - 1
        - "1"
        - "Sample_Game_1"
    """
    sample_game_text = str(sample_game)
    if sample_game_text.startswith("Sample_Game_"):
        folder_name = sample_game_text
    else:
        folder_name = f"Sample_Game_{sample_game_text}"
    return METRICA_DATA_DIR / folder_name


def ensure_processed_directories() -> None:
    """Create the standard processed output directories if needed."""
    for path in (
        PROCESSED_DIR,
        PROCESSED_SKILLCORNER_DIR,
        PROCESSED_METRICA_DIR,
        VALIDATION_REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
