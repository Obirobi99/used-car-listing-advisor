import json
from pathlib import Path

import numpy as np

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR


def ensure_directories() -> None:
    """Create all project directories used by scripts."""
    for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_json(data, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=_json_default)


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_lower(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()


def safe_float(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    text = (
        text.replace("$", "")
        .replace(",", "")
        .replace("USD", "")
        .replace("usd", "")
        .replace("mi.", "")
        .replace("miles", "")
        .replace("mile", "")
        .replace("km", "")
        .strip()
    )
    try:
        return float(text)
    except ValueError:
        return np.nan


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

