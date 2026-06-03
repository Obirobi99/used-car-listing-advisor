import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _env_optional_positive_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"none", "all", "unlimited", "0"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer or one of none/all/unlimited/0.") from exc
    if parsed <= 0:
        return None
    return parsed


RANDOM_STATE = 42
TEST_SIZE = 0.2
CURRENT_YEAR = 2026

DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
FIGURE_DIR = Path("reports/figures")

CRAIGSLIST_RAW_FILE = DATA_RAW_DIR / "kaggle_craigslist_vehicles.csv"
CARSCOM_RAW_FILE = DATA_RAW_DIR / "kaggle_carscom_used_cars.csv"

PROCESSED_FILE = DATA_PROCESSED_DIR / "project_listings.csv"

STRUCTURED_MODEL_FILE = MODEL_DIR / "price_model_structured.joblib"
INTEGRATED_MODEL_FILE = MODEL_DIR / "price_model_integrated.joblib"
TEXT_VECTORIZER_FILE = MODEL_DIR / "text_vectorizer.joblib"
TEXT_RISK_MODEL_FILE = MODEL_DIR / "text_risk_model.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"

CURRENCY_DEFAULT = "model currency units"

# High-performance VM defaults: use all available CPU cores and full feature/data
# capacity unless an environment variable intentionally caps a setting.
TRAIN_N_JOBS = _env_int("TRAIN_N_JOBS", -1)
MAX_TRAIN_ROWS = _env_optional_positive_int("MAX_TRAIN_ROWS")
ONE_HOT_MAX_CATEGORIES = _env_optional_positive_int("ONE_HOT_MAX_CATEGORIES")
HIGH_CARDINALITY_MAX_CATEGORIES = _env_optional_positive_int("HIGH_CARDINALITY_MAX_CATEGORIES")
TFIDF_MAX_FEATURES = _env_optional_positive_int("TFIDF_MAX_FEATURES")
RANDOM_FOREST_N_ESTIMATORS = _env_int("RANDOM_FOREST_N_ESTIMATORS", 300)
LOGISTIC_REGRESSION_MAX_ITER = _env_int("LOGISTIC_REGRESSION_MAX_ITER", 2000)
