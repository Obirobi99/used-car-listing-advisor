import math

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    INTEGRATED_MODEL_FILE,
    PROCESSED_FILE,
    REPORT_DIR,
    STRUCTURED_MODEL_FILE,
    TEXT_RISK_MODEL_FILE,
    TEXT_VECTORIZER_FILE,
)
from src.nlp_features import add_nlp_features
from src.preprocessing_numeric import clean_numeric_data, engineer_numeric_features
from src.utils import ensure_directories, print_section


EXPECTED_MODEL_FILES = [
    STRUCTURED_MODEL_FILE,
    INTEGRATED_MODEL_FILE,
    TEXT_VECTORIZER_FILE,
    TEXT_RISK_MODEL_FILE,
]

EXPECTED_REPORT_FILES = [
    REPORT_DIR / "metrics_numeric.json",
    REPORT_DIR / "metrics_nlp.json",
    REPORT_DIR / "metrics_integrated.json",
    REPORT_DIR / "sample_predictions.csv",
]


def _print_file_status(paths):
    for path in paths:
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{status}: {path}")


def evaluate_saved_models() -> None:
    ensure_directories()
    print_section("Saved Model Files")
    _print_file_status(EXPECTED_MODEL_FILES)

    print_section("Saved Report Files")
    _print_file_status(EXPECTED_REPORT_FILES)

    if not PROCESSED_FILE.exists():
        print(f"Processed data missing at {PROCESSED_FILE}. Run python -m src.data_loading first.")
        return

    if INTEGRATED_MODEL_FILE.exists():
        model_path = INTEGRATED_MODEL_FILE
        model_name = "integrated structured + NLP model"
    elif STRUCTURED_MODEL_FILE.exists():
        model_path = STRUCTURED_MODEL_FILE
        model_name = "structured-only model"
        print("Run python -m src.train_integrated to create the integrated model.")
    else:
        print("No price model found. Run python -m src.train_numeric first.")
        return

    df = pd.read_csv(PROCESSED_FILE)
    df = add_nlp_features(df)
    prepared = engineer_numeric_features(clean_numeric_data(df))
    if prepared.empty:
        print("No valid rows available for final evaluation.")
        return

    model = joblib.load(model_path)
    y_true = prepared["seller_price"]
    y_pred = np.clip(model.predict(prepared), 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(prepared) > 1 else float("nan")

    print_section("Final Evaluation Summary")
    print(f"Model evaluated: {model_name}")
    print(f"Rows evaluated: {len(prepared)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    sample = prepared.copy()
    sample["actual_price"] = y_true
    sample["predicted_price"] = y_pred
    sample["absolute_error"] = np.abs(y_true - y_pred)
    sample["percentage_error"] = sample["absolute_error"] / y_true * 100
    columns = [
        "actual_price",
        "predicted_price",
        "absolute_error",
        "percentage_error",
        "brand",
        "model",
        "year",
        "mileage_km",
        "source_name",
    ]
    sample.sort_values("absolute_error", ascending=False).head(200)[columns].to_csv(
        REPORT_DIR / "sample_predictions.csv", index=False
    )
    print(f"Updated {REPORT_DIR / 'sample_predictions.csv'}")


if __name__ == "__main__":
    evaluate_saved_models()

