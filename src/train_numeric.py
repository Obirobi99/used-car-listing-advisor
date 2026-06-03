import os
import math
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("reports") / ".matplotlib"))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    FIGURE_DIR,
    MAX_TRAIN_ROWS,
    METADATA_FILE,
    PROCESSED_FILE,
    RANDOM_STATE,
    RANDOM_FOREST_N_ESTIMATORS,
    REPORT_DIR,
    STRUCTURED_MODEL_FILE,
    TEST_SIZE,
    TRAIN_N_JOBS,
)
from src.data_loading import load_and_prepare_data
from src.preprocessing_numeric import build_preprocessor, prepare_features_and_target
from src.utils import ensure_directories, load_json, print_section, save_json


def _load_processed_data() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        print("Processed file not found. Running data preparation first.")
        return load_and_prepare_data()
    return pd.read_csv(PROCESSED_FILE)


def _maybe_sample(df: pd.DataFrame) -> pd.DataFrame:
    if MAX_TRAIN_ROWS is not None and len(df) > MAX_TRAIN_ROWS:
        print(f"Sampling {MAX_TRAIN_ROWS} rows from {len(df)} rows because MAX_TRAIN_ROWS is set.")
        return df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE).copy()
    return df.copy()


def _regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _save_histogram(series, title: str, xlabel: str, path: Path, bins: int = 40) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=bins, color="#3b82f6", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_eda_plots(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _save_histogram(df["seller_price"], "Seller Price Distribution", "Seller price", FIGURE_DIR / "price_distribution.png")
    _save_histogram(df["mileage_km"], "Mileage Distribution", "Mileage (km)", FIGURE_DIR / "mileage_distribution.png")
    _save_histogram(df["car_age"], "Car Age Distribution", "Car age", FIGURE_DIR / "car_age_distribution.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(df["mileage_km"], df["seller_price"], alpha=0.35, s=14, color="#0f766e")
    plt.title("Mileage vs Price")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Seller price")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mileage_vs_price.png")
    plt.close()

    if "brand" in df.columns:
        top_brands = df["brand"].value_counts().head(10).index
        brand_prices = df[df["brand"].isin(top_brands)].groupby("brand")["seller_price"].mean().sort_values()
        if not brand_prices.empty:
            plt.figure(figsize=(9, 5))
            brand_prices.plot(kind="barh", color="#f59e0b")
            plt.title("Average Price by Top Brands")
            plt.xlabel("Average seller price")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "average_price_by_brand_top10.png")
            plt.close()


def _save_evaluation_plots(y_test, predictions, metrics_by_model: dict) -> None:
    best_name = min(metrics_by_model, key=lambda name: metrics_by_model[name]["RMSE"])
    best_pred = predictions[best_name]
    residuals = y_test - best_pred

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_pred, alpha=0.55, color="#2563eb")
    min_value = min(y_test.min(), best_pred.min())
    max_value = max(y_test.max(), best_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="#111827", linestyle="--")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(best_pred, residuals, alpha=0.55, color="#dc2626")
    plt.axhline(0, color="#111827", linestyle="--")
    plt.title(f"Residual Plot ({best_name})")
    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residual_plot.png")
    plt.close()

    comparison = pd.DataFrame(metrics_by_model).T.sort_values("RMSE")
    plt.figure(figsize=(8, 5))
    comparison["RMSE"].plot(kind="bar", color="#10b981")
    plt.title("Model Comparison by RMSE")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison_rmse.png")
    plt.close()


def _candidate_models() -> dict:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=TRAIN_N_JOBS,
            min_samples_leaf=3,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=TRAIN_N_JOBS,
            min_samples_leaf=3,
        ),
    }


def train_numeric_models() -> dict:
    ensure_directories()
    print_section("Structured Numeric Model Training")
    raw_df = _maybe_sample(_load_processed_data())
    X, y, _, engineered_df = prepare_features_and_target(raw_df, include_nlp_features=False)

    if len(X) < 8:
        raise ValueError("At least 8 valid rows are required to train numeric models.")

    _save_eda_plots(engineered_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    metrics_by_model = {}
    predictions = {}
    fitted_models = {}
    for name, estimator in _candidate_models().items():
        print(f"Training {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(engineered_df, include_nlp_features=False)),
                ("model", estimator),
            ]
        )
        try:
            pipeline.fit(X_train, y_train)
            pred = np.clip(pipeline.predict(X_test), 0, None)
            metrics_by_model[name] = _regression_metrics(y_test, pred)
            predictions[name] = pred
            fitted_models[name] = pipeline
            print(f"{name}: {metrics_by_model[name]}")
        except Exception as exc:
            metrics_by_model[name] = {"error": str(exc)}
            print(f"WARNING: {name} failed and was skipped: {exc}")

    valid_models = {name: metrics for name, metrics in metrics_by_model.items() if "RMSE" in metrics}
    if not valid_models:
        raise ValueError("No numeric model trained successfully.")

    best_name = min(valid_models, key=lambda name: valid_models[name]["RMSE"])
    best_model = fitted_models[best_name]
    STRUCTURED_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, STRUCTURED_MODEL_FILE)
    print(f"Saved best structured model ({best_name}) to {STRUCTURED_MODEL_FILE}")

    _save_evaluation_plots(y_test, predictions, valid_models)

    sample = engineered_df.loc[X_test.index].copy()
    sample_predictions = pd.DataFrame(
        {
            "actual_price": y_test,
            "predicted_price": predictions[best_name],
            "absolute_error": np.abs(y_test - predictions[best_name]),
            "percentage_error": np.abs(y_test - predictions[best_name]) / y_test * 100,
            "brand": sample["brand"],
            "model": sample["model"],
            "year": sample["year"],
            "mileage_km": sample["mileage_km"],
            "source_name": sample["source_name"],
        }
    ).sort_values("absolute_error", ascending=False)
    sample_predictions.to_csv(REPORT_DIR / "sample_predictions.csv", index=False)

    metrics = {
        "dataset_shape_summary": {"rows": int(len(engineered_df)), "columns": int(engineered_df.shape[1])},
        "column_summary": list(engineered_df.columns),
        "missing_value_summary": engineered_df.isna().sum().to_dict(),
        "source_distribution": engineered_df["source_name"].value_counts().to_dict(),
        "model_metrics": metrics_by_model,
        "best_model": best_name,
        "best_metrics": valid_models[best_name],
        "error_analysis": {
            "top_prediction_errors": sample_predictions.head(10).to_dict(orient="records"),
            "likely_causes": [
                "rare brands or models with few comparable listings",
                "unrealistic or strategic seller prices",
                "incomplete condition and title information",
                "cross-source differences between Craigslist and Cars.com rows",
                "mileage outliers",
                "missing seller descriptions for secondary-source rows",
                "older vehicles with collector or poor-condition effects",
                "luxury vehicles where options and trim matter but are not fully modeled",
            ],
        },
        "figures": [
            "reports/figures/price_distribution.png",
            "reports/figures/mileage_distribution.png",
            "reports/figures/car_age_distribution.png",
            "reports/figures/mileage_vs_price.png",
            "reports/figures/average_price_by_brand_top10.png",
            "reports/figures/model_comparison_rmse.png",
            "reports/figures/actual_vs_predicted.png",
            "reports/figures/residual_plot.png",
        ],
    }
    save_json(metrics, REPORT_DIR / "metrics_numeric.json")

    metadata = load_json(METADATA_FILE) if METADATA_FILE.exists() else {}
    metadata.update(
        {
            "structured_model_file": str(STRUCTURED_MODEL_FILE),
            "structured_best_model": best_name,
            "structured_best_metrics": valid_models[best_name],
        }
    )
    save_json(metadata, METADATA_FILE)
    return metrics


if __name__ == "__main__":
    train_numeric_models()
