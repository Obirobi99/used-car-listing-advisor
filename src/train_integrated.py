import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    INTEGRATED_MODEL_FILE,
    METADATA_FILE,
    PROCESSED_FILE,
    RANDOM_STATE,
    RANDOM_FOREST_N_ESTIMATORS,
    REPORT_DIR,
    TEST_SIZE,
    TRAIN_N_JOBS,
)
from src.data_loading import load_and_prepare_data
from src.nlp_features import add_nlp_features
from src.preprocessing_numeric import (
    build_preprocessor,
    clean_numeric_data,
    engineer_numeric_features,
    get_categorical_feature_columns,
    get_numeric_feature_columns,
)
from src.utils import ensure_directories, load_json, print_section, save_json


def _load_processed_data() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        print("Processed file not found. Running data preparation first.")
        return load_and_prepare_data()
    return pd.read_csv(PROCESSED_FILE)


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _feature_frame(df: pd.DataFrame, include_nlp_features: bool):
    feature_columns = get_numeric_feature_columns(include_nlp_features) + get_categorical_feature_columns(
        include_nlp_features
    )
    feature_columns = [column for column in feature_columns if column in df.columns]
    return df[feature_columns].copy()


def _candidate_models() -> dict:
    return {
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


def _train_group(X_train, X_test, y_train, y_test, base_df, include_nlp_features: bool) -> tuple[dict, dict]:
    metrics_by_model = {}
    fitted = {}
    for name, estimator in _candidate_models().items():
        print(f"Training {'integrated' if include_nlp_features else 'structured'} {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(base_df, include_nlp_features=include_nlp_features)),
                ("model", estimator),
            ]
        )
        try:
            pipeline.fit(X_train, y_train)
            pred = np.clip(pipeline.predict(X_test), 0, None)
            metrics_by_model[name] = _metrics(y_test, pred)
            fitted[name] = pipeline
        except Exception as exc:
            metrics_by_model[name] = {"error": str(exc)}
            print(f"WARNING: {name} failed: {exc}")
    return metrics_by_model, fitted


def train_integrated_models() -> dict:
    ensure_directories()
    print_section("Integrated Structured + NLP Model Training")
    df = add_nlp_features(_load_processed_data())
    cleaned = engineer_numeric_features(clean_numeric_data(df))

    if len(cleaned) < 8:
        raise ValueError("At least 8 valid rows are required to train integrated models.")

    y = cleaned["seller_price"].copy()
    X_structured = _feature_frame(cleaned, include_nlp_features=False)
    X_integrated = _feature_frame(cleaned, include_nlp_features=True)

    indices_train, indices_test = train_test_split(
        cleaned.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_train = y.loc[indices_train]
    y_test = y.loc[indices_test]

    structured_metrics, _ = _train_group(
        X_structured.loc[indices_train],
        X_structured.loc[indices_test],
        y_train,
        y_test,
        cleaned,
        include_nlp_features=False,
    )
    integrated_metrics, integrated_models = _train_group(
        X_integrated.loc[indices_train],
        X_integrated.loc[indices_test],
        y_train,
        y_test,
        cleaned,
        include_nlp_features=True,
    )

    valid_integrated = {name: metrics for name, metrics in integrated_metrics.items() if "RMSE" in metrics}
    valid_structured = {name: metrics for name, metrics in structured_metrics.items() if "RMSE" in metrics}
    if not valid_integrated:
        raise ValueError("No integrated model trained successfully.")

    best_integrated_name = min(valid_integrated, key=lambda name: valid_integrated[name]["RMSE"])
    joblib.dump(integrated_models[best_integrated_name], INTEGRATED_MODEL_FILE)
    print(f"Saved best integrated model ({best_integrated_name}) to {INTEGRATED_MODEL_FILE}")

    best_structured_rmse = min((m["RMSE"] for m in valid_structured.values()), default=None)
    best_integrated_rmse = valid_integrated[best_integrated_name]["RMSE"]
    if best_structured_rmse is not None and best_integrated_rmse < best_structured_rmse:
        interpretation = "Integrated NLP features improved RMSE on this split."
        improvement = best_structured_rmse - best_integrated_rmse
    else:
        interpretation = (
            "Integrated NLP features did not improve RMSE on this split, but they still improve "
            "interpretability and risk-aware recommendation logic."
        )
        improvement = 0.0 if best_structured_rmse is None else best_structured_rmse - best_integrated_rmse

    metrics = {
        "structured_only_model_metrics": structured_metrics,
        "structured_plus_nlp_model_metrics": integrated_metrics,
        "best_integrated_model": best_integrated_name,
        "best_integrated_metrics": valid_integrated[best_integrated_name],
        "rmse_difference_structured_minus_integrated": float(improvement),
        "interpretation": interpretation,
        "nlp_features_used": [
            "description_length_words",
            "risk_keyword_count",
            "positive_keyword_count",
            "text_risk_score",
            "nlp_risk_label",
        ],
    }
    save_json(metrics, REPORT_DIR / "metrics_integrated.json")

    metadata = load_json(METADATA_FILE) if METADATA_FILE.exists() else {}
    metadata.update(
        {
            "integrated_model_file": str(INTEGRATED_MODEL_FILE),
            "integrated_best_model": best_integrated_name,
            "integrated_best_metrics": valid_integrated[best_integrated_name],
        }
    )
    save_json(metadata, METADATA_FILE)
    return metrics


if __name__ == "__main__":
    train_integrated_models()
