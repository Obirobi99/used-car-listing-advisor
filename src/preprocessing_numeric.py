import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CURRENT_YEAR,
    HIGH_CARDINALITY_MAX_CATEGORIES,
    ONE_HOT_MAX_CATEGORIES,
)
from src.utils import safe_lower


BASE_NUMERIC_FEATURES = [
    "year",
    "mileage_km",
    "car_age",
    "mileage_per_year",
    "log_mileage",
    "engine_size_l",
]

BASE_CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "body_type",
    "condition",
    "title_status",
    "drive",
    "paint_color",
    "location_region",
    "source_name",
]

NLP_NUMERIC_FEATURES = [
    "description_length_words",
    "risk_keyword_count",
    "positive_keyword_count",
    "text_risk_score",
]

NLP_CATEGORICAL_FEATURES = ["nlp_risk_label"]


def _make_one_hot_encoder():
    kwargs = {"handle_unknown": "ignore"}
    if ONE_HOT_MAX_CATEGORIES is not None:
        kwargs["max_categories"] = ONE_HOT_MAX_CATEGORIES

    try:
        return OneHotEncoder(**kwargs, sparse_output=True)
    except TypeError:
        try:
            return OneHotEncoder(**kwargs, sparse=True)
        except TypeError:
            kwargs.pop("max_categories", None)
            return OneHotEncoder(**kwargs)


def _limit_high_cardinality(df: pd.DataFrame, columns, max_categories: int | None = None) -> pd.DataFrame:
    result = df.copy()
    if max_categories is None:
        return result
    for column in columns:
        if column not in result.columns:
            continue
        counts = result[column].value_counts(dropna=False)
        if len(counts) > max_categories:
            keep = set(counts.head(max_categories).index)
            result[column] = result[column].where(result[column].isin(keep), "other")
    return result


def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_columns = ["seller_price", "year", "mileage_km", "engine_size_l"]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in BASE_CATEGORICAL_FEATURES + NLP_CATEGORICAL_FEATURES:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(safe_lower).replace("", "unknown")

    cleaned = cleaned[
        cleaned["seller_price"].notna()
        & (cleaned["seller_price"] > 0)
        & cleaned["mileage_km"].notna()
        & (cleaned["mileage_km"] >= 0)
        & cleaned["year"].notna()
        & (cleaned["year"] >= 1980)
        & (cleaned["year"] <= CURRENT_YEAR)
    ].copy()

    # Quantile filtering removes extreme listing artifacts without hard-coding market prices.
    # It is skipped on tiny demo data so the fallback remains trainable.
    if len(cleaned) >= 100:
        for column in ["seller_price", "mileage_km"]:
            lower = cleaned[column].quantile(0.01)
            upper = cleaned[column].quantile(0.99)
            cleaned = cleaned[cleaned[column].between(lower, upper)].copy()

    cleaned = _limit_high_cardinality(
        cleaned,
        ["model", "location_region"],
        max_categories=HIGH_CARDINALITY_MAX_CATEGORIES,
    )
    return cleaned


def engineer_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["year"] = pd.to_numeric(engineered["year"], errors="coerce")
    engineered["mileage_km"] = pd.to_numeric(engineered["mileage_km"], errors="coerce")
    engineered["seller_price"] = pd.to_numeric(engineered.get("seller_price", np.nan), errors="coerce")
    engineered["engine_size_l"] = pd.to_numeric(engineered.get("engine_size_l", np.nan), errors="coerce")
    engineered["car_age"] = (CURRENT_YEAR - engineered["year"]).clip(lower=0)
    engineered["mileage_per_year"] = engineered["mileage_km"] / np.maximum(engineered["car_age"], 1)
    engineered["log_mileage"] = np.log1p(engineered["mileage_km"].clip(lower=0))
    engineered["seller_price_log"] = np.log1p(engineered["seller_price"].clip(lower=0))

    for column in BASE_CATEGORICAL_FEATURES + NLP_CATEGORICAL_FEATURES:
        if column in engineered.columns:
            engineered[column] = engineered[column].map(safe_lower).replace("", "unknown")

    for column in NLP_NUMERIC_FEATURES:
        if column in engineered.columns:
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce").fillna(0)

    return engineered


def get_numeric_feature_columns(include_nlp_features: bool = False):
    columns = list(BASE_NUMERIC_FEATURES)
    if include_nlp_features:
        columns.extend(NLP_NUMERIC_FEATURES)
    return columns


def get_categorical_feature_columns(include_nlp_features: bool = False):
    columns = list(BASE_CATEGORICAL_FEATURES)
    if include_nlp_features:
        columns.extend(NLP_CATEGORICAL_FEATURES)
    return columns


def build_preprocessor(df: pd.DataFrame, include_nlp_features: bool = False) -> ColumnTransformer:
    numeric_features = [col for col in get_numeric_feature_columns(include_nlp_features) if col in df.columns]
    categorical_features = [
        col for col in get_categorical_feature_columns(include_nlp_features) if col in df.columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def prepare_features_and_target(df: pd.DataFrame, include_nlp_features: bool = False):
    cleaned = clean_numeric_data(df)
    engineered = engineer_numeric_features(cleaned)
    feature_columns = get_numeric_feature_columns(include_nlp_features) + get_categorical_feature_columns(
        include_nlp_features
    )
    feature_columns = [col for col in feature_columns if col in engineered.columns]
    X = engineered[feature_columns].copy()
    y = engineered["seller_price"].copy()
    preprocessor = build_preprocessor(engineered, include_nlp_features=include_nlp_features)
    return X, y, preprocessor, engineered
