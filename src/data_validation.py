import pandas as pd

from src.config import CURRENT_YEAR


REQUIRED_PROCESSED_COLUMNS = [
    "listing_id",
    "source_name",
    "brand",
    "model",
    "year",
    "mileage_km",
    "fuel_type",
    "transmission",
    "body_type",
    "condition",
    "location_country",
    "location_region",
    "currency",
    "seller_price",
    "seller_description",
]


def _raise_with_examples(message: str, series: pd.Series, limit: int = 10) -> None:
    examples = series.head(limit).tolist()
    raise ValueError(f"{message}. Example values: {examples}")


def validate_processed_schema(df: pd.DataFrame) -> None:
    missing_columns = [col for col in REQUIRED_PROCESSED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Processed data is missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Processed data has no rows after mapping and cleaning.")

    if df["source_name"].isna().all():
        raise ValueError("source_name is missing for all rows.")
    if df["listing_id"].isna().all():
        raise ValueError("listing_id is missing for all rows.")

    duplicated = df["listing_id"].astype(str).duplicated().sum()
    if duplicated:
        print(f"WARNING: Found {duplicated} duplicated listing_id values.")


def validate_price_values(df: pd.DataFrame) -> None:
    prices = pd.to_numeric(df["seller_price"], errors="coerce")
    bad_numeric = df.loc[prices.isna(), "seller_price"]
    if len(bad_numeric) > 0:
        _raise_with_examples("seller_price contains non-numeric values", bad_numeric)

    invalid = df.loc[prices <= 0, "seller_price"]
    if len(invalid) > 0:
        _raise_with_examples("seller_price must be greater than 0", invalid)


def validate_year_values(df: pd.DataFrame) -> None:
    years = pd.to_numeric(df["year"], errors="coerce")
    bad_numeric = df.loc[years.isna(), "year"]
    if len(bad_numeric) > 0:
        _raise_with_examples("year contains non-numeric values", bad_numeric)

    invalid = df.loc[(years < 1980) | (years > CURRENT_YEAR), "year"]
    if len(invalid) > 0:
        _raise_with_examples(f"year must be between 1980 and {CURRENT_YEAR}", invalid)


def validate_mileage_values(df: pd.DataFrame) -> None:
    mileage = pd.to_numeric(df["mileage_km"], errors="coerce")
    bad_numeric = df.loc[mileage.isna(), "mileage_km"]
    if len(bad_numeric) > 0:
        _raise_with_examples("mileage_km contains non-numeric values", bad_numeric)

    invalid = df.loc[mileage < 0, "mileage_km"]
    if len(invalid) > 0:
        _raise_with_examples("mileage_km must be greater than or equal to 0", invalid)


def validate_text_values(df: pd.DataFrame, require_text: bool = True) -> None:
    descriptions = df["seller_description"].fillna("").astype(str).str.strip()
    craigslist_mask = df["source_name"].astype(str).str.lower().eq("kaggle_craigslist")

    if require_text and craigslist_mask.any():
        craigslist_non_empty = descriptions[craigslist_mask].ne("").sum()
        if craigslist_non_empty == 0:
            raise ValueError(
                "seller_description is empty for all Craigslist rows, so NLP training cannot run."
            )

    nlp_rows = descriptions.ne("").sum()
    if require_text and nlp_rows == 0:
        raise ValueError("No non-empty seller_description values are available for NLP.")

    if nlp_rows < len(df):
        print(f"WARNING: {len(df) - nlp_rows} rows have empty seller_description values.")


def _warn_missingness(df: pd.DataFrame) -> None:
    missing_share = df.isna().mean().sort_values(ascending=False)
    high_missing = missing_share[missing_share > 0.4]
    for column, share in high_missing.items():
        print(f"WARNING: Column '{column}' has {share:.1%} missing values.")


def _validate_brand_model_coverage(df: pd.DataFrame) -> None:
    for column in ["brand", "model"]:
        values = df[column].fillna("").astype(str).str.strip()
        empty_share = values.eq("").mean()
        unknown_share = values.str.lower().eq("unknown").mean()
        weak_share = empty_share + unknown_share
        if weak_share > 0.8:
            raise ValueError(f"{column} is empty or Unknown for more than 80% of rows.")
        if weak_share > 0.2:
            print(f"WARNING: {column} is empty or Unknown for {weak_share:.1%} of rows.")


def validate_all_data(df: pd.DataFrame, require_text: bool = True) -> None:
    validate_processed_schema(df)
    validate_price_values(df)
    validate_year_values(df)
    validate_mileage_values(df)
    validate_text_values(df, require_text=require_text)
    _validate_brand_model_coverage(df)
    _warn_missingness(df)

    print("\nMissing values summary:")
    print(df.isna().sum().sort_values(ascending=False))
    print("\nSource distribution:")
    print(df["source_name"].value_counts(dropna=False))

