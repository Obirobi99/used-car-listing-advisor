import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CARSCOM_RAW_FILE,
    CRAIGSLIST_RAW_FILE,
    CURRENT_YEAR,
    METADATA_FILE,
    PROCESSED_FILE,
)
from src.data_validation import REQUIRED_PROCESSED_COLUMNS, validate_all_data
from src.utils import ensure_directories, print_section, safe_float, save_json


INTERNAL_COLUMNS = [
    "listing_id",
    "source_name",
    "brand",
    "model",
    "year",
    "mileage_km",
    "fuel_type",
    "transmission",
    "body_type",
    "engine_size_l",
    "condition",
    "title_status",
    "drive",
    "paint_color",
    "location_country",
    "location_region",
    "currency",
    "seller_price",
    "seller_description",
]


def load_raw_craigslist():
    if not CRAIGSLIST_RAW_FILE.exists():
        print(f"Craigslist file not found: {CRAIGSLIST_RAW_FILE}")
        return None
    df = pd.read_csv(CRAIGSLIST_RAW_FILE, low_memory=False)
    print(f"Loaded Craigslist data: {df.shape}")
    print(f"Detected Craigslist columns: {list(df.columns)}")
    return df


def load_raw_carscom():
    if not CARSCOM_RAW_FILE.exists():
        print(f"Cars.com file not found: {CARSCOM_RAW_FILE}")
        return None
    df = pd.read_csv(CARSCOM_RAW_FILE, low_memory=False)
    print(f"Loaded Cars.com data: {df.shape}")
    print(f"Detected Cars.com columns: {list(df.columns)}")
    return df


def _empty_internal_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=INTERNAL_COLUMNS)


def _extract_engine_size_l(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*l", text)
    if match:
        return safe_float(match.group(1))
    return safe_float(value)


def _first_existing_column(df: pd.DataFrame, candidates) -> str | None:
    normalized = {col.lower().strip().replace(" ", "_"): col for col in df.columns}
    for candidate in candidates:
        key = candidate.lower().strip().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def _series_or_default(df: pd.DataFrame, column: str | None, default):
    if column and column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def create_demo_data() -> pd.DataFrame:
    print(
        "WARNING: Real Kaggle CSV files not found. Using DEMO ONLY fallback data. "
        "This is not valid for final submission."
    )
    rows = [
        ("demo_cl_001", "kaggle_craigslist", "toyota", "corolla", 2018, 89000, "gas", "automatic", "sedan", np.nan, "excellent", "clean", "fwd", "silver", "USA", "ca", "USD", 14200, "One owner Toyota Corolla, full service history, recently serviced, new tires, clean title, no accident."),
        ("demo_cl_002", "kaggle_craigslist", "honda", "civic", 2017, 104000, "gas", "manual", "sedan", np.nan, "good", "clean", "fwd", "blue", "USA", "ny", "USD", 12800, "Well maintained Civic with service history, new brakes, inspection passed, no issues."),
        ("demo_cl_003", "kaggle_craigslist", "ford", "focus", 2015, 142000, "gas", "automatic", "hatchback", np.nan, "fair", "clean", "fwd", "white", "USA", "tx", "USD", 7200, "Runs and drives, minor scratches, some cosmetic damage, high mileage, recently serviced."),
        ("demo_cl_004", "kaggle_craigslist", "volkswagen", "golf", 2016, 126000, "diesel", "manual", "hatchback", np.nan, "good", "clean", "fwd", "black", "USA", "wa", "USD", 9800, "VW Golf diesel, minor scratches on bumper, small rust spot, no inspection available, service history, no accident."),
        ("demo_cl_005", "kaggle_craigslist", "bmw", "328i", 2013, 171000, "gas", "automatic", "sedan", np.nan, "fair", "rebuilt", "rwd", "gray", "USA", "fl", "USD", 9600, "BMW 328i sold as is, oil leak, check engine light, needs repair, mechanic special."),
        ("demo_cl_006", "kaggle_craigslist", "nissan", "altima", 2014, 160000, "gas", "automatic", "sedan", np.nan, "fair", "salvage", "fwd", "red", "USA", "az", "USD", 5900, "Salvage title after accident, warning light on, not running every time."),
        ("demo_cl_007", "kaggle_craigslist", "subaru", "outback", 2019, 93000, "gas", "automatic", "wagon", np.nan, "excellent", "clean", "awd", "green", "USA", "co", "USD", 21300, "Garage kept Outback, one owner, warranty remaining, accident free, low mileage."),
        ("demo_cl_008", "kaggle_craigslist", "chevrolet", "silverado", 2016, 188000, "gas", "automatic", "pickup", np.nan, "good", "clean", "4wd", "black", "USA", "oh", "USD", 22400, "Work truck with high mileage, rust on bed, runs strong, new tires."),
        ("demo_cl_009", "kaggle_craigslist", "hyundai", "elantra", 2020, 61000, "gas", "automatic", "sedan", np.nan, "excellent", "clean", "fwd", "white", "USA", "il", "USD", 16600, "Excellent condition, clean title, no damage, no rust, recently serviced."),
        ("demo_cl_010", "kaggle_craigslist", "audi", "a4", 2015, 151000, "gas", "automatic", "sedan", np.nan, "fair", "clean", "awd", "silver", "USA", "ma", "USD", 11000, "Audi A4 has transmission issue and electrical issue, sold as is, urgent sale."),
        ("demo_cl_011", "kaggle_craigslist", "mazda", "cx-5", 2021, 52000, "gas", "automatic", "suv", np.nan, "excellent", "clean", "awd", "blue", "USA", "or", "USD", 24800, "CX-5 one owner, full service history, excellent condition, no accident, new brakes."),
        ("demo_cl_012", "kaggle_craigslist", "jeep", "wrangler", 2012, 179000, "gas", "manual", "suv", np.nan, "fair", "clean", "4wd", "yellow", "USA", "ut", "USD", 15700, "Jeep with overheating issue, rust underneath, smoke at startup, needs repair."),
        ("demo_cc_001", "kaggle_carscom", "toyota", "camry", 2019, 76000, "gas", "automatic", "sedan", 2.5, "good", "clean", "unknown", "white", "USA", "carscom", "USD", 18900, ""),
        ("demo_cc_002", "kaggle_carscom", "ford", "escape", 2018, 99000, "gas", "automatic", "suv", 1.5, "good", "clean", "unknown", "gray", "USA", "carscom", "USD", 15100, ""),
        ("demo_cc_003", "kaggle_carscom", "tesla", "model 3", 2020, 68000, "electric", "automatic", "sedan", np.nan, "excellent", "clean", "rwd", "black", "USA", "carscom", "USD", 28600, ""),
        ("demo_cc_004", "kaggle_carscom", "kia", "sorento", 2017, 132000, "gas", "automatic", "suv", 2.4, "good", "clean", "unknown", "silver", "USA", "carscom", "USD", 13900, ""),
        ("demo_cc_005", "kaggle_carscom", "mercedes-benz", "c-class", 2016, 118000, "gas", "automatic", "sedan", 2.0, "good", "clean", "rwd", "black", "USA", "carscom", "USD", 17400, ""),
        ("demo_cc_006", "kaggle_carscom", "ram", "1500", 2019, 97000, "gas", "automatic", "pickup", 5.7, "good", "clean", "4wd", "red", "USA", "carscom", "USD", 30400, ""),
    ]
    return pd.DataFrame(rows, columns=INTERNAL_COLUMNS)


def map_craigslist_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    mapped = pd.DataFrame(index=df.index)
    mapped["listing_id"] = _series_or_default(df, "id", "").astype(str)
    mapped["source_name"] = "kaggle_craigslist"
    mapped["brand"] = _series_or_default(df, "manufacturer", "Unknown")
    mapped["model"] = _series_or_default(df, "model", "Unknown")
    mapped["year"] = pd.to_numeric(_series_or_default(df, "year", np.nan), errors="coerce")
    odometer_miles = pd.to_numeric(_series_or_default(df, "odometer", np.nan), errors="coerce")
    mapped["mileage_km"] = odometer_miles * 1.60934
    mapped["fuel_type"] = _series_or_default(df, "fuel", "Unknown")
    mapped["transmission"] = _series_or_default(df, "transmission", "Unknown")
    mapped["body_type"] = _series_or_default(df, "type", "Unknown")
    mapped["engine_size_l"] = np.nan
    mapped["condition"] = _series_or_default(df, "condition", "Unknown")
    mapped["title_status"] = _series_or_default(df, "title_status", "Unknown")
    mapped["drive"] = _series_or_default(df, "drive", "Unknown")
    mapped["paint_color"] = _series_or_default(df, "paint_color", "Unknown")
    mapped["location_country"] = "USA"
    state = _series_or_default(df, "state", "")
    region = _series_or_default(df, "region", "Unknown")
    mapped["location_region"] = state.fillna("").astype(str).str.strip()
    mapped.loc[mapped["location_region"].eq(""), "location_region"] = region
    mapped["currency"] = "USD"
    mapped["seller_price"] = pd.to_numeric(_series_or_default(df, "price", np.nan), errors="coerce")
    mapped["seller_description"] = _series_or_default(df, "description", "").fillna("")
    return mapped[INTERNAL_COLUMNS]


def map_carscom_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "listing_id": ["id", "listing_id", "vin", "stock_number"],
        "brand": ["brand", "make", "manufacturer"],
        "model": ["model", "vehicle_model"],
        "year": ["model_year", "year"],
        "mileage": ["milage", "mileage", "odometer", "miles", "mileage_mi", "mileage_km"],
        "seller_price": ["price", "list_price", "seller_price"],
        "fuel_type": ["fuel_type", "fuel"],
        "transmission": ["transmission"],
        "body_type": ["body_type", "body", "vehicle_type", "type"],
        "title_status": ["title_status", "clean_title"],
        "condition": ["condition"],
        "drive": ["drive", "drivetrain"],
        "paint_color": ["exterior_color", "paint_color", "color"],
        "engine_size_l": ["engine_size", "engine"],
        "seller_description": ["description", "seller_description"],
        "location_region": ["state", "region", "dealer_state", "location"],
    }

    detected = {key: _first_existing_column(df, candidates) for key, candidates in mapping.items()}
    print("Cars.com safe column mapping:")
    for project_column, raw_column in detected.items():
        if raw_column:
            print(f"  {project_column} <- {raw_column}")
        else:
            print(f"  WARNING: {project_column} could not be mapped and will use a safe default.")

    critical = ["brand", "model", "year", "mileage", "seller_price"]
    unusable = [
        name
        for name in critical
        if detected[name] is None or _series_or_default(df, detected[name], np.nan).notna().sum() == 0
    ]
    if len(unusable) == len(critical):
        print("WARNING: Cars.com source is unusable because critical columns are unavailable. Skipping it.")
        return _empty_internal_frame()

    mapped = pd.DataFrame(index=df.index)
    if detected["listing_id"]:
        mapped["listing_id"] = df[detected["listing_id"]].astype(str)
    else:
        mapped["listing_id"] = [f"carscom_{i}" for i in range(len(df))]
        print("WARNING: Cars.com listing_id not found. Generated deterministic row ids.")

    mapped["source_name"] = "kaggle_carscom"
    mapped["brand"] = _series_or_default(df, detected["brand"], "Unknown")
    mapped["model"] = _series_or_default(df, detected["model"], "Unknown")
    mapped["year"] = pd.to_numeric(_series_or_default(df, detected["year"], np.nan), errors="coerce")

    raw_mileage = pd.to_numeric(_series_or_default(df, detected["mileage"], np.nan).map(safe_float), errors="coerce")
    mileage_column_name = (detected["mileage"] or "").lower()
    if "km" in mileage_column_name:
        mapped["mileage_km"] = raw_mileage
    else:
        mapped["mileage_km"] = raw_mileage * 1.60934

    mapped["fuel_type"] = _series_or_default(df, detected["fuel_type"], "Unknown")
    mapped["transmission"] = _series_or_default(df, detected["transmission"], "Unknown")
    mapped["body_type"] = _series_or_default(df, detected["body_type"], "Unknown")
    mapped["engine_size_l"] = _series_or_default(df, detected["engine_size_l"], np.nan).map(_extract_engine_size_l)
    mapped["condition"] = _series_or_default(df, detected["condition"], "Unknown")

    title_status = _series_or_default(df, detected["title_status"], "Unknown").astype(str)
    accident_column = _first_existing_column(df, ["accident", "accidents", "accident_history"])
    if accident_column and detected["title_status"] is None:
        accident_text = df[accident_column].fillna("").astype(str).str.lower()
        title_status = np.where(
            accident_text.str.contains("accident|damage", regex=True),
            "accident reported",
            "Unknown",
        )
    mapped["title_status"] = title_status
    mapped["drive"] = _series_or_default(df, detected["drive"], "Unknown")
    mapped["paint_color"] = _series_or_default(df, detected["paint_color"], "Unknown")
    mapped["location_country"] = "USA"
    mapped["location_region"] = _series_or_default(df, detected["location_region"], "carscom")
    mapped["currency"] = "USD"
    mapped["seller_price"] = pd.to_numeric(
        _series_or_default(df, detected["seller_price"], np.nan).map(safe_float), errors="coerce"
    )
    mapped["seller_description"] = _series_or_default(df, detected["seller_description"], "").fillna("")
    return mapped[INTERNAL_COLUMNS]


def _clean_mapped_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    cleaned = df.copy()
    for column in ["seller_price", "year", "mileage_km", "engine_size_l"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    critical_mask = (
        cleaned["seller_price"].notna()
        & (cleaned["seller_price"] > 0)
        & cleaned["year"].notna()
        & cleaned["year"].between(1980, CURRENT_YEAR)
        & cleaned["mileage_km"].notna()
        & (cleaned["mileage_km"] >= 0)
        & cleaned["brand"].fillna("").astype(str).str.strip().ne("")
        & cleaned["model"].fillna("").astype(str).str.strip().ne("")
    )
    cleaned = cleaned.loc[critical_mask].copy()
    removed = before - len(cleaned)
    if removed:
        print(f"Removed {removed} rows with invalid critical price/year/mileage/brand/model fields.")

    text_columns = [
        "brand",
        "model",
        "fuel_type",
        "transmission",
        "body_type",
        "condition",
        "title_status",
        "drive",
        "paint_color",
        "location_country",
        "location_region",
        "currency",
        "seller_description",
    ]
    for column in text_columns:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str)
    cleaned["seller_description"] = cleaned["seller_description"].replace("Unknown", "")
    cleaned["listing_id"] = cleaned["listing_id"].fillna("").astype(str)
    return cleaned[INTERNAL_COLUMNS]


def save_processed_data(df: pd.DataFrame) -> None:
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"Saved processed data to {PROCESSED_FILE}")


def load_and_prepare_data() -> pd.DataFrame:
    ensure_directories()
    print_section("Loading Raw Data")
    craigslist_raw = load_raw_craigslist()
    carscom_raw = load_raw_carscom()

    frames = []
    using_demo = craigslist_raw is None and carscom_raw is None

    if using_demo:
        frames.append(create_demo_data())
    else:
        if craigslist_raw is not None:
            frames.append(map_craigslist_to_schema(craigslist_raw))
        if carscom_raw is not None:
            carscom_mapped = map_carscom_to_schema(carscom_raw)
            if not carscom_mapped.empty:
                frames.append(carscom_mapped)

    if not frames:
        raise ValueError("No usable data sources were available after mapping.")

    processed = pd.concat(frames, ignore_index=True)
    processed = _clean_mapped_rows(processed)

    print_section("Processed Data Validation")
    print(f"Processed shape: {processed.shape}")
    print(f"Processed columns: {list(processed.columns)}")
    validate_all_data(processed, require_text=True)
    save_processed_data(processed)

    metadata = {
        "using_demo_data": using_demo,
        "processed_rows": int(len(processed)),
        "source_counts": processed["source_name"].value_counts().to_dict(),
        "raw_files": {
            "craigslist": str(CRAIGSLIST_RAW_FILE),
            "carscom": str(CARSCOM_RAW_FILE),
        },
    }
    save_json(metadata, METADATA_FILE)
    return processed


if __name__ == "__main__":
    load_and_prepare_data()
