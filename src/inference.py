import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CARSCOM_RAW_FILE,
    CRAIGSLIST_RAW_FILE,
    CURRENCY_DEFAULT,
    CURRENT_YEAR,
    INTEGRATED_MODEL_FILE,
    METADATA_FILE,
    PROCESSED_FILE,
    REPORT_DIR,
    STRUCTURED_MODEL_FILE,
)
from src.nlp_features import add_nlp_features, generate_buyer_explanation
from src.preprocessing_numeric import clean_numeric_data, engineer_numeric_features
from src.utils import load_json, safe_float


REAL_DATA_MODEL_DIR = Path("models") / "models"
REAL_DATA_INTEGRATED_MODEL_FILE = REAL_DATA_MODEL_DIR / "price_model_integrated.joblib"
REAL_DATA_STRUCTURED_MODEL_FILE = REAL_DATA_MODEL_DIR / "price_model_structured.joblib"
REAL_DATA_METADATA_FILE = REAL_DATA_MODEL_DIR / "metadata.json"
REAL_DATA_REPORT_DIR = Path("reports") / "reports"


def _metadata_uses_real_data(metadata_file: Path) -> bool:
    if not metadata_file.exists():
        return False
    metadata = load_json(metadata_file)
    return metadata.get("using_demo_data") is False


def _load_model_for_inference(model_file: Path):
    model = joblib.load(model_file)
    estimator = model.steps[-1][1] if hasattr(model, "steps") and model.steps else model
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1
    return model


def _load_price_model():
    model_sets = []
    if _metadata_uses_real_data(REAL_DATA_METADATA_FILE):
        model_sets.append(
            (
                REAL_DATA_INTEGRATED_MODEL_FILE,
                REAL_DATA_STRUCTURED_MODEL_FILE,
                REAL_DATA_METADATA_FILE,
                "real-data artifact set",
            )
        )
    model_sets.append((INTEGRATED_MODEL_FILE, STRUCTURED_MODEL_FILE, METADATA_FILE, "top-level artifact set"))

    for integrated_file, structured_file, metadata_file, source_label in model_sets:
        if integrated_file.exists():
            return (
                _load_model_for_inference(integrated_file),
                f"integrated structured + NLP model ({source_label}: {integrated_file.parent})",
                metadata_file,
                integrated_file,
                source_label,
            )
        if structured_file.exists():
            return (
                _load_model_for_inference(structured_file),
                f"structured-only model ({source_label}: {structured_file.parent})",
                metadata_file,
                structured_file,
                source_label,
            )
    return None, None, None, None, "none"


def _file_status(path: Path | None) -> str:
    if path is None:
        return "not selected"
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        return f"{path.resolve()} - present, {size_mb:.2f} MB"
    return f"{path.resolve()} - missing"


def _report_dir_for_metadata(metadata_file: Path | None, source_label: str) -> Path:
    if source_label == "real-data artifact set":
        return REAL_DATA_REPORT_DIR
    if metadata_file is not None and metadata_file.parent == REAL_DATA_MODEL_DIR:
        return REAL_DATA_REPORT_DIR
    if metadata_file is not None and metadata_file.exists():
        metadata = load_json(metadata_file)
        if metadata.get("using_demo_data") is False and (REAL_DATA_REPORT_DIR / "metrics_integrated.json").exists():
            return REAL_DATA_REPORT_DIR
    return REPORT_DIR


def _runtime_file_summary(
    model_file: Path | None,
    metadata_file: Path | None,
    model_used: str | None,
    source_label: str,
) -> str:
    report_dir = _report_dir_for_metadata(metadata_file, source_label)
    metadata = load_json(metadata_file) if metadata_file is not None and metadata_file.exists() else {}
    raw_files = metadata.get("raw_files", {})
    processed_rows = metadata.get("processed_rows", "unknown")
    using_demo_data = metadata.get("using_demo_data", "unknown")

    lines = [
        "Runtime files used for this website prediction:",
        f"- Price prediction model: {_file_status(model_file)}",
        f"- Model metadata: {_file_status(metadata_file)}",
        f"- Model selected: {model_used or 'none'}",
        f"- Metadata says using_demo_data: {using_demo_data}",
        f"- Metadata processed_rows: {processed_rows}",
        "",
        "Evaluation evidence files available on this deployment:",
        f"- Numeric metrics: {_file_status(report_dir / 'metrics_numeric.json')}",
        f"- NLP metrics: {_file_status(report_dir / 'metrics_nlp.json')}",
        f"- Integrated metrics: {_file_status(report_dir / 'metrics_integrated.json')}",
        f"- Sample predictions: {_file_status(report_dir / 'sample_predictions.csv')}",
        "",
        "Data files note:",
        "- The website prediction does not read the raw CSV datasets live. It loads the saved model above and uses the typed listing values.",
        f"- Processed CSV for training/evaluation reference: {_file_status(PROCESSED_FILE)}",
        f"- Craigslist raw CSV from metadata: {_file_status(Path(raw_files.get('craigslist', CRAIGSLIST_RAW_FILE)))}",
        f"- Cars.com raw CSV from metadata: {_file_status(Path(raw_files.get('carscom', CARSCOM_RAW_FILE)))}",
        "",
        "Code modules used during prediction:",
        f"- Inference and decision logic: {_file_status(Path('src/inference.py'))}",
        f"- Numeric preprocessing: {_file_status(Path('src/preprocessing_numeric.py'))}",
        f"- NLP feature extraction: {_file_status(Path('src/nlp_features.py'))}",
    ]
    return "\n".join(lines)


def _round_money(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    unit = 100 if abs(value) >= 20000 else 50
    return float(max(0, round(value / unit) * unit))


def _round_signed_money(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    unit = 100 if abs(value) >= 20000 else 50
    return float(round(value / unit) * unit)


def _risk_discount_range(label: str, score: float) -> tuple[float, float]:
    label = (label or "Low").lower()
    score = max(0.0, min(1.0, float(score)))
    if label == "high":
        low = 0.12 + 0.03 * max(score - 0.6, 0) / 0.4
        high = 0.16 + 0.04 * max(score - 0.6, 0) / 0.4
        return min(low, 0.20), min(high, 0.20)
    if label == "medium":
        low = 0.05 + 0.02 * max(score - 0.25, 0) / 0.35
        high = 0.08 + 0.02 * max(score - 0.25, 0) / 0.35
        return min(low, 0.10), min(high, 0.10)
    return 0.0, 0.03


def _price_status(difference_percent: float) -> str:
    if difference_percent > 15:
        return "Overpriced"
    if abs(difference_percent) <= 10:
        return "Fairly priced"
    if difference_percent < -10:
        return "Possibly good deal, but check risks"
    return "Slightly above predicted value"


def _confidence_notes(row: dict, model_used: str, metadata_file: Path | None) -> str:
    notes = []
    fields_to_check = [
        "brand",
        "model",
        "fuel_type",
        "transmission",
        "body_type",
        "condition",
        "location_region",
    ]
    missing = [
        field
        for field in fields_to_check
        if str(row.get(field, "")).strip().lower() in {"", "unknown", "nan", "none"}
    ]
    if len(missing) >= 3:
        notes.append(f"Many fields are missing or Unknown: {', '.join(missing)}.")
    year = safe_float(row.get("year"))
    mileage = safe_float(row.get("mileage_km"))
    if math.isfinite(year) and (year < 1980 or year > CURRENT_YEAR):
        notes.append(f"The year {year:.0f} is outside the expected training range.")
    if math.isfinite(mileage) and mileage > 350000:
        notes.append("Mileage is extremely high, so model uncertainty is higher.")
    if "integrated" not in model_used:
        notes.append("Integrated model is not available; using the structured-only price model.")
    if metadata_file is not None and metadata_file.exists():
        metadata = load_json(metadata_file)
        if metadata.get("using_demo_data"):
            notes.append("Model metadata says training used DEMO ONLY fallback data; this is not valid for final submission.")
    return " ".join(notes) if notes else "No major confidence warnings from the available fields."


def predict_listing(
    brand,
    model,
    year,
    mileage_km,
    fuel_type,
    transmission,
    body_type,
    engine_size_l,
    condition,
    title_status,
    drive,
    paint_color,
    location_country,
    location_region,
    currency,
    seller_price,
    seller_description,
):
    price_model, model_used, metadata_file, model_file, source_label = _load_price_model()
    if price_model is None:
        runtime_file_summary = _runtime_file_summary(model_file, metadata_file, model_used, source_label)
        return {
            "error": "Models not found. Please run the training scripts locally first.",
            "predicted_price": None,
            "seller_price": safe_float(seller_price),
            "price_difference": None,
            "price_difference_percent": None,
            "price_status": "Model unavailable",
            "nlp_risk_label": "Unknown",
            "nlp_risk_score": None,
            "risk_terms_found": "",
            "positive_terms_found": "",
            "recommended_offer_low": None,
            "recommended_offer_high": None,
            "final_recommendation": "Models not found. Please run python -m src.train_numeric and python -m src.train_integrated first.",
            "explanation": "Models not found. Please run the training scripts locally first.",
            "model_used": "none",
            "model_confidence_note": "Models not found. Please run the training scripts locally first.",
            "runtime_file_summary": runtime_file_summary,
        }

    seller_price_value = safe_float(seller_price)
    row = {
        "listing_id": "manual_input",
        "source_name": "manual_app_input",
        "brand": brand,
        "model": model,
        "year": safe_float(year),
        "mileage_km": safe_float(mileage_km),
        "fuel_type": fuel_type,
        "transmission": transmission,
        "body_type": body_type,
        "engine_size_l": safe_float(engine_size_l),
        "condition": condition,
        "title_status": title_status,
        "drive": drive,
        "paint_color": paint_color,
        "location_country": location_country,
        "location_region": location_region,
        "currency": currency or CURRENCY_DEFAULT,
        "seller_price": seller_price_value,
        "seller_description": seller_description or "",
    }

    input_df = pd.DataFrame([row])
    input_df = add_nlp_features(input_df)
    model_input = engineer_numeric_features(clean_numeric_data(input_df))
    if model_input.empty:
        runtime_file_summary = _runtime_file_summary(model_file, metadata_file, model_used, source_label)
        return {
            "error": "Input values are invalid. Check price, year, mileage, brand, and model.",
            "predicted_price": None,
            "seller_price": seller_price_value,
            "price_difference": None,
            "price_difference_percent": None,
            "price_status": "Invalid input",
            "nlp_risk_label": "Unknown",
            "nlp_risk_score": None,
            "risk_terms_found": "",
            "positive_terms_found": "",
            "recommended_offer_low": None,
            "recommended_offer_high": None,
            "final_recommendation": "Please enter valid positive price and mileage values and a year between 1980 and 2026.",
            "explanation": "The input row did not pass the same basic validation used in training.",
            "model_used": model_used,
            "model_confidence_note": "Invalid input.",
            "runtime_file_summary": runtime_file_summary,
        }

    predicted_price = float(np.clip(price_model.predict(model_input)[0], 0, None))
    price_difference = seller_price_value - predicted_price
    price_difference_percent = (price_difference / predicted_price * 100) if predicted_price > 0 else 0.0
    status = _price_status(price_difference_percent)

    nlp_row = input_df.iloc[0]
    risk_label = nlp_row["nlp_risk_label"]
    risk_score = float(nlp_row["text_risk_score"])
    discount_low, discount_high = _risk_discount_range(risk_label, risk_score)
    anchor_price = min(predicted_price, seller_price_value) if seller_price_value > 0 else predicted_price
    offer_high = _round_money(anchor_price * (1 - discount_low))
    offer_low = _round_money(anchor_price * (1 - discount_high))
    if offer_low > offer_high:
        offer_low, offer_high = offer_high, offer_low

    if risk_label == "High":
        recommendation = "High risk: negotiate strongly and arrange an independent inspection before buying."
    elif status == "Overpriced":
        recommendation = "Negotiate down toward the model estimate and use the text risk signals as leverage."
    elif status.startswith("Possibly good deal"):
        recommendation = "Potentially attractive price, but verify the risk signals before treating it as a bargain."
    else:
        recommendation = "Reasonable candidate if inspection and documents confirm the listing details."

    result = {
        "predicted_price": _round_money(predicted_price),
        "seller_price": float(seller_price_value),
        "price_difference": _round_signed_money(price_difference),
        "price_difference_percent": float(price_difference_percent),
        "price_status": status,
        "nlp_risk_label": risk_label,
        "nlp_risk_score": risk_score,
        "risk_terms_found": nlp_row["risk_terms_found"],
        "positive_terms_found": nlp_row["positive_terms_found"],
        "recommended_offer_low": offer_low,
        "recommended_offer_high": offer_high,
        "final_recommendation": recommendation,
        "model_used": model_used,
        "model_confidence_note": _confidence_notes(row, model_used, metadata_file),
        "runtime_file_summary": _runtime_file_summary(model_file, metadata_file, model_used, source_label),
    }
    result["explanation"] = generate_buyer_explanation(result)
    return result
