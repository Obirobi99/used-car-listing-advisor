import re

import pandas as pd

from src.utils import safe_lower


RISK_KEYWORDS = [
    "accident",
    "crash",
    "damaged",
    "damage",
    "rust",
    "engine problem",
    "engine issue",
    "engine failure",
    "gearbox problem",
    "transmission issue",
    "oil leak",
    "warning light",
    "check engine",
    "check engine light",
    "needs repair",
    "repair needed",
    "not running",
    "export only",
    "sold as is",
    "urgent sale",
    "no inspection",
    "failed inspection",
    "high mileage",
    "minor scratch",
    "minor scratches",
    "salvage",
    "rebuilt",
    "rebuild",
    "mechanic special",
    "overheating",
    "smoke",
    "bad clutch",
    "electrical issue",
]

POSITIVE_KEYWORDS = [
    "full service history",
    "service history",
    "recently serviced",
    "new tires",
    "new tyres",
    "new brakes",
    "warranty",
    "inspection passed",
    "accident free",
    "well maintained",
    "one owner",
    "garage kept",
    "low mileage",
    "clean title",
    "no accident",
    "no damage",
    "no rust",
    "no issues",
    "no repair needed",
    "excellent condition",
    "never crashed",
]

NEGATED_RISK_PHRASES = {
    "accident": ["no accident", "accident free", "never crashed"],
    "crash": ["never crashed", "no crash"],
    "damage": ["no damage", "undamaged"],
    "damaged": ["no damage", "undamaged"],
    "rust": ["no rust", "rust free"],
    "engine problem": ["no engine problem", "no engine problems"],
    "engine issue": ["no engine issue", "no engine issues"],
    "transmission issue": ["no transmission issue", "no transmission issues"],
    "repair needed": ["no repair needed"],
    "needs repair": ["no repair needed"],
    "salvage": ["no salvage"],
}

SEVERE_TERMS = {
    "engine problem",
    "engine issue",
    "engine failure",
    "gearbox problem",
    "transmission issue",
    "oil leak",
    "warning light",
    "check engine",
    "check engine light",
    "not running",
    "failed inspection",
    "salvage",
    "rebuilt",
    "mechanic special",
    "overheating",
    "smoke",
    "bad clutch",
    "electrical issue",
}


def clean_text(text) -> str:
    text = safe_lower(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase)
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def contains_negated_phrase(text, term) -> bool:
    cleaned = clean_text(text)
    for phrase in NEGATED_RISK_PHRASES.get(term, []):
        if _contains_phrase(cleaned, phrase):
            return True

    # Generic short-window negation for phrases such as "no serious accident".
    words = cleaned.split()
    term_words = term.split()
    if not term_words:
        return False
    for index in range(len(words) - len(term_words) + 1):
        if words[index : index + len(term_words)] == term_words:
            window = words[max(0, index - 3) : index]
            if any(word in {"no", "not", "never", "without"} for word in window):
                return True
    return False


def extract_keyword_features(text):
    cleaned = clean_text(text)
    risk_terms = []
    positive_terms = []

    for term in RISK_KEYWORDS:
        if _contains_phrase(cleaned, term) and not contains_negated_phrase(cleaned, term):
            risk_terms.append(term)

    for term in POSITIVE_KEYWORDS:
        if _contains_phrase(cleaned, term):
            positive_terms.append(term)

    return {
        "cleaned_description": cleaned,
        "description_length_words": len(cleaned.split()) if cleaned else 0,
        "risk_keyword_count": len(risk_terms),
        "positive_keyword_count": len(positive_terms),
        "risk_terms_found": risk_terms,
        "positive_terms_found": positive_terms,
    }


def score_keyword_features(features: dict) -> float:
    risk_terms = features["risk_terms_found"]
    positive_terms = features["positive_terms_found"]

    score = 0.0
    for term in risk_terms:
        score += 0.22 if term in SEVERE_TERMS else 0.15

    # Very short descriptions can be risky because they reveal little.
    if 0 < features["description_length_words"] < 12:
        score += 0.08

    score -= min(0.25, 0.05 * len(positive_terms))
    return float(max(0.0, min(1.0, score)))


def compute_positive_signal_score(text) -> float:
    features = extract_keyword_features(text)
    return min(1.0, features["positive_keyword_count"] / 5.0)


def compute_text_risk_score(text) -> float:
    features = extract_keyword_features(text)
    return score_keyword_features(features)


def _risk_label_from_score(score: float) -> str:
    if score < 0.25:
        return "Low"
    if score < 0.6:
        return "Medium"
    return "High"


def generate_weak_risk_label(text) -> str:
    return _risk_label_from_score(compute_text_risk_score(text))


def add_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    descriptions = result.get(
        "seller_description",
        pd.Series("", index=result.index, dtype="object"),
    ).fillna("").astype(str)

    result["cleaned_description"] = ""
    result["description_length_words"] = 0
    result["risk_keyword_count"] = 0
    result["positive_keyword_count"] = 0
    result["risk_terms_found"] = ""
    result["positive_terms_found"] = ""
    result["text_risk_score"] = 0.0
    result["nlp_risk_label"] = "Low"

    non_empty_descriptions = descriptions[descriptions.str.strip().ne("")]
    if non_empty_descriptions.empty:
        return result

    feature_rows = []
    for text in non_empty_descriptions:
        features = extract_keyword_features(text)
        score = score_keyword_features(features)
        features["text_risk_score"] = score
        features["nlp_risk_label"] = _risk_label_from_score(score)
        features["risk_terms_found"] = "; ".join(features["risk_terms_found"])
        features["positive_terms_found"] = "; ".join(features["positive_terms_found"])
        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows, index=non_empty_descriptions.index)
    for column in feature_df.columns:
        result.loc[feature_df.index, column] = feature_df[column]
    return result


def generate_buyer_explanation(result_dict) -> str:
    predicted = result_dict.get("predicted_price")
    seller = result_dict.get("seller_price")
    price_status = result_dict.get("price_status", "Unknown")
    risk_label = result_dict.get("nlp_risk_label", "Unknown")
    risk_terms = result_dict.get("risk_terms_found") or "none"
    positive_terms = result_dict.get("positive_terms_found") or "none"
    difference_percent = result_dict.get("price_difference_percent", 0.0)

    return (
        f"The listing is {price_status.lower()} based on the model estimate. "
        f"The seller price is {difference_percent:.1f}% away from the predicted fair value "
        f"({seller:,.0f} vs {predicted:,.0f}). "
        f"The description risk level is {risk_label}. Risk signals found: {risk_terms}. "
        f"Positive signals found: {positive_terms}. Use this as a screening tool and still inspect the car."
    )
