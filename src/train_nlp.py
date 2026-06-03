from collections import Counter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.config import (
    LOGISTIC_REGRESSION_MAX_ITER,
    METADATA_FILE,
    PROCESSED_FILE,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TEXT_RISK_MODEL_FILE,
    TEXT_VECTORIZER_FILE,
    TRAIN_N_JOBS,
)
from src.data_loading import load_and_prepare_data
from src.nlp_features import add_nlp_features, clean_text
from src.utils import ensure_directories, load_json, print_section, save_json


def _load_processed_data() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        print("Processed file not found. Running data preparation first.")
        return load_and_prepare_data()
    return pd.read_csv(PROCESSED_FILE)


def _choose_example(df: pd.DataFrame, label: str) -> dict:
    rows = df[df["nlp_risk_label"].eq(label)]
    if rows.empty:
        return {"label": label, "text": "", "note": "No example found in current data."}
    row = rows.iloc[0]
    return {
        "label": label,
        "text": row["seller_description"],
        "risk_terms_found": row["risk_terms_found"],
        "positive_terms_found": row["positive_terms_found"],
        "text_risk_score": row["text_risk_score"],
    }


def _known_edge_cases() -> list[dict]:
    examples = [
        "no accident, no damage, no repair needed",
        "accident free and recently serviced",
        "minor scratches on bumper but inspection passed",
        "engine problem with check engine light and oil leak",
    ]
    rows = []
    for text in examples:
        features = add_nlp_features(pd.DataFrame({"seller_description": [text]})).iloc[0]
        rows.append(
            {
                "text": text,
                "generated_label": features["nlp_risk_label"],
                "risk_terms_found": features["risk_terms_found"],
                "positive_terms_found": features["positive_terms_found"],
                "text_risk_score": features["text_risk_score"],
            }
        )
    return rows


def train_nlp_models() -> dict:
    ensure_directories()
    print_section("NLP Risk Model Training")
    df = _load_processed_data()
    text_mask = df["seller_description"].fillna("").astype(str).str.strip().ne("")
    df_text = add_nlp_features(df.loc[text_mask].copy())

    if len(df_text) < 6:
        raise ValueError("At least 6 non-empty seller descriptions are required for NLP training.")

    df_text["weak_label"] = df_text["nlp_risk_label"]
    class_distribution = dict(Counter(df_text["weak_label"]))
    print(f"Weak-label class distribution: {class_distribution}")

    stratify = df_text["weak_label"] if min(class_distribution.values()) >= 2 else None
    train_df, test_df = train_test_split(
        df_text, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=1,
        max_features=TFIDF_MAX_FEATURES,
    )
    X_train = vectorizer.fit_transform(train_df["seller_description"])
    X_test = vectorizer.transform(test_df["seller_description"])
    y_train = train_df["weak_label"]
    y_test = test_df["weak_label"]

    clf = LogisticRegression(
        max_iter=LOGISTIC_REGRESSION_MAX_ITER,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=TRAIN_N_JOBS,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = sorted(df_text["weak_label"].unique().tolist())
    tfidf_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }

    joblib.dump(vectorizer, TEXT_VECTORIZER_FILE)
    joblib.dump(clf, TEXT_RISK_MODEL_FILE)
    print(f"Saved NLP vectorizer to {TEXT_VECTORIZER_FILE}")
    print(f"Saved NLP risk model to {TEXT_RISK_MODEL_FILE}")

    metrics = {
        "label_type": "weak labels generated from rule-based NLP",
        "number_of_text_rows": int(len(df_text)),
        "class_distribution": class_distribution,
        "rule_based_summary": {
            "average_text_risk_score": float(df_text["text_risk_score"].mean()),
            "average_risk_keyword_count": float(df_text["risk_keyword_count"].mean()),
            "average_positive_keyword_count": float(df_text["positive_keyword_count"].mean()),
            "note": "Rule-based labels are transparent but not human-verified ground truth.",
        },
        "tfidf_logistic_regression_metrics": tfidf_metrics,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        },
        "qualitative_examples": {
            "low_risk_example": _choose_example(df_text, "Low"),
            "medium_risk_example": _choose_example(df_text, "Medium"),
            "high_risk_example": _choose_example(df_text, "High"),
            "failure_cases_and_negation_checks": _known_edge_cases(),
        },
        "limitations": [
            "Risk labels are weak labels generated from keyword and negation rules, not human annotations.",
            "Seller descriptions can omit important defects or exaggerate positives.",
            "Keyword logic can still be misled by sarcasm, spelling errors, and complex phrasing.",
            "The TF-IDF classifier learns from weak labels, so it inherits some rule-based bias.",
        ],
    }
    save_json(metrics, REPORT_DIR / "metrics_nlp.json")

    metadata = load_json(METADATA_FILE) if METADATA_FILE.exists() else {}
    metadata.update(
        {
            "text_vectorizer_file": str(TEXT_VECTORIZER_FILE),
            "text_risk_model_file": str(TEXT_RISK_MODEL_FILE),
            "nlp_label_type": "weak labels generated from rule-based NLP",
        }
    )
    save_json(metadata, METADATA_FILE)
    return metrics


if __name__ == "__main__":
    train_nlp_models()
