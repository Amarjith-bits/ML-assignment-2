"""
Shared helpers used by every training script (one script per model).
Keeping the split and preprocessing here guarantees all five models are
trained and evaluated on exactly the same data.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths are built from this file's location, so the scripts work no matter
# which folder you run them from (laptop, BITS Virtual Lab, Streamlit Cloud).
ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
TEST_CSV = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"

TARGET = "Revenue"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Weekend",
]
# OperatingSystems, Browser, Region and TrafficType are integer IDs, not
# quantities, so they are treated as categories.
CATEGORICAL_COLS = [
    "Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType",
]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast booleans to 0/1 so the same code works for the raw file and for
    any CSV a user uploads in the Streamlit app."""
    df = df.copy()
    df["Weekend"] = df["Weekend"].astype(int)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(int)
    return df


def load_and_split():
    """Return X_train, X_test, y_train, y_test and write test_data.csv."""
    df = clean_frame(pd.read_csv(RAW_CSV))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,  # keep the ~15% purchase rate in both splits
        random_state=RANDOM_STATE,
    )

    # Held-out file that will be uploaded to the Streamlit app.
    # It keeps the target column, as the assignment requires.
    test_out = X_test.copy()
    test_out[TARGET] = y_test
    test_out.to_csv(TEST_CSV, index=False)

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                # dense output so GaussianNB can use the same preprocessor
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """The six metrics required by the assignment (binary problem)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def fit_evaluate_save(filename, classifier, split, preprocessor=None):
    """Wrap `classifier` with the shared preprocessor, train it on the training
    split, score it on the held-out split, save the pipeline to model/ and
    return the metrics dict. A model-specific `preprocessor` can be passed;
    otherwise the shared one is used."""
    X_train, X_test, y_train, y_test = split

    prep = preprocessor if preprocessor is not None else build_preprocessor()
    pipe = Pipeline(steps=[("prep", prep), ("clf", classifier)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    score = pipe.predict_proba(X_test)[:, 1]  # probability that Revenue = 1
    results = compute_metrics(y_test, pred, score)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / filename)
    return results