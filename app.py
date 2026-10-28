"""
Online Shopper Purchase Prediction - Streamlit app.

Loads the five saved scikit-learn pipelines from model/ and evaluates them on
an uploaded test CSV. Nothing is retrained here: every .joblib file already
contains its own preprocessing, so raw rows can be passed straight in.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

TARGET = "Revenue"
CLASS_NAMES = ["No purchase", "Purchase"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]


# ----------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False)
def load_model(display_name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[display_name])


def to_binary(series: pd.Series) -> pd.Series:
    """Turn True/False, TRUE/FALSE, 'True'/'False' or 0/1 into 0/1 integers."""
    if series.dtype == object:
        series = series.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        )
    return series.astype(int)


def validate_and_prepare(df: pd.DataFrame):
    """Return (X, y) or raise ValueError with a readable message."""
    needed = FEATURES + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("The CSV is missing these required columns: " + ", ".join(missing))

    data = df[needed].copy()
    if data.isna().any().any():
        raise ValueError("The CSV contains empty cells. Please upload the test file without missing values.")

    try:
        data["Weekend"] = to_binary(data["Weekend"])
        data[TARGET] = to_binary(data[TARGET])
    except (ValueError, TypeError):
        raise ValueError("`Weekend` and `Revenue` must be True/False or 1/0.")

    return data[FEATURES], data[TARGET]


def score_model(display_name: str, X: pd.DataFrame, y: pd.Series):
    """Predict with a saved pipeline and compute the six metrics."""
    pipe = load_model(display_name)
    pred = pipe.predict(X)
    prob = pipe.predict_proba(X)[:, 1]

    both_classes = y.nunique() == 2  # AUC is undefined with a single class
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob) if both_classes else np.nan,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred


def draw_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=13)
    fig.tight_layout()
    return fig


def draw_class_balance(y_true):
    counts = [int((y_true == 0).sum()), int((y_true == 1).sum())]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    bars = ax.bar(CLASS_NAMES, counts, color=["#9ecae1", "#3182bd"])
    ax.set_ylabel("Sessions")
    ax.set_title("Class balance of this file", fontsize=10)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n:,}\n({n / sum(counts):.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.25)
    fig.tight_layout()
    return fig


def draw_metric_bars(metrics: dict, model_name: str):
    """Bar chart of the six metrics for one model (y fixed to 0-1)."""
    names = list(metrics.keys())
    values = [0.0 if np.isnan(v) else v for v in metrics.values()]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(names, values, color="#3182bd")
    for bar, v in zip(bars, metrics.values()):
        label = "n/a" if np.isnan(v) else f"{v:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0), label,
                ha="center", va="bottom", fontsize=9)
    low = min(0.0, min(values))  # MCC can be negative
    ax.set_ylim(low, 1.1)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name}: all six metrics", fontsize=10)
    fig.tight_layout()
    return fig


def draw_model_comparison(table, metric_names=("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")):
    """Grouped bars: x = the five models, y = metric value (0 to 1)."""
    models = list(table.index)
    x = np.arange(len(models))
    width = 0.8 / len(metric_names)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, metric in enumerate(metric_names):
        ax.bar(x + (k - (len(metric_names) - 1) / 2) * width, table[metric],
               width, label=metric)
    ax.set_xticks(x, labels=models, rotation=15)
    ax.set_ylabel("Score")
    low = min(0.0, float(np.nanmin(table[list(metric_names)].values)))  # MCC can be negative
    ax.set_ylim(low, 1.0)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_title("Model comparison on the uploaded test data")
    ax.legend(ncol=len(metric_names), loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- page
st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒", layout="wide")

st.title("🛒 Online Shopper Purchase Predictor")
st.write(
    "Evaluate five saved classification models on browsing-session data and see "
    "which ones best predict whether a visit ends in a purchase."
)

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload a test CSV", type="csv")
    use_sample = False
    if SAMPLE_CSV.exists():
        use_sample = st.checkbox("Use the bundled test_data.csv instead")
    st.caption(
        "The file must contain the 17 input features and the `Revenue` column. "
        "Only test data should be uploaded (the models are not retrained)."
    )

    st.header("2. Model")
    chosen_model = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

# pick the data source
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source_label = uploaded.name
elif use_sample:
    raw = pd.read_csv(SAMPLE_CSV)
    source_label = "test_data.csv (bundled)"
else:
    st.info("👈 Upload a test CSV in the sidebar to get started.")
    with st.expander("Which columns are required?"):
        st.write(", ".join(FEATURES + [TARGET]))
    st.stop()

try:
    X_test, y_test = validate_and_prepare(raw)
except ValueError as err:
    st.error(str(err))
    st.stop()

# summary of the uploaded data
n_buy = int(y_test.sum())
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(y_test):,}")
c2.metric("Purchases", f"{n_buy:,}")
c3.metric("Purchase rate", f"{n_buy / len(y_test):.1%}")
with st.expander("Class balance of this file"):
    st.pyplot(draw_class_balance(y_test))
with st.expander(f"Preview of {source_label}"):
    st.dataframe(raw.head(20))

tab_single, tab_all = st.tabs([f"📊 {chosen_model}", "🏁 All models compared"])

# ----------------------------------------------------- selected model tab
with tab_single:
    try:
        metrics, predictions = score_model(chosen_model, X_test, y_test)
    except Exception as err:  # keep the app alive if a model file is missing or incompatible
        st.error(f"Could not run {chosen_model}: {err}")
        st.stop()

    st.subheader("Evaluation metrics")
    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}")
    if np.isnan(metrics["AUC"]):
        st.caption("AUC needs both classes in the uploaded file, so it cannot be computed here.")

    st.pyplot(draw_metric_bars(metrics, chosen_model))

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(draw_confusion_matrix(y_test, predictions, chosen_model))
    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T.round(3)
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df)
        st.caption("Precision, recall and F1 in the metrics above are for the 'Purchase' class.")

# --------------------------------------------------------- all models tab
with tab_all:
    st.subheader("Same test data, every model")
    rows = {}
    for name in MODEL_FILES:
        try:
            rows[name], _ = score_model(name, X_test, y_test)
        except Exception as err:
            st.warning(f"Skipped {name}: {err}")
    if rows:
        table = pd.DataFrame(rows).T.round(4)
        table.index.name = "ML Model Name"
        st.dataframe(table)
        best = table["MCC"].idxmax()
        st.success(f"Highest MCC on this file: **{best}** ({table.loc[best, 'MCC']:.4f})")
        st.pyplot(draw_model_comparison(table))
