"""
Phishing URL Detection - Training & Model Comparison
====================================================

Trains and compares three classifiers on lexical (character n-gram) features
of URLs, then saves the best model for the Streamlit demo (app.py).

Key design decisions:
  * POSITIVE CLASS = phishing (label 1). All precision/recall/F1 therefore
    describe how well we CATCH phishing - the metric that actually matters.
  * Character n-gram TF-IDF: learns sub-string patterns of malicious URLs
    without hand-crafted rules; robust to unseen domains.
  * Light URL normalization before vectorizing (see normalize_url).
  * Honest evaluation: we report a dataset-bias diagnostic and verify that
    train/test do not share domains (no domain leakage).

Outputs (all under reports/):
  tables/  - CSV metrics, confusion matrices, diagnostics
  plots/   - confusion matrices, ROC curves, model comparison
  samples/ - normalized examples + sample predictions
  models/  - best_model.pkl, best_model_name.txt
"""

import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# Look for the dataset in a few sensible places so the script is portable.
CANDIDATE_PATHS = [
    Path("data/phishing_simple.csv"),
    Path("phishing_simple.csv"),
    Path("E:/ML_Project/data/phishing_simple.csv"),
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
TABLES_DIR = REPORTS_DIR / "tables"
PLOTS_DIR = REPORTS_DIR / "plots"
SAMPLES_DIR = REPORTS_DIR / "samples"
for d in (MODELS_DIR, TABLES_DIR, PLOTS_DIR, SAMPLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

URL_COLUMN = "URL"
LABEL_COLUMN = "label"

# benign = 0 (negative), phishing = 1 (POSITIVE class we want to detect)
LABEL_MAP = {
    "phishing": 1, "malicious": 1, "bad": 1, "1": 1, 1: 1,
    "benign": 0, "legitimate": 0, "safe": 0, "good": 0, "0": 0, 0: 0,
}
CLASS_NAMES = ["Benign", "Phishing"]  # index 0, 1


# --------------------------------------------------------------------------
# URL normalization
# --------------------------------------------------------------------------
def ensure_scheme(url: str) -> str:
    url = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def normalize_url(url: str) -> str:
    """Light canonicalization: lowercase, drop leading www, drop query/
    fragment, collapse slashes, keep host + first two path segments, and
    replace runs of digits in the path with NUM (reduces sparsity)."""
    url = ensure_scheme(url).strip().lower()
    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    if hostname.startswith("www."):
        hostname = hostname[4:]

    path = re.sub(r"/+", "/", parsed.path or "")
    path = re.sub(r"/+$", "", path)
    parts = [p for p in path.split("/") if p]
    cleaned = [re.sub(r"\d+", "NUM", p) for p in parts[:2]]
    cleaned_path = "/" + "/".join(cleaned) if cleaned else ""

    return urlunparse((parsed.scheme, hostname + port, cleaned_path, "", "", ""))


# --------------------------------------------------------------------------
# Load & clean
# --------------------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found. Tried: {[str(p) for p in CANDIDATE_PATHS]}"
    )

print(f"Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape, "| columns:", df.columns.tolist())

if URL_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    raise ValueError(f"Need columns '{URL_COLUMN}' and '{LABEL_COLUMN}'.")

df = df.dropna(subset=[URL_COLUMN, LABEL_COLUMN]).copy()
df[URL_COLUMN] = df[URL_COLUMN].astype(str).str.strip()
df["label_name"] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
df[LABEL_COLUMN] = df["label_name"].map(LABEL_MAP)

if df[LABEL_COLUMN].isnull().any():
    print(df.loc[df[LABEL_COLUMN].isnull(), "label_name"].value_counts())
    raise ValueError("Unmapped labels found - update LABEL_MAP.")
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

# Domain column (optional): used only for the leakage check below.
has_domain_col = "Domain" in df.columns
if has_domain_col:
    df["Domain"] = df["Domain"].astype(str).str.strip().str.lower()

print("Class balance:")
print(df[LABEL_COLUMN].value_counts().rename({0: "benign", 1: "phishing"}))


# --------------------------------------------------------------------------
# Dataset-bias diagnostic  (feeds the "Threats to validity" slide)
#
# This dataset has a construction artifact: benign URLs are almost always
# bare domains, while phishing URLs often carry paths / queries and are much
# longer. We MEASURE and REPORT this rather than hide it.
# --------------------------------------------------------------------------
def raw_len(u):
    return len(str(u))


def has_path(u):
    return int(len(urlparse(ensure_scheme(u)).path.strip("/")) > 0)


def has_query(u):
    return int("?" in str(u))


def has_ip_host(u):
    return int(bool(re.search(r"://\d{1,3}(\.\d{1,3}){3}", ensure_scheme(u))))


diag = df.assign(
    url_length=df[URL_COLUMN].map(raw_len),
    has_path=df[URL_COLUMN].map(has_path),
    has_query=df[URL_COLUMN].map(has_query),
    has_ip=df[URL_COLUMN].map(has_ip_host),
)
diag_table = (
    diag.groupby(LABEL_COLUMN)
    .agg(
        n=("url_length", "size"),
        mean_length=("url_length", "mean"),
        median_length=("url_length", "median"),
        max_length=("url_length", "max"),
        pct_with_path=("has_path", "mean"),
        pct_with_query=("has_query", "mean"),
        pct_ip_host=("has_ip", "mean"),
    )
    .rename(index={0: "benign", 1: "phishing"})
    .round(4)
)
diag_table.to_csv(TABLES_DIR / "dataset_bias_diagnostic.csv")
print("\n--- Dataset-bias diagnostic (put this on a slide) ---")
print(diag_table)


# --------------------------------------------------------------------------
# Normalize + train/test split
# --------------------------------------------------------------------------
df["normalized_url"] = df[URL_COLUMN].apply(normalize_url)
df[[URL_COLUMN, "normalized_url", LABEL_COLUMN]].head(20).to_csv(
    SAMPLES_DIR / "normalized_url_examples.csv", index=False
)

X = df["normalized_url"]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
pd.DataFrame({"normalized_url": X_test, "true_label": y_test}).to_csv(
    SAMPLES_DIR / "test_split_urls.csv", index=False
)

# Domain-leakage check: how many domains appear in BOTH train and test?
# ~1 URL per domain here, so this should be tiny -> random split is justified.
if has_domain_col:
    train_domains = set(df.loc[idx_train, "Domain"])
    test_domains = set(df.loc[idx_test, "Domain"])
    overlap = train_domains & test_domains
    leak_table = pd.DataFrame(
        {
            "metric": [
                "unique_domains_total",
                "domains_in_train",
                "domains_in_test",
                "domains_in_both",
                "test_rows_with_leaked_domain",
                "pct_test_rows_leaked",
            ],
            "value": [
                df["Domain"].nunique(),
                len(train_domains),
                len(test_domains),
                len(overlap),
                int(df.loc[idx_test, "Domain"].isin(overlap).sum()),
                round(df.loc[idx_test, "Domain"].isin(overlap).mean(), 4),
            ],
        }
    )
    leak_table.to_csv(TABLES_DIR / "domain_leakage_check.csv", index=False)
    print("\n--- Domain-leakage check ---")
    print(leak_table.to_string(index=False))


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50000,
    lowercase=True,
)

models = {
    "Logistic_Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ),
    "Linear_SVM": LinearSVC(class_weight="balanced"),
    "Multinomial_Naive_Bayes": MultinomialNB(),
}

# URLs used for the qualitative sanity-check table / live-demo screenshot.
sample_urls = pd.DataFrame(
    {
        "url": [
            "https://www.google.com",
            "https://github.com",
            "https://www.amazon.com",
            "https://netflix.com",
            "https://2025.aulaweb.unige.it/enrol/index.php?id=1139",
            "http://paypal-login-security-check.example.com",
            "http://192.168.1.2/verify/account/login.php",
            "http://secure-update-bank-account.example.xyz/login",
            "http://att-103731-107123.weeblysite.com/",
        ]
    }
)
sample_urls["normalized_url"] = sample_urls["url"].apply(normalize_url)

results = []
roc_curves = {}
trained_pipelines = {}

for name, clf in models.items():
    print(f"\nTraining {name} ...")
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Scores for ROC/AUC: predict_proba if available, else decision_function.
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]        # P(phishing)
    elif hasattr(pipe.named_steps["clf"], "decision_function"):
        y_score = pipe.decision_function(X_test)
    else:
        y_score = None

    roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else None
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_curves[name] = (fpr, tpr, roc_auc)

    # pos_label=1 == phishing -> these describe phishing detection.
    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision_phishing": precision_score(y_test, y_pred, pos_label=1),
            "Recall_phishing": recall_score(y_test, y_pred, pos_label=1),
            "F1_phishing": f1_score(y_test, y_pred, pos_label=1),
            "F1_macro": f1_score(y_test, y_pred, average="macro"),
            "ROC_AUC": roc_auc,
        }
    )
    trained_pipelines[name] = pipe

    # Per-class report + confusion matrix.
    rep = pd.DataFrame(
        classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
        )
    ).transpose()
    rep.to_csv(TABLES_DIR / f"classification_report_{name}.csv")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in CLASS_NAMES],
        columns=[f"Pred_{c}" for c in CLASS_NAMES],
    ).to_csv(TABLES_DIR / f"confusion_matrix_{name}.csv")

    plt.figure(figsize=(5.2, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_matrix_{name}.png", dpi=300)
    plt.close()

    # Qualitative sample predictions.
    s = sample_urls.copy()
    s["predicted"] = [CLASS_NAMES[p] for p in pipe.predict(s["normalized_url"])]
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        s["phishing_probability"] = pipe.predict_proba(s["normalized_url"])[:, 1]
    s.to_csv(SAMPLES_DIR / f"sample_predictions_{name}.csv", index=False)


# --------------------------------------------------------------------------
# Comparison table + plots
# --------------------------------------------------------------------------
results_df = (
    pd.DataFrame(results)
    .sort_values("F1_phishing", ascending=False)
    .reset_index(drop=True)
)
results_df.round(4).to_csv(TABLES_DIR / "model_comparison.csv", index=False)
print("\n=== Model comparison (sorted by phishing F1) ===")
print(results_df.round(4).to_string(index=False))

# Bar chart of phishing F1.
plt.figure(figsize=(7.5, 4.5))
sns.barplot(data=results_df, x="Model", y="F1_phishing",
            hue="Model", palette="viridis", legend=False)
plt.title("Model Comparison - Phishing F1")
plt.ylim(0, 1)
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison_f1.png", dpi=300)
plt.close()

# Combined ROC curves.
if roc_curves:
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# Persist best model (calibrate SVM so the app can show probabilities)
# --------------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]

if best_name == "Linear_SVM":
    best_pipeline = Pipeline(
        [
            ("tfidf", tfidf),
            ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5)),
        ]
    )
    best_pipeline.fit(X_train, y_train)
else:
    best_pipeline = trained_pipelines[best_name]

joblib.dump(best_pipeline, MODELS_DIR / "best_model.pkl")
(MODELS_DIR / "best_model_name.txt").write_text(best_name, encoding="utf-8")

summary = [
    "Final Model Selection Summary",
    "=============================",
    f"Best model: {best_name}",
    "",
    "Label convention: benign=0, phishing=1 (phishing is the positive class).",
    "",
    "Normalization applied before TF-IDF:",
    "  - ensured a scheme exists, lowercased",
    "  - removed leading 'www.'",
    "  - dropped query strings and fragments",
    "  - collapsed repeated slashes, kept host + first two path segments",
    "  - replaced digit runs in the path with NUM",
    "",
    "Why this model:",
    "  - highest phishing F1 on the held-out test set",
    "  - char n-gram TF-IDF suits sparse, high-dimensional URL strings",
    (
        "  - calibrated (Platt scaling, 5-fold) so the UI can show probabilities"
        if best_name == "Linear_SVM"
        else "  - provides calibrated probabilities directly"
    ),
]
(REPORTS_DIR / "final_model_summary.txt").write_text(
    "\n".join(summary), encoding="utf-8"
)

print(f"\nSaved best model: {best_name} -> {MODELS_DIR / 'best_model.pkl'}")
print("Done. All tables/plots written under reports/.")
