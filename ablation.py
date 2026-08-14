"""
Ablation Study - justifying the design choices
==============================================
(Logistic Regression) so that only the *features* change:

  1. Does URL normalization actually help, or hurt?
        raw char n-grams   vs   normalized char n-grams
  2. Do hand-crafted lexical features add anything on top of n-grams?
        normalized n-grams  vs  normalized n-grams + engineered features
  3. THE HONESTY CHECK: how far do a handful of trivial structural features
     (length, has_path, ...) get on their own? If this is already very high,
     it proves the dataset is largely separable by a construction artifact.

Outputs:
  reports/tables/ablation_results.csv
  reports/tables/engineered_feature_weights.csv
  reports/plots/ablation_comparison.png
"""

import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RANDOM_STATE = 42
CANDIDATE_PATHS = [
    Path("data/phishing_simple.csv"),
    Path("phishing_simple.csv"),
    Path("E:/ML_Project/data/phishing_simple (1).csv"),
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])
TABLES_DIR = Path("reports/tables")
PLOTS_DIR = Path("reports/plots")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"phishing": 1, "benign": 0}


# --------------------------------------------------------------------------
# URL helpers
# --------------------------------------------------------------------------
def ensure_scheme(url: str) -> str:
    url = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def normalize_url(url: str) -> str:
    url = ensure_scheme(url).strip().lower()
    p = urlparse(url)
    host = p.hostname or ""
    port = f":{p.port}" if p.port else ""
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+$", "", re.sub(r"/+", "/", p.path or ""))
    parts = [x for x in path.split("/") if x]
    cp = [re.sub(r"\d+", "NUM", x) for x in parts[:2]]
    cpath = "/" + "/".join(cp) if cp else ""
    return urlunparse((p.scheme, host + port, cpath, "", "", ""))


ENGINEERED_FEATURES = [
    "url_length", "hostname_length", "path_length",
    "num_dots", "num_hyphens", "num_digits", "num_slashes",
    "num_special", "num_subdomains", "has_ip", "has_at", "is_https",
]


def extract_features(url: str) -> list:
    """Cheap, interpretable lexical features from the raw URL."""
    u = ensure_scheme(url)
    p = urlparse(u)
    host = p.hostname or ""
    path = p.path or ""
    return [
        len(u),
        len(host),
        len(path),
        u.count("."),
        u.count("-"),
        sum(c.isdigit() for c in u),
        u.count("/"),
        sum(c in "@%=&?_~" for c in u),
        max(host.count(".") - 1, 0),
        int(bool(re.search(r"^\d{1,3}(\.\d{1,3}){3}$", host))),
        int("@" in u),
        int(p.scheme == "https"),
    ]


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
print(f"Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH).dropna(subset=["URL", "label"]).copy()
df["URL"] = df["URL"].astype(str).str.strip()
df["y"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
df = df.dropna(subset=["y"])
df["y"] = df["y"].astype(int)

df["raw_url"] = df["URL"].str.lower()
df["norm_url"] = df["URL"].apply(normalize_url)
feat = np.array([extract_features(u) for u in df["URL"]], dtype=float)

y = df["y"].values
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(
    idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
y_tr, y_te = y[idx_tr], y[idx_te]


def make_tfidf():
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=2, max_features=50000, lowercase=True,
    )


def evaluate(name, X_tr, X_te):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    score = clf.predict_proba(X_te)[:, 1]
    row = {
        "Configuration": name,
        "Accuracy": accuracy_score(y_te, pred),
        "F1_phishing": f1_score(y_te, pred, pos_label=1),
        "ROC_AUC": roc_auc_score(y_te, score),
        "N_features": X_tr.shape[1],
    }
    print(f"  {name:<38} acc={row['Accuracy']:.4f}  "
          f"F1={row['F1_phishing']:.4f}  AUC={row['ROC_AUC']:.4f}")
    return row, clf


rows = []

# 1) Raw char n-grams
print("\n[1] Raw URL char n-grams")
v = make_tfidf()
Xtr = v.fit_transform(df["raw_url"].values[idx_tr])
Xte = v.transform(df["raw_url"].values[idx_te])
rows.append(evaluate("Raw char n-grams", Xtr, Xte)[0])

# 2) Normalized char n-grams (the deployed representation)
print("[2] Normalized URL char n-grams  (deployed)")
v = make_tfidf()
Xtr_n = v.fit_transform(df["norm_url"].values[idx_tr])
Xte_n = v.transform(df["norm_url"].values[idx_te])
rows.append(evaluate("Normalized char n-grams", Xtr_n, Xte_n)[0])

# 3) Normalized n-grams + engineered lexical features
print("[3] Normalized n-grams + engineered features")
scaler = StandardScaler()
f_tr = csr_matrix(scaler.fit_transform(feat[idx_tr]))
f_te = csr_matrix(scaler.transform(feat[idx_te]))
rows.append(evaluate("Normalized n-grams + features",
                     hstack([Xtr_n, f_tr]).tocsr(),
                     hstack([Xte_n, f_te]).tocsr())[0])

# 4) Engineered features ONLY  (the honesty check)
print("[4] Engineered features ONLY  (artifact check)")
row, clf_feat = evaluate("Engineered features only", f_tr, f_te)
rows.append(row)

# --------------------------------------------------------------------------
# Save results
# --------------------------------------------------------------------------
res = pd.DataFrame(rows)
res.round(4).to_csv(TABLES_DIR / "ablation_results.csv", index=False)
print("\n=== Ablation summary ===")
print(res.round(4).to_string(index=False))

# Which engineered features drive the "features-only" model?
weights = (
    pd.DataFrame(
        {"feature": ENGINEERED_FEATURES, "weight": clf_feat.coef_[0]}
    )
    .assign(abs_weight=lambda d: d["weight"].abs())
    .sort_values("abs_weight", ascending=False)
    .drop(columns="abs_weight")
)
weights.round(4).to_csv(TABLES_DIR / "engineered_feature_weights.csv", index=False)
print("\nTop engineered features (features-only model):")
print(weights.head(6).to_string(index=False))

# Plot
plt.figure(figsize=(8, 4.5))
order = res.sort_values("F1_phishing")
sns.barplot(data=order, y="Configuration", x="F1_phishing",
            hue="Configuration", palette="crest", legend=False)
plt.xlim(0, 1)
plt.title("Ablation - Phishing F1 by feature configuration")
for i, val in enumerate(order["F1_phishing"]):
    plt.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ablation_comparison.png", dpi=300)
plt.close()

print("\nSaved: reports/tables/ablation_results.csv, "
      "engineered_feature_weights.csv, reports/plots/ablation_comparison.png")
