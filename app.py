"""
Phishing URL Detector - Streamlit demo
======================================
Loads the model saved by train_and_compare.py and scores a pasted URL.

Label convention (matches training): benign = 0, phishing = 1.
So predict_proba[:, 1] is P(phishing) and a prediction of 1 means phishing.

Run:  streamlit run app.py
"""

import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️",
                   layout="centered")

MODEL_PATH = Path("models/best_model.pkl")
MODEL_NAME_PATH = Path("models/best_model_name.txt")
COMPARISON_PATH = Path("reports/tables/model_comparison.csv")
SUMMARY_PATH = Path("reports/final_model_summary.txt")


# --------------------------------------------------------------------------
# URL normalization - MUST match train_and_compare.py exactly
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


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------------------------------
# Guard: model must exist
# --------------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_and_compare.py` first.")
    st.stop()

model = load_model()
best_model_name = (
    MODEL_NAME_PATH.read_text(encoding="utf-8").strip()
    if MODEL_NAME_PATH.exists() else "Best Model"
)
comparison_df = pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else None
summary_text = SUMMARY_PATH.read_text(encoding="utf-8") if SUMMARY_PATH.exists() else None

has_proba = hasattr(model.named_steps["clf"], "predict_proba")

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🛡️ Phishing URL Detector")
st.write(
    "Paste a URL. It is normalized, then scored by the selected model "
    "(benign = 0, phishing = 1)."
)

with st.expander("Selected model details"):
    st.write(f"Deployed model: **{best_model_name}**")
    if summary_text:
        st.text(summary_text)

url_input = st.text_input(
    "Full URL", placeholder="https://example.com/login?session=123"
)

if st.button("Analyze URL"):
    if not url_input.strip():
        st.warning("Please enter a URL.")
    else:
        normalized = normalize_url(url_input)
        prediction = int(model.predict([normalized])[0])  # 1 = phishing

        st.markdown("---")
        st.subheader("Detection result")
        st.write(f"Normalized URL scored: `{normalized}`")

        if has_proba:
            proba = model.predict_proba([normalized])[0]
            p_benign, p_phishing = float(proba[0]), float(proba[1])
            risk_score = p_phishing * 100
            confidence = max(proba) * 100

            c1, c2, c3 = st.columns(3)
            with c1:
                if prediction == 1:
                    st.error("⚠️ Phishing")
                else:
                    st.success("✅ Legitimate")
            with c2:
                st.metric("Confidence", f"{confidence:.2f}%")
            with c3:
                if risk_score < 30:
                    level = "🟢 Low"
                elif risk_score < 70:
                    level = "🟡 Medium"
                else:
                    level = "🔴 High"
                st.metric("Risk level", level)

            st.metric("Phishing risk score", f"{risk_score:.1f} / 100")
            st.progress(min(max(p_phishing, 0.0), 1.0))
            st.caption(
                f"P(phishing) = {p_phishing:.4f}   |   "
                f"P(benign) = {p_benign:.4f}"
            )
        else:
            if prediction == 1:
                st.error("⚠️ Phishing")
            else:
                st.success("✅ Legitimate")
            st.info("This model does not expose calibrated probabilities.")

if comparison_df is not None:
    st.markdown("---")
    st.subheader("Model comparison (held-out test set)")
    st.dataframe(comparison_df, width='stretch')
    st.caption(
        "Note: scores are high partly because, in this public dataset, benign "
        "URLs are almost all bare domains while phishing URLs carry paths and "
        "are longer. See the report's threats-to-validity section."
    )
