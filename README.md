# Phishing URL Detection with Machine Learning

**🔗 [Live Demo](https://phisingdetectionwebapp-inspironman.streamlit.app/)** 

Detect whether a URL is **phishing** or **legitimate** using only the text of the
link — no page fetching, no external lookups. Three classifiers are trained and
compared on ~235,000 URLs, followed by an **ablation study that shows the
headline 99% accuracy is partly a dataset artifact** rather than real-world skill.

> The interesting part of this project isn't the 99% accuracy — it's the analysis
> proving *why* that number overstates real-world performance.

---

## Key results

| Model | Accuracy | Phishing F1 | ROC-AUC |
|-------|----------|-------------|---------|
| **Linear SVM** (deployed) | 0.990 | **0.988** | 0.997 |
| Logistic Regression | 0.986 | 0.983 | 0.996 |
| Multinomial Naive Bayes | 0.963 | 0.956 | 0.992 |

Evaluated on a held-out 20% test set (47,159 URLs). Phishing is the **positive
class**, so precision/recall/F1 measure how well phishing is actually caught.

---

## The honest finding

The scores look excellent — but an ablation revealed two things worth reporting:

1. **URL normalization slightly *hurt* performance** — raw character n-grams
   scored 0.996 F1 versus 0.983 for the normalized version.
2. **12 trivial structural features (URL length, slash count, digit count…)
   reach 0.992 F1 on their own** — nearly matching a 50,000-feature n-gram model.

A dataset-bias diagnostic explains why: in this dataset, benign URLs are almost
always bare domains (0% contain a path), while phishing URLs are longer and
often carry paths. So the classifiers can largely separate the classes using
trivial structure — a **dataset construction artifact**, not a robust signal.

**Takeaway:** the reported accuracy would not fully hold on realistic data where
legitimate URLs also contain paths and queries. This is documented as the primary
threat to validity rather than hidden.

---

## Approach

```
Raw URL → normalize → character n-grams (3–5) → TF-IDF → classifier → phishing / legitimate
```

- **Features:** character n-grams (3–5 chars) weighted with TF-IDF. Character-level
  features catch typosquatting and generalize to unseen domains.
- **Models:** Logistic Regression, Linear SVM, Multinomial Naive Bayes — trained on
  identical features for a fair comparison.
- **Evaluation:** 80/20 stratified split, phishing as the positive class, plus a
  domain-leakage check (only 8.1% of test URLs share a domain with training).
- **Deployment:** the best model (calibrated Linear SVM) is served in an
  interactive Streamlit app.

---

## Repository structure

```
.
├── train_and_compare.py   # trains & compares the 3 models, saves the best
├── ablation.py            # design-choice experiments (the honest finding)
├── app.py                 # Streamlit demo — paste a URL, get a verdict
├── requirements.txt
├── data/                  # place phishing_simple.csv here
├── models/                # best_model.pkl (generated)
└── reports/               # metrics tables & plots (generated)
```

---

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. put the dataset at data/phishing_simple.csv, then train
python train_and_compare.py

# 3. reproduce the ablation study
python ablation.py

# 4. launch the interactive demo
streamlit run app.py
```

Run all commands from the project root.

---

## Dataset

~235,795 labelled URLs (57% benign / 43% phishing) with columns `URL`, `Domain`,
`label`, `length`. Sourced from Kaggle.
*(Not included in this repo — download separately and place at `data/phishing_simple.csv`.)*

---

## Tech stack

Python · scikit-learn · pandas · Streamlit · matplotlib / seaborn

---

## Possible improvements

- Train on a benign corpus containing realistic path-bearing URLs (e.g. Common
  Crawl samples) to remove the length/path artifact.
- Use a **domain-disjoint** train/test split for a stricter generalization test.
- Add richer features (domain age, TLS certificate info, host reputation).
- Expect accuracy to *drop* on fairer data — that lower number would be the more
  trustworthy one.

---

## License

MIT 

---

*Built as a Master's-level ML course project (University of Genova).*