# Phishing URL Detection

Lexical (character n-gram) phishing URL classifier with a model comparison,
an ablation study, and a Streamlit demo. Machine Learning course project.

## Layout
```
phishing_project/
  data/phishing_simple.csv      # dataset (URL, Domain, label, length)
  train_and_compare.py          # train + compare 3 models, save best
  ablation.py                   # design-justification experiments
  app.py                        # Streamlit demo
  requirements.txt
  reports/                      # tables + plots (created by the scripts)
  models/                       # best_model.pkl (created by training)
```

## Setup
```bash
pip install -r requirements.txt
```
The scripts look for the CSV at `data/phishing_simple.csv` (edit the
`CANDIDATE_PATHS` list at the top of each script if yours lives elsewhere).

## Run
```bash
python train_and_compare.py     # writes reports/ + models/best_model.pkl
python ablation.py              # writes ablation tables + plot
streamlit run app.py            # interactive demo (needs the model first)
```

## What each output is for (slide mapping)
| File | Slide |
|---|---|
| `reports/tables/dataset_bias_diagnostic.csv` | Threats to validity |
| `reports/tables/domain_leakage_check.csv`    | Evaluation setup |
| `reports/tables/model_comparison.csv`        | Results |
| `reports/plots/confusion_matrix_*.png`       | Results |
| `reports/plots/roc_curves.png`               | Results |
| `reports/tables/ablation_results.csv`        | Why these choices |
| `reports/tables/engineered_feature_weights.csv` | Why these choices / the artifact |
| `reports/plots/ablation_comparison.png`      | Why these choices |

## Key results (held-out 20% test set)
- Best model: **Linear SVM** — accuracy 0.990, phishing F1 0.988, AUC 0.997.
- Convention: **benign = 0, phishing = 1** (phishing is the positive class),
  so precision/recall/F1 measure how well phishing is caught.

## Honest caveats (discuss these in the oral)
- In this dataset benign URLs are almost all bare domains (mean 27 chars,
  0% with a path) while phishing URLs are longer (mean 46 chars) and 27% carry
  a path. 12 trivial structural features alone reach ~0.99 F1 — so the high
  scores partly reflect a dataset construction artifact, not real-world skill.
- Random split: ~8% of test URLs share a domain with training (modest
  leakage). A domain-disjoint split would give a more conservative estimate.
