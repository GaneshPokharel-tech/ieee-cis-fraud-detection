![CI](https://github.com/GaneshPokharel-tech/ieee-cis-fraud-detection/actions/workflows/ci.yml/badge.svg)

# IEEE-CIS Fraud Detection (ML + Explainability)

This repo is a reproducible ML project using the Kaggle IEEE-CIS Fraud Detection dataset.
Workflow: EDA -> LightGBM baseline -> evaluation -> SHAP explainability.

## Results (time-based split)
- Split: 80/20 by TransactionDT
- PR-AUC: 0.5427
- ROC-AUC: 0.9052
- Best-F1 threshold: 0.7293

## Visuals
**PR Curve**
![](reports/figures/pr_curve_lgbm.png)

**ROC Curve**
![](reports/figures/roc_curve_lgbm.png)

**Confusion Matrix**
![](reports/figures/confusion_matrix_lgbm.png)

**Top Feature Importance**
![](reports/figures/feature_importance_top20.png)

**SHAP Summary**
![](reports/figures/shap_summary_bar.png)


## Structure
- notebooks/ : EDA only
- src/       : training/eval/explain scripts
- configs/   : parameters (paths, split)
- reports/   : metrics + figures

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## How to Run (reproducible)

### 1) Train baseline (LightGBM)
```bash
source .venv/bin/activate
python -m src.train_lgbm --config configs/baseline.yaml

