# IEEE-CIS Fraud Detection (ML + Explainability)

This repo is a reproducible ML project using the Kaggle IEEE-CIS Fraud Detection dataset.
Workflow: EDA -> LightGBM baseline -> evaluation -> SHAP explainability.

## Results (time-based split)
- Split: 80/20 by TransactionDT
- PR-AUC: 0.5427
- ROC-AUC: 0.9052
- Best-F1 threshold: 0.7293

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

