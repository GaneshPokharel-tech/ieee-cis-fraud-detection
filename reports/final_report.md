# IEEE-CIS Fraud Detection — Final Report

## 1. Problem
Online transactions have fraud. The goal is to predict `isFraud` using transaction + identity features.

## 2. Data
- Source: Kaggle IEEE-CIS Fraud Detection
- train_transaction: 590,540 rows, 394 cols
- train_identity: 144,233 rows, 41 cols
- merged train: 590,540 rows, 434 cols
- class imbalance: about 3.5% fraud

## 3. Method
- Merge on TransactionID (left join)
- Time-based split using TransactionDT (80% train, 20% validation)
- Drop columns with >90% missing (computed on train only)
- Model: LightGBM baseline with class imbalance handling (scale_pos_weight)

## 4. Results
Validation:
- PR-AUC: 0.5427
- ROC-AUC: 0.9052
- Best-F1 threshold: 0.7293
- Confusion matrix: [[112617, 1427], [2106, 1958]]

## 5. Explainability (SHAP)
Top drivers (sample):
- Transaction amount
- Email domain features
- Card features (card1, card2, card6)
- Some C/V features

Saved files:
- reports/figures/shap_summary_bar.png
- reports/shap_top30.csv

## 6. Repo Usage
Train:
python -m src.train_lgbm --config configs/baseline.yaml

Evaluate:
python -m src.evaluate --config configs/baseline.yaml

Explain:
python -m src.explain --config configs/baseline.yaml

