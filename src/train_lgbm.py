from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_prep import high_missing_cols, load_train_merged, time_split


def prepare_xy(df: pd.DataFrame, target: str, drop_cols: list[str]):
    X = df.drop(columns=[target] + drop_cols)
    y = df[target].astype(int)

    # LightGBM can handle categorical if dtype is 'category'
    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].astype("category")

    return X, y


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    data_dir = cfg["data_dir"]
    target = cfg["target"]
    time_col = cfg["time_col"]
    id_col = cfg["id_col"]
    q = float(cfg["time_split_quantile"])
    miss_th = float(cfg["drop_high_missing_threshold"])

    df = load_train_merged(data_dir)
    train_df, val_df, cut = time_split(df, time_col=time_col, q=q)

    drop_missing = high_missing_cols(
        train_df,
        threshold=miss_th,
        exclude=(target, id_col, time_col),
    )

    # time used only for split
    drop_cols = [id_col, time_col] + drop_missing

    X_train, y_train = prepare_xy(train_df, target, drop_cols)
    X_val, y_val = prepare_xy(val_df, target, drop_cols)

    # class imbalance handling
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / max(pos, 1))

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.set_params(scale_pos_weight=scale_pos_weight)

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    pr_auc = float(average_precision_score(y_val, proba))
    roc_auc = float(roc_auc_score(y_val, proba))

    print(f"time split cut ({q:.2f} quantile): {cut}")
    print(f"train size: {X_train.shape} | val size: {X_val.shape}")
    print(f"dropped high-missing cols (> {miss_th:.2f}): {len(drop_missing)}")
    print("PR-AUC:", pr_auc)
    print("ROC-AUC:", roc_auc)

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/lgbm_metrics.json").write_text(
        json.dumps(
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "time_split_quantile": q,
                "time_split_cut": cut,
                "dropped_high_missing_cols": len(drop_missing),
                "train_shape": list(X_train.shape),
                "val_shape": list(X_val.shape),
                "scale_pos_weight": float(scale_pos_weight),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
