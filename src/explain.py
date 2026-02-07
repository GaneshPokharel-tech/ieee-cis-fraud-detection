from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import shap
from lightgbm import LGBMClassifier

from src.data_prep import high_missing_cols, load_train_merged, time_split


def prepare_xy(df: pd.DataFrame, target: str, drop_cols: list[str]):
    X = df.drop(columns=[target] + drop_cols)
    y = df[target].astype(int)

    # LightGBM categorical support
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
    train_df, val_df, _ = time_split(df, time_col=time_col, q=q)

    drop_missing = high_missing_cols(
        train_df,
        threshold=miss_th,
        exclude=(target, id_col, time_col),
    )
    drop_cols = [id_col, time_col] + drop_missing

    X_train, y_train = prepare_xy(train_df, target, drop_cols)
    X_val, y_val = prepare_xy(val_df, target, drop_cols)

    # imbalance handling
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
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    # SHAP on a sample (fast + stable)
    n = min(5000, len(X_val))
    Xs = X_val.sample(n=n, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    # shap_values can be list (binary) or array depending on version
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Summary bar plot
    shap.summary_plot(sv, Xs, plot_type="bar", show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary_bar.png", dpi=150)
    plt.close()

    # Also save top features (by mean |SHAP|)
    mean_abs = np.abs(sv).mean(axis=0)
    top = (
        pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(30)
    )
    Path("reports").mkdir(parents=True, exist_ok=True)
    top.to_csv("reports/shap_top30.csv", index=False)

    print("Saved:", str(fig_dir / "shap_summary_bar.png"))
    print("Saved: reports/shap_top30.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
