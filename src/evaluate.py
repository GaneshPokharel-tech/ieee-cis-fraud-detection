from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)

from src.data_prep import high_missing_cols, load_train_merged, time_split


def prepare_xy(df: pd.DataFrame, target: str, drop_cols: list[str]):
    X = df.drop(columns=[target] + drop_cols)
    y = df[target].astype(int)

    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].astype("category")

    return X, y


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1[:-1]))
    return float(thr[best_idx])


def plot_confusion(cm, path: Path):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix (val)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    drop_cols = [id_col, time_col] + drop_missing

    X_train, y_train = prepare_xy(train_df, target, drop_cols)
    X_val, y_val = prepare_xy(val_df, target, drop_cols)

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

    proba = model.predict_proba(X_val)[:, 1]
    pr_auc = float(average_precision_score(y_val, proba))
    roc_auc = float(roc_auc_score(y_val, proba))

    thr = best_f1_threshold(y_val.values, proba)
    preds = (proba >= thr).astype(int)
    cm = confusion_matrix(y_val, preds)

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_val, proba)
    fig, ax = plt.subplots()
    ax.plot(rec, prec)
    ax.set_title("Precision-Recall Curve (val)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    plt.tight_layout()
    fig.savefig(fig_dir / "pr_curve_lgbm.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_val, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_title("ROC Curve (val)")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.tight_layout()
    fig.savefig(fig_dir / "roc_curve_lgbm.png", dpi=150)
    plt.close(fig)

    # Confusion matrix plot
    plot_confusion(cm, fig_dir / "confusion_matrix_lgbm.png")

    # Feature importance
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    imp = (
        pd.DataFrame({"feature": names, "gain": gains})
        .sort_values("gain", ascending=False)
        .head(20)
        .sort_values("gain", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"], imp["gain"])
    ax.set_title("Top 20 Feature Importance (gain)")
    plt.tight_layout()
    fig.savefig(fig_dir / "feature_importance_top20.png", dpi=150)
    plt.close(fig)

    Path("reports").mkdir(parents=True, exist_ok=True)
    booster.save_model("reports/lgbm_model.txt")

    Path("reports/eval_metrics.json").write_text(
        json.dumps(
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "best_f1_threshold": thr,
                "confusion_matrix": cm.tolist(),
                "time_split_quantile": q,
                "time_split_cut": cut,
                "dropped_high_missing_cols": len(drop_missing),
                "scale_pos_weight": float(scale_pos_weight),
            },
            indent=2,
        )
    )

    print("PR-AUC:", pr_auc)
    print("ROC-AUC:", roc_auc)
    print("Best F1 threshold:", thr)
    print("Confusion matrix:", cm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
