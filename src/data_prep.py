from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_train_merged(data_dir: str | Path) -> pd.DataFrame:
    """Load train_transaction + train_identity and merge on TransactionID."""
    data_dir = Path(data_dir)
    tt = pd.read_csv(data_dir / "train_transaction.csv")
    ti = pd.read_csv(data_dir / "train_identity.csv")
    df = tt.merge(ti, on="TransactionID", how="left")
    return df


def high_missing_cols(
    df: pd.DataFrame,
    threshold: float,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Columns whose missing-rate > threshold, excluding selected columns."""
    exclude_set = set(exclude)
    miss = df.isna().mean()
    return [c for c, r in miss.items() if (r > threshold and c not in exclude_set)]


def time_split(df: pd.DataFrame, time_col: str, q: float):
    """Split by time quantile: early -> train, later -> val."""
    cut = float(df[time_col].quantile(q))
    train_df = df[df[time_col] <= cut].copy()
    val_df = df[df[time_col] > cut].copy()
    return train_df, val_df, cut
