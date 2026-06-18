from __future__ import annotations

import numpy as np
import pandas as pd


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    positives = y_true == 1
    n_pos = positives.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2 * roc_auc(y_true, y_score) - 1


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    bad_total = max(df["y"].sum(), 1)
    good_total = max((1 - df["y"]).sum(), 1)
    bad_cdf = df["y"].cumsum() / bad_total
    good_cdf = (1 - df["y"]).cumsum() / good_total
    return float((bad_cdf - good_cdf).abs().max())


def capture_rate_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    top_n = max(int(len(df) * k), 1)
    total_bad = max(df["y"].sum(), 1)
    return float(df.head(top_n)["y"].sum() / total_bad)


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df["bucket"] = pd.qcut(df["score"], q=bins, duplicates="drop")
    table = (
        df.groupby("bucket", observed=True)
        .agg(
            accounts=("y", "size"),
            predicted_pd=("score", "mean"),
            observed_default_rate=("y", "mean"),
        )
        .reset_index(drop=True)
    )
    return table


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts = np.unique(cuts)
    if len(cuts) <= 2:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=cuts)
    actual_counts, _ = np.histogram(actual, bins=cuts)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, 1)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, 1)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

