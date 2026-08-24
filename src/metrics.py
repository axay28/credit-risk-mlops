from __future__ import annotations

import numpy as np
import pandas as pd


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must be one-dimensional arrays of equal length")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score must contain only finite values")

    # Average ranks are required for tied predictions. Assigning arbitrary sequential
    # ranks makes AUC depend on the input order when multiple applicants share a score.
    ranks = pd.Series(y_score).rank(method="average").to_numpy()

    positives = y_true == 1
    n_pos = positives.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2 * roc_auc(y_true, y_score) - 1


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    return float(np.mean((y_score - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, bins: int = 10
) -> float:
    table = calibration_table(y_true, y_score, bins=bins)
    total = max(int(table["accounts"].sum()), 1)
    gaps = (table["predicted_pd"] - table["observed_default_rate"]).abs()
    return float((gaps * table["accounts"] / total).sum())


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


def threshold_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exposure: np.ndarray,
    thresholds: np.ndarray | None = None,
    loss_given_default: float = 0.45,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    if not (y_true.shape == y_score.shape == exposure.shape):
        raise ValueError("labels, scores, and exposure must have the same shape")
    thresholds = thresholds if thresholds is not None else np.arange(0.05, 0.51, 0.025)
    rows = []
    total_defaults = max(int(y_true.sum()), 1)
    for threshold in thresholds:
        approved = y_score < threshold
        approved_count = int(approved.sum())
        approved_defaults = int(y_true[approved].sum())
        rows.append(
            {
                "threshold": float(threshold),
                "approval_rate": float(approved.mean()),
                "approved_bad_rate": approved_defaults / max(approved_count, 1),
                "defaults_captured": 1 - approved_defaults / total_defaults,
                "expected_loss": float(
                    (exposure[approved] * y_score[approved] * loss_given_default).sum()
                ),
            }
        )
    return pd.DataFrame(rows)
