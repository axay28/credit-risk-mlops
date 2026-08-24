from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "annual_income",
    "loan_amount",
    "interest_rate",
    "credit_score",
    "dti",
    "employment_years",
    "delinquencies_2y",
    "open_credit_lines",
    "prior_defaults",
]
PURPOSE_VALUES = ["debt_consolidation", "home_improvement", "medical", "small_business", "education"]
FEATURES = NUMERIC_FEATURES + [f"purpose_{p}" for p in PURPOSE_VALUES]
FEATURE_LABELS = {
    "annual_income": "Lower annual income",
    "loan_amount": "Higher requested loan amount",
    "interest_rate": "Higher interest rate",
    "credit_score": "Lower credit score",
    "dti": "Higher debt-to-income ratio",
    "employment_years": "Shorter employment history",
    "delinquencies_2y": "Recent delinquencies",
    "open_credit_lines": "Number of open credit lines",
    "prior_defaults": "Prior default history",
    **{f"purpose_{purpose}": f"Loan purpose: {purpose.replace('_', ' ')}" for purpose in PURPOSE_VALUES},
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def featurize(df: pd.DataFrame, stats: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    work = df.copy()
    for purpose in PURPOSE_VALUES:
        work[f"purpose_{purpose}"] = (work["loan_purpose"] == purpose).astype(float)

    raw = work[FEATURES].astype(float)
    if stats is None:
        means = raw.mean().to_dict()
        stds = raw.std(ddof=0).replace(0, 1).to_dict()
        stats = {"means": means, "stds": stds, "features": FEATURES}

    x = raw.copy()
    for col in FEATURES:
        x[col] = (x[col] - stats["means"][col]) / stats["stds"][col]
    return x.to_numpy(dtype=float), stats


class LogisticRiskModel:
    def __init__(self, lr: float = 0.08, epochs: int = 1400, l2: float = 0.02):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.stats: dict[str, Any] | None = None
        self.calibration_a = 1.0
        self.calibration_b = 0.0

    def fit(self, df: pd.DataFrame, target: str = "defaulted") -> "LogisticRiskModel":
        x, self.stats = featurize(df)
        y = df[target].to_numpy(dtype=float)
        self.weights = np.zeros(x.shape[1], dtype=float)
        self.bias = 0.0

        for _ in range(self.epochs):
            pred = sigmoid(x @ self.weights + self.bias)
            error = pred - y
            grad_w = (x.T @ error) / len(y) + self.l2 * self.weights
            grad_b = float(error.mean())
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        return self

    def decision_function(self, df: pd.DataFrame) -> np.ndarray:
        if self.weights is None or self.stats is None:
            raise RuntimeError("Model is not fitted")
        x, _ = featurize(df, self.stats)
        return x @ self.weights + self.bias

    def predict_proba(self, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        logits = self.decision_function(df)
        if calibrated:
            logits = self.calibration_a * logits + self.calibration_b
        return sigmoid(logits)

    def fit_calibrator(
        self,
        df: pd.DataFrame,
        target: str = "defaulted",
        lr: float = 0.05,
        epochs: int = 1000,
    ) -> "LogisticRiskModel":
        logits = self.decision_function(df)
        y = df[target].to_numpy(dtype=float)
        self.calibration_a = 1.0
        self.calibration_b = 0.0
        for _ in range(epochs):
            pred = sigmoid(self.calibration_a * logits + self.calibration_b)
            error = pred - y
            self.calibration_a -= lr * float(np.mean(error * logits))
            self.calibration_b -= lr * float(error.mean())
        return self

    def reason_codes(self, df: pd.DataFrame, top_n: int = 3) -> list[list[dict[str, float | str]]]:
        if self.weights is None or self.stats is None:
            raise RuntimeError("Model is not fitted")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        x, _ = featurize(df, self.stats)
        contributions = x * self.weights
        results = []
        for row_index, row in enumerate(contributions):
            risk_indices = np.argsort(row)[::-1]
            applicant_purpose = str(df.iloc[row_index]["loan_purpose"])
            positive_indices = [
                idx
                for idx in risk_indices
                if row[idx] > 0
                and (
                    not FEATURES[idx].startswith("purpose_")
                    or FEATURES[idx] == f"purpose_{applicant_purpose}"
                )
            ][:top_n]
            results.append(
                [
                    {
                        "feature": FEATURES[idx],
                        "reason": FEATURE_LABELS[FEATURES[idx]],
                        "contribution": round(float(row[idx]), 4),
                    }
                    for idx in positive_indices
                ]
            )
        return results

    def save(self, path: Path) -> None:
        if self.weights is None or self.stats is None:
            raise RuntimeError("Model is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "stats": self.stats,
            "features": FEATURES,
            "calibration": {"a": self.calibration_a, "b": self.calibration_b},
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LogisticRiskModel":
        payload = json.loads(path.read_text())
        model = cls()
        model.weights = np.array(payload["weights"], dtype=float)
        model.bias = float(payload["bias"])
        model.stats = payload["stats"]
        calibration = payload.get("calibration", {})
        model.calibration_a = float(calibration.get("a", 1.0))
        model.calibration_b = float(calibration.get("b", 0.0))
        return model

    def feature_importance(self) -> pd.DataFrame:
        if self.weights is None:
            raise RuntimeError("Model is not fitted")
        return pd.DataFrame({"feature": FEATURES, "coefficient": self.weights}).sort_values(
            "coefficient", ascending=False
        )
