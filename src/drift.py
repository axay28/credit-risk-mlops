from __future__ import annotations

from pathlib import Path

import pandas as pd

from metrics import psi
from model import NUMERIC_FEATURES, LogisticRiskModel


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "credit_risk_model.json"


def main() -> None:
    reference = pd.read_csv(DATA_DIR / "loans_train.csv")
    current = pd.read_csv(DATA_DIR / "loans_current.csv")
    model = LogisticRiskModel.load(MODEL_PATH)

    rows = []
    for feature in NUMERIC_FEATURES:
        value = psi(reference[feature].to_numpy(), current[feature].to_numpy())
        rows.append(
            {
                "feature": feature,
                "psi": value,
                "status": "investigate" if value >= 0.20 else "watch" if value >= 0.10 else "stable",
            }
        )

    reference_scores = model.predict_proba(reference)
    current_scores = model.predict_proba(current)
    rows.append(
        {
            "feature": "model_score",
            "psi": psi(reference_scores, current_scores),
            "status": "investigate" if psi(reference_scores, current_scores) >= 0.20 else "watch",
        }
    )

    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    report.to_csv(DATA_DIR / "drift_report.csv", index=False)
    print(report.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

