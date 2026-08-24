from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from model import LogisticRiskModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_risk_model.json"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/score.py '{\"annual_income\": 900000, ...}'")

    applicant = json.loads(sys.argv[1])
    model = LogisticRiskModel.load(MODEL_PATH)
    row = pd.DataFrame([applicant])
    pd_default = float(model.predict_proba(row)[0])
    decision = "decline" if pd_default >= 0.30 else "review" if pd_default >= 0.18 else "approve"

    print(
        json.dumps(
            {
                "probability_of_default": round(pd_default, 4),
                "decision": decision,
                "reason_codes": model.reason_codes(row)[0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
