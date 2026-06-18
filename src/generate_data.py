from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_dataset(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    annual_income = rng.lognormal(mean=np.log(850000), sigma=0.45, size=n).clip(180000, 3500000)
    credit_score = rng.normal(690, 65, size=n).clip(420, 850)
    loan_amount = rng.lognormal(mean=np.log(420000), sigma=0.55, size=n).clip(50000, 2500000)
    interest_rate = rng.normal(12.5, 4.0, size=n).clip(6, 32)
    dti = (loan_amount / annual_income + rng.normal(0.12, 0.08, size=n)).clip(0.02, 0.95)
    employment_years = rng.gamma(shape=2.2, scale=2.0, size=n).clip(0, 25)
    delinquencies_2y = rng.poisson(lam=np.clip((710 - credit_score) / 140, 0.05, 3.0))
    open_credit_lines = rng.poisson(lam=6, size=n).clip(1, 30)
    prior_defaults = rng.binomial(1, np.clip((650 - credit_score) / 500, 0.01, 0.35))
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "medical", "small_business", "education"],
        size=n,
        p=[0.42, 0.18, 0.14, 0.16, 0.10],
    )

    purpose_risk = {
        "debt_consolidation": 0.10,
        "home_improvement": -0.15,
        "medical": 0.25,
        "small_business": 0.45,
        "education": -0.05,
    }
    purpose_effect = np.array([purpose_risk[p] for p in loan_purpose])

    logit = (
        -4.2
        + 2.8 * dti
        + 0.055 * interest_rate
        + 0.45 * prior_defaults
        + 0.18 * delinquencies_2y
        - 0.0065 * (credit_score - 650)
        - 0.045 * employment_years
        + 0.00000038 * loan_amount
        - 0.00000018 * annual_income
        + purpose_effect
    )
    default_probability = sigmoid(logit)
    defaulted = rng.binomial(1, default_probability)

    return pd.DataFrame(
        {
            "annual_income": annual_income.round(2),
            "loan_amount": loan_amount.round(2),
            "interest_rate": interest_rate.round(2),
            "credit_score": credit_score.round(0).astype(int),
            "dti": dti.round(4),
            "employment_years": employment_years.round(1),
            "delinquencies_2y": delinquencies_2y.astype(int),
            "open_credit_lines": open_credit_lines.astype(int),
            "prior_defaults": prior_defaults.astype(int),
            "loan_purpose": loan_purpose,
            "defaulted": defaulted.astype(int),
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = make_dataset(n=12000, seed=42)
    current = make_dataset(n=3500, seed=7)
    current["interest_rate"] = (current["interest_rate"] + 1.25).clip(6, 32)
    current["dti"] = (current["dti"] + 0.04).clip(0.02, 0.95)

    train.to_csv(DATA_DIR / "loans_train.csv", index=False)
    current.to_csv(DATA_DIR / "loans_current.csv", index=False)
    print(f"Wrote {DATA_DIR / 'loans_train.csv'} ({len(train)} rows)")
    print(f"Wrote {DATA_DIR / 'loans_current.csv'} ({len(current)} rows)")


if __name__ == "__main__":
    main()

