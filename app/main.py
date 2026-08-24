from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import sys

from bootstrap import ensure_artifacts

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model import LogisticRiskModel  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_risk_model.json"
REPORT_PATH = ROOT / "data" / "model_report.json"
DRIFT_PATH = ROOT / "data" / "drift_report.csv"
ensure_artifacts()

app = FastAPI(
    title="Credit Risk MLOps API",
    description="Credit default scoring API with model metrics and drift report endpoints.",
    version="1.0.0",
)
model = LogisticRiskModel.load(MODEL_PATH)


class Applicant(BaseModel):
    annual_income: float
    loan_amount: float
    interest_rate: float
    credit_score: int
    dti: float
    employment_years: float
    delinquencies_2y: int
    open_credit_lines: int
    prior_defaults: int
    loan_purpose: str


class BatchApplicants(BaseModel):
    applicants: list[Applicant]


def applicant_to_dict(applicant: Applicant) -> dict:
    if hasattr(applicant, "model_dump"):
        return applicant.model_dump()
    return applicant.dict()


def decision_from_probability(probability: float) -> str:
    if probability >= 0.30:
        return "decline"
    if probability >= 0.18:
        return "review"
    return "approve"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded"}


@app.get("/metrics")
def metrics() -> dict:
    if not REPORT_PATH.exists():
        return {"error": "model report not found; run python src/train.py"}
    return json.loads(REPORT_PATH.read_text())


@app.get("/drift")
def drift() -> list[dict]:
    if not DRIFT_PATH.exists():
        return [{"error": "drift report not found; run python src/drift.py"}]
    return pd.read_csv(DRIFT_PATH).to_dict(orient="records")


@app.post("/score")
def score(applicant: Applicant) -> dict[str, object]:
    row = pd.DataFrame([applicant_to_dict(applicant)])
    probability = float(model.predict_proba(row)[0])
    return {
        "probability_of_default": round(probability, 4),
        "decision": decision_from_probability(probability),
        "reason_codes": model.reason_codes(row)[0],
    }


@app.post("/score-batch")
def score_batch(batch: BatchApplicants) -> dict[str, list[dict[str, object]]]:
    rows = pd.DataFrame([applicant_to_dict(applicant) for applicant in batch.applicants])
    probabilities = model.predict_proba(rows)
    reasons = model.reason_codes(rows)
    scores = [
        {
            "probability_of_default": round(float(probability), 4),
            "decision": decision_from_probability(float(probability)),
            "reason_codes": reason_codes[index],
        }
        for index, probability in enumerate(probabilities)
    ]
    return {"scores": scores}
