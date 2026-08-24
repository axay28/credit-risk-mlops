import pandas as pd

from src.generate_data import make_dataset
from src.model import LogisticRiskModel


def test_calibration_round_trip_and_reason_codes(tmp_path) -> None:
    data = make_dataset(n=400, seed=3)
    model = LogisticRiskModel(epochs=50).fit(data.iloc[:300]).fit_calibrator(
        data.iloc[300:], epochs=50
    )
    before = model.predict_proba(data.iloc[:2])
    reasons = model.reason_codes(data.iloc[:2], top_n=2)
    path = tmp_path / "model.json"
    model.save(path)
    restored = LogisticRiskModel.load(path)

    assert restored.predict_proba(data.iloc[:2]).tolist() == before.tolist()
    assert len(reasons) == 2
    assert all(len(row) <= 2 for row in reasons)
    for applicant, applicant_reasons in zip(data.iloc[:2].itertuples(), reasons):
        purpose_features = [
            reason["feature"] for reason in applicant_reasons if reason["feature"].startswith("purpose_")
        ]
        assert purpose_features in ([], [f"purpose_{applicant.loan_purpose}"])
