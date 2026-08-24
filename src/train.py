from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from metrics import (
    brier_score,
    calibration_table,
    capture_rate_at_k,
    expected_calibration_error,
    gini,
    ks_statistic,
    roc_auc,
    threshold_table,
)
from model import LogisticRiskModel


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "credit_risk_model.json"
MLFLOW_DIR = ROOT / "mlruns"


def log_to_mlflow(report: dict[str, Any], model: LogisticRiskModel) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; skipping experiment logging.")
        return

    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("credit-risk-default-model")
    with mlflow.start_run(run_name="numpy-logistic-credit-risk"):
        mlflow.log_params({"model_family": "logistic_regression", "optimizer": "gradient_descent"})
        for key, value in report.items():
            mlflow.log_metric(key, float(value))
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(DATA_DIR / "calibration_table.csv"))
        mlflow.log_artifact(str(DATA_DIR / "feature_importance.csv"))
        mlflow.set_tag("use_case", "credit_default_prediction")
        mlflow.set_tag("deployment_target", "fastapi")
        mlflow.log_text(model.feature_importance().to_string(index=False), "feature_importance.txt")


def main() -> None:
    df = pd.read_csv(DATA_DIR / "loans_train.csv")
    train = df.sample(frac=0.7, random_state=42)
    remainder = df.drop(train.index)
    validation = remainder.sample(frac=0.5, random_state=42)
    test = remainder.drop(validation.index)

    candidates = []
    for l2 in [0.0, 0.005, 0.02, 0.08]:
        candidate = LogisticRiskModel(l2=l2).fit(train)
        validation_scores = candidate.predict_proba(validation, calibrated=False)
        candidates.append(
            {
                "l2": l2,
                "validation_auc": roc_auc(validation["defaulted"].to_numpy(), validation_scores),
                "validation_brier": brier_score(
                    validation["defaulted"].to_numpy(), validation_scores
                ),
                "model": candidate,
            }
        )
    selected = max(candidates, key=lambda row: (row["validation_auc"], -row["validation_brier"]))
    model = selected["model"]
    comparison = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "model"} for row in candidates]
    )
    comparison["selected"] = comparison["l2"] == selected["l2"]
    model.fit_calibrator(validation)
    scores = model.predict_proba(test)
    y = test["defaulted"].to_numpy()

    report = {
        "default_rate": float(y.mean()),
        "auc": roc_auc(y, scores),
        "gini": gini(y, scores),
        "ks": ks_statistic(y, scores),
        "capture_rate_top_10pct": capture_rate_at_k(y, scores, 0.10),
        "capture_rate_top_20pct": capture_rate_at_k(y, scores, 0.20),
        "brier_score": brier_score(y, scores),
        "expected_calibration_error": expected_calibration_error(y, scores),
        "selected_l2": model.l2,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    pd.Series(report).to_json(DATA_DIR / "model_report.json", indent=2)
    calibration_table(y, scores).to_csv(DATA_DIR / "calibration_table.csv", index=False)
    model.feature_importance().to_csv(DATA_DIR / "feature_importance.csv", index=False)
    comparison.to_csv(DATA_DIR / "model_comparison.csv", index=False)
    threshold_table(y, scores, test["loan_amount"].to_numpy()).to_csv(
        DATA_DIR / "threshold_analysis.csv", index=False
    )
    log_to_mlflow(report, model)

    print("Saved model:", MODEL_PATH)
    print(pd.Series(report).round(4).to_string())


if __name__ == "__main__":
    main()
