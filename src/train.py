from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from metrics import calibration_table, capture_rate_at_k, gini, ks_statistic, roc_auc
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
    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)

    model = LogisticRiskModel()
    model.fit(train)
    scores = model.predict_proba(test)
    y = test["defaulted"].to_numpy()

    report = {
        "default_rate": float(y.mean()),
        "auc": roc_auc(y, scores),
        "gini": gini(y, scores),
        "ks": ks_statistic(y, scores),
        "capture_rate_top_10pct": capture_rate_at_k(y, scores, 0.10),
        "capture_rate_top_20pct": capture_rate_at_k(y, scores, 0.20),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    pd.Series(report).to_json(DATA_DIR / "model_report.json", indent=2)
    calibration_table(y, scores).to_csv(DATA_DIR / "calibration_table.csv", index=False)
    model.feature_importance().to_csv(DATA_DIR / "feature_importance.csv", index=False)
    log_to_mlflow(report, model)

    print("Saved model:", MODEL_PATH)
    print(pd.Series(report).round(4).to_string())


if __name__ == "__main__":
    main()
