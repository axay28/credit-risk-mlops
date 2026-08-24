from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "credit_risk_model.json"

sys.path.append(str(SRC_DIR))


def ensure_artifacts() -> None:
    required = [
        DATA_DIR / "loans_train.csv",
        DATA_DIR / "loans_current.csv",
        DATA_DIR / "model_report.json",
        DATA_DIR / "calibration_table.csv",
        DATA_DIR / "feature_importance.csv",
        DATA_DIR / "model_comparison.csv",
        DATA_DIR / "threshold_analysis.csv",
        DATA_DIR / "drift_report.csv",
        MODEL_PATH,
    ]
    if all(path.exists() for path in required):
        return

    from generate_data import main as generate_data
    from train import main as train_model
    from drift import main as build_drift_report

    generate_data()
    train_model()
    build_drift_report()
