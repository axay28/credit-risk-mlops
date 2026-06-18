from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from bootstrap import ensure_artifacts

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


st.set_page_config(page_title="Credit Risk MLOps", layout="wide")
st.title("Credit Risk MLOps Dashboard")
with st.spinner("Preparing model artifacts..."):
    ensure_artifacts()


@st.cache_data
def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


report_path = DATA_DIR / "model_report.json"
calibration_path = DATA_DIR / "calibration_table.csv"
importance_path = DATA_DIR / "feature_importance.csv"
drift_path = DATA_DIR / "drift_report.csv"
loans_path = DATA_DIR / "loans_train.csv"

missing = [path.name for path in [report_path, calibration_path, importance_path, drift_path, loans_path] if not path.exists()]
if missing:
    st.warning("Run `python src/generate_data.py && python src/train.py && python src/drift.py` first.")
    st.write("Missing:", ", ".join(missing))
    st.stop()

report = read_json(report_path)
calibration = read_csv(calibration_path)
importance = read_csv(importance_path)
drift = read_csv(drift_path)
loans = read_csv(loans_path)

metric_cols = st.columns(5)
metric_cols[0].metric("AUC", f"{report['auc']:.3f}")
metric_cols[1].metric("Gini", f"{report['gini']:.3f}")
metric_cols[2].metric("KS", f"{report['ks']:.3f}")
metric_cols[3].metric("Capture @10%", f"{report['capture_rate_top_10pct']:.3f}")
metric_cols[4].metric("Default Rate", f"{report['default_rate']:.3f}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Calibration")
    st.line_chart(calibration[["predicted_pd", "observed_default_rate"]])
    st.caption("Predicted probability of default versus observed default rate by score bucket.")

with right:
    st.subheader("Feature Weights")
    plot_data = importance.assign(abs_weight=importance["coefficient"].abs()).sort_values("abs_weight", ascending=False).head(10)
    st.bar_chart(plot_data.set_index("feature")["coefficient"])
    st.caption("Top signed coefficients from the standardized logistic model.")

st.subheader("Population Stability Index")
st.dataframe(drift, use_container_width=True)
st.bar_chart(drift.set_index("feature")["psi"])

st.subheader("Training Data Snapshot")
st.dataframe(loans.head(100), use_container_width=True)
