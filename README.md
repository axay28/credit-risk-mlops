# Credit Risk MLOps

Production-style credit default modeling project focused on skills that are not already central in Radiant:

- credit risk metrics: AUC, Gini, KS, capture rate, calibration
- feature engineering for lending data
- model serving patterns
- drift monitoring with PSI
- reproducible train/evaluate/score workflow

This project intentionally avoids being another RAG or chatbot demo. It is designed for data scientist / AI engineer roles in fintech, lending, and practical ML teams.

## Project Shape

```text
credit-risk-mlops/
  app/main.py              # Optional FastAPI scoring service
  app/dashboard.py         # Optional Streamlit dashboard
  data/                    # Generated datasets and reports
  mlruns/                  # MLflow tracking store after training
  src/
    generate_data.py       # Synthetic lending dataset generator
    model.py               # Numpy logistic regression implementation
    metrics.py             # AUC, Gini, KS, capture, calibration, PSI
    train.py               # Train/evaluate/save model
    score.py               # Score one borrower from CLI JSON
    drift.py               # Compare current batch against reference
```

## Quickstart

```bash
python src/generate_data.py
python src/train.py
python src/score.py '{"annual_income": 900000, "loan_amount": 350000, "interest_rate": 13.5, "credit_score": 690, "dti": 0.32, "employment_years": 4, "delinquencies_2y": 1, "open_credit_lines": 6, "prior_defaults": 0, "loan_purpose": "debt_consolidation"}'
python src/drift.py
```

Use the bundled Codex Python in this workspace if your system Python is bare:

```bash
/Users/akshaymulgund/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/generate_data.py
```

## Optional Apps

Install the extra app dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI scoring service:

```bash
uvicorn app.main:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /metrics`
- `GET /drift`
- `POST /score`
- `POST /score-batch`

Run the Streamlit monitoring dashboard:

```bash
streamlit run app/dashboard.py
```

Run the MLflow UI after training:

```bash
mlflow ui --backend-store-uri ./mlruns
```

`src/train.py` logs parameters, metrics, model artifact, calibration table, and feature importance into MLflow when `mlflow` is installed. If MLflow is not installed, training still works and prints a skip message.

## Streamlit Cloud Deploy

Push this folder as its own GitHub repository, then deploy on Streamlit Community Cloud with:

- Repository: `axay28/credit-risk-mlops`
- Branch: `main`
- Main file path: `app/dashboard.py`

The dashboard calls `app/bootstrap.py` on startup, so generated datasets, model artifacts, metrics, and drift reports are created automatically if they are missing in the cloud environment.

## Current Benchmark

On the generated holdout set:

| Metric | Value |
| --- | ---: |
| Default rate | 0.1721 |
| AUC | 0.7426 |
| Gini | 0.4852 |
| KS | 0.3895 |
| Capture rate @ top 10% risk | 0.2542 |
| Capture rate @ top 20% risk | 0.4213 |

Example scoring output:

```json
{
  "probability_of_default": 0.083,
  "decision": "approve"
}
```
