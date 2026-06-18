# Credit Risk MLOps

Live app: https://credit-risk-mlops-dvum3nelkcgipt4aea5laa.streamlit.app/

GitHub repo: https://github.com/axay28/credit-risk-mlops

An end-to-end credit default modeling project that shows practical machine learning engineering for fintech and lending use cases. The system generates a realistic synthetic lending portfolio, trains a credit risk model, evaluates it with industry-style risk metrics, monitors population drift, exposes a scoring API, and ships a Streamlit dashboard.

This project was built to demonstrate skills beyond RAG and GenAI chatbots: traditional ML modeling discipline, model evaluation, calibration, drift monitoring, deployment patterns, and MLOps tooling.

## What I Built

The project simulates a lending workflow:

1. Generate borrower and loan application data.
2. Train a probability-of-default model.
3. Evaluate the model using credit-risk metrics.
4. Produce calibration, feature importance, and drift reports.
5. Serve applicant scores through FastAPI.
6. Display model performance and monitoring views in Streamlit.
7. Log training metrics and artifacts to MLflow when available.

## Why This Project Matters

Most AI portfolios show chat demos. This project shows the parts of ML work that companies still need in production:

- choosing useful model metrics, not just accuracy
- understanding default risk ranking with AUC, Gini, KS, and capture rate
- checking whether predicted probabilities are calibrated
- monitoring feature and score drift over time
- exposing model predictions through an API
- documenting a reproducible train/evaluate/serve workflow

It is especially relevant for roles such as:

- Data Scientist
- AI/ML Engineer
- MLOps Engineer
- Fintech ML Engineer
- Credit Risk Data Scientist

## Tech Stack

- Python
- NumPy
- pandas
- FastAPI
- Streamlit
- MLflow
- Docker
- Uvicorn

The core model is a from-scratch NumPy logistic regression model. I intentionally implemented this instead of hiding everything behind a library so the project demonstrates understanding of feature scaling, optimization, coefficients, probability scoring, and evaluation.

## Project Structure

```text
credit-risk-mlops/
  app/
    bootstrap.py          # Generates model/data artifacts automatically for deployment
    dashboard.py          # Streamlit model monitoring dashboard
    main.py               # FastAPI scoring and monitoring API
  src/
    generate_data.py      # Synthetic lending data generator
    model.py              # NumPy logistic regression model and feature engineering
    metrics.py            # AUC, Gini, KS, capture rate, calibration, PSI
    train.py              # Train, evaluate, save model, optionally log to MLflow
    score.py              # CLI scoring for one borrower
    drift.py              # PSI drift monitoring for current vs reference data
  Dockerfile
  Makefile
  requirements.txt
```

## Model Features

The synthetic loan application data includes:

- annual income
- loan amount
- interest rate
- credit score
- debt-to-income ratio
- employment years
- delinquencies in the last 2 years
- open credit lines
- prior defaults
- loan purpose

The model predicts `probability_of_default` and assigns a decision band:

- `approve` for lower-risk applications
- `review` for medium-risk applications
- `decline` for higher-risk applications

## Current Benchmark

On the generated holdout set:

| Metric | Value |
| --- | ---: |
| Default rate | 0.1721 |
| AUC | 0.7426 |
| Gini | 0.4852 |
| KS | 0.3895 |
| Capture rate at top 10% risk | 0.2542 |
| Capture rate at top 20% risk | 0.4213 |

What these mean:

- **AUC** measures how well the model ranks risky borrowers above safer borrowers.
- **Gini** is a common credit-risk transformation of AUC: `2 * AUC - 1`.
- **KS** measures separation between defaulted and non-defaulted borrowers.
- **Capture rate** measures how many defaults are captured in the highest-risk population slice.
- **Calibration** compares predicted probability of default against observed default rate.
- **PSI** measures whether the current scoring population has drifted from the training population.

## Streamlit Dashboard

The deployed dashboard shows:

- headline model metrics
- calibration chart
- signed feature weights
- population stability index drift table
- PSI chart
- sample training data

Run locally:

```bash
streamlit run app/dashboard.py
```

Open:

```text
http://localhost:8501
```

The dashboard uses `app/bootstrap.py`, so if generated data/model artifacts are missing, they are created automatically at startup. This makes the app deploy cleanly on Streamlit Community Cloud.

## FastAPI Service

Run the API:

```bash
uvicorn app.main:app --reload
```

Open interactive docs:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

- `GET /health`
- `GET /metrics`
- `GET /drift`
- `POST /score`
- `POST /score-batch`

Example `/score` request:

```json
{
  "annual_income": 900000,
  "loan_amount": 350000,
  "interest_rate": 13.5,
  "credit_score": 690,
  "dti": 0.32,
  "employment_years": 4,
  "delinquencies_2y": 1,
  "open_credit_lines": 6,
  "prior_defaults": 0,
  "loan_purpose": "debt_consolidation"
}
```

Example response:

```json
{
  "probability_of_default": 0.083,
  "decision": "approve"
}
```

## MLflow Tracking

`src/train.py` logs the following to MLflow when `mlflow` is installed:

- model parameters
- AUC, Gini, KS, capture rate, default rate
- saved model artifact
- calibration table
- feature importance report

Run the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

If MLflow is not installed, training still runs and prints a skip message.

## Local Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate data, train, score, and check drift:

```bash
python src/generate_data.py
python src/train.py
python src/score.py '{"annual_income": 900000, "loan_amount": 350000, "interest_rate": 13.5, "credit_score": 690, "dti": 0.32, "employment_years": 4, "delinquencies_2y": 1, "open_credit_lines": 6, "prior_defaults": 0, "loan_purpose": "debt_consolidation"}'
python src/drift.py
```

Or use Make:

```bash
make train
make score
make drift
make dashboard
make api
make mlflow
```

## Streamlit Cloud Deployment

Deployment settings:

- Repository: `axay28/credit-risk-mlops`
- Branch: `main`
- Main file path: `app/dashboard.py`

The app is already deployed here:

https://credit-risk-mlops-dvum3nelkcgipt4aea5laa.streamlit.app/

## Portfolio Resume Bullet

Built an end-to-end credit default modeling and MLOps system with synthetic lending data, feature engineering, from-scratch NumPy logistic regression, calibration analysis, AUC/Gini/KS/capture-rate reporting, PSI drift monitoring, MLflow experiment tracking, FastAPI scoring endpoints, and a deployed Streamlit dashboard.
