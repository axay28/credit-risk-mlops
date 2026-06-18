# Project Documentation

This document explains the project architecture, workflow, modeling choices, and operational components for the credit risk MLOps system.

## Overview

Credit Risk MLOps is a production-style machine learning project for probability-of-default modeling. It simulates loan applications, trains a credit risk model, evaluates it with credit-risk metrics, monitors data drift with PSI, exposes a FastAPI scoring service, and presents model performance in a Streamlit dashboard.

The project covers:

- traditional supervised machine learning
- fintech-oriented credit risk modeling
- model calibration and ranking metrics
- drift monitoring
- API-based model serving
- MLflow experiment tracking
- Streamlit deployment

## End-to-End Workflow

The project follows this flow:

1. `src/generate_data.py` creates synthetic lending data.
2. `src/model.py` transforms raw applicant fields into model features.
3. `src/train.py` trains the logistic regression model.
4. `src/metrics.py` calculates model performance metrics.
5. `src/drift.py` compares current loan application data against the reference training data.
6. `app/main.py` exposes model scoring through FastAPI.
7. `app/dashboard.py` displays metrics and monitoring artifacts in Streamlit.
8. `app/bootstrap.py` generates missing artifacts automatically for deployment.

## Data Generation

The data is synthetic but designed to behave like loan application data. Each row represents one borrower/application.

Fields:

- `annual_income`
- `loan_amount`
- `interest_rate`
- `credit_score`
- `dti`
- `employment_years`
- `delinquencies_2y`
- `open_credit_lines`
- `prior_defaults`
- `loan_purpose`
- `defaulted`

The default label is generated from a risk formula:

- higher debt-to-income increases default risk
- higher interest rate increases risk
- prior defaults increase risk
- delinquency count increases risk
- lower credit score increases risk
- longer employment history reduces risk
- higher income reduces risk
- some loan purposes are riskier than others

This gives the model a meaningful signal to learn while keeping the project self-contained and deployable without external datasets.

## Feature Engineering

`src/model.py` handles:

- numeric feature selection
- one-hot encoding for `loan_purpose`
- mean/std standardization
- storage of feature statistics for inference

Standardization matters because the logistic model is trained with gradient descent. Features like income and loan amount have much larger numeric scales than delinquency counts or DTI, so scaling keeps optimization stable.

## Model

The model is a from-scratch logistic regression implemented with NumPy.

Logistic regression is a good fit because:

- it is common in credit risk modeling
- it produces a probability of default
- it is interpretable through coefficients
- it keeps the modeling logic transparent
- it lets the project focus on MLOps and risk metrics

Training process:

1. Initialize weights.
2. Generate predicted probabilities with sigmoid.
3. Compute prediction error.
4. Update weights with gradient descent.
5. Apply L2 regularization.

Output:

- `models/credit_risk_model.json`

The model file contains:

- weights
- bias
- feature means
- feature standard deviations
- feature order

## Metrics

The project does not rely on plain accuracy because credit datasets can be imbalanced and ranking quality matters more than raw classification accuracy.

### AUC

AUC measures whether risky borrowers are ranked above safer borrowers. AUC around 0.74 indicates useful separation.

### Gini

Gini is widely used in credit risk and is computed as:

```text
Gini = 2 * AUC - 1
```

The current Gini is about 0.49.

### KS

KS measures the maximum separation between cumulative distributions of defaults and non-defaults. It is common in scorecard-style evaluation.

### Capture Rate

Capture rate measures how many defaults are captured by reviewing the highest-risk slice of the application population.

### Calibration

Calibration compares predicted default probability against observed default rate by score bucket. It checks whether predicted probabilities match actual default frequency.

### PSI

PSI stands for Population Stability Index. It measures whether the current application population has drifted from the training/reference population.

Rough interpretation:

- below 0.10: stable
- 0.10 to 0.20: watch
- above 0.20: investigate

In this project, the generated current dataset intentionally shifts interest rate and DTI slightly so the drift report has meaningful behavior to display.

## FastAPI Service

`app/main.py` exposes:

- `GET /health`: checks service health
- `GET /metrics`: returns training metrics
- `GET /drift`: returns drift table
- `POST /score`: scores one applicant
- `POST /score-batch`: scores multiple applicants

The API loads the saved model and uses the same feature engineering statistics from training. This keeps inference transformations consistent with training.

## Streamlit Dashboard

`app/dashboard.py` is the public-facing dashboard.

It shows:

- AUC
- Gini
- KS
- capture rate
- default rate
- calibration chart
- feature coefficient chart
- PSI drift table
- PSI bar chart
- training data preview

Live app:

https://credit-risk-mlops-dvum3nelkcgipt4aea5laa.streamlit.app/

## MLflow

`src/train.py` logs to MLflow when installed.

Logged items:

- model family
- optimizer
- default rate
- AUC
- Gini
- KS
- capture rates
- model artifact
- calibration CSV
- feature importance CSV

If MLflow is missing, training still runs. This keeps the project robust in lightweight environments like Streamlit Community Cloud.

## Bootstrap

`app/bootstrap.py` checks for required generated files:

- train dataset
- current dataset
- model report
- calibration table
- feature importance
- drift report
- saved model

If any are missing, it runs:

1. generate data
2. train model
3. build drift report

This is useful because generated artifacts are not committed to GitHub, while Streamlit Community Cloud still needs them at runtime.

## Files Not Committed

The project intentionally does not commit generated files such as:

- `data/*.csv`
- `data/*.json`
- `models/*.json`
- `mlruns/`

These are reproducible artifacts generated from source code.

## Local Development

```bash
cd "/Users/akshaymulgund/Documents/New project 2/credit-risk-mlops"
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
python src/drift.py
streamlit run app/dashboard.py
uvicorn app.main:app --reload
```

Open:

- Streamlit: `http://localhost:8501`
- FastAPI docs: `http://127.0.0.1:8000/docs`
- MLflow: `http://127.0.0.1:5000`

## Engineering Highlights

- From-scratch NumPy logistic regression keeps the model implementation transparent.
- Credit-risk metrics are used instead of generic accuracy.
- Train-time artifacts are separated from inference-time logic.
- PSI drift monitoring shows post-training population stability.
- FastAPI supports both single and batch scoring.
- Streamlit deployment works without committed generated artifacts because bootstrap creates them at startup.
- The project is self-contained with synthetic data, so it can run without third-party datasets or credentials.

## Known Constraints

- The dataset is synthetic, not a real lending dataset.
- The model is logistic regression, not CatBoost or XGBoost.
- There is no production database or scheduler.
- The monitoring report is batch-based, not real-time.
- Thresholds are illustrative and not regulated credit policy.

These constraints keep the project focused on the machine learning engineering workflow rather than production lending operations.

## Future Work

High-value future upgrades:

1. Add XGBoost or CatBoost comparison.
2. Add SHAP explainability.
3. Add a model registry promotion step in MLflow.
4. Add GitHub Actions CI.
5. Add Docker Compose for API and dashboard.
6. Add a Postgres-backed scoring history table.
7. Add threshold optimization by expected profit/loss.
8. Add fairness checks across synthetic demographic segments.
