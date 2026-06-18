# Internal Project Documentation

This document is written for interview prep and personal understanding. It explains what this project does, why each part exists, and how to talk about it.

## One-Minute Explanation

I built a production-style credit risk MLOps project. It simulates loan applications, trains a probability-of-default model, evaluates it with credit-risk metrics like AUC, Gini, KS, and capture rate, checks calibration, monitors data drift with PSI, exposes a FastAPI scoring service, and presents the results in a deployed Streamlit dashboard.

The point was to show ML engineering skills that are not visible from my RAG/GenAI work: traditional ML modeling, risk evaluation, model monitoring, API deployment, and MLOps workflow design.

## Why I Built This

My resume already shows:

- RAG and healthcare AI through Radiant
- AWS/data pipelines through Atgeir
- dashboards/reporting through Jerseystem
- GenAI experience through research work

The missing portfolio signal was:

- traditional supervised ML
- fintech/credit modeling
- model calibration
- drift monitoring
- FastAPI model serving
- MLflow-style experiment tracking
- a deployed dashboard

This project fills that gap.

## End-to-End Workflow

The project follows this flow:

1. `src/generate_data.py` creates synthetic lending data.
2. `src/model.py` turns raw applicant fields into model features.
3. `src/train.py` trains the logistic regression model.
4. `src/metrics.py` calculates model performance metrics.
5. `src/drift.py` compares a current portfolio against the reference portfolio.
6. `app/main.py` exposes model scoring through FastAPI.
7. `app/dashboard.py` displays metrics and monitoring artifacts in Streamlit.
8. `app/bootstrap.py` makes deployment easier by generating missing artifacts automatically.

## Data Generation

The data is synthetic but designed to behave like loan data. Each row is a borrower/application.

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

The default label is generated from a realistic risk formula:

- higher debt-to-income increases default risk
- higher interest rate increases risk
- prior defaults increase risk
- delinquency count increases risk
- lower credit score increases risk
- longer employment history reduces risk
- higher income reduces risk
- some loan purposes are riskier than others

This gives the model a signal to learn while keeping the project self-contained and deployable without external datasets.

## Feature Engineering

`src/model.py` handles:

- numeric feature selection
- one-hot encoding for `loan_purpose`
- mean/std standardization
- storing feature statistics for inference

Standardization matters because the logistic model is trained with gradient descent. Features like income and loan amount have much larger numeric scales than delinquency counts or DTI, so scaling makes optimization stable.

## Model

The model is a from-scratch logistic regression implemented with NumPy.

Why logistic regression?

- It is common in credit risk modeling.
- It produces a probability of default.
- It is interpretable through coefficients.
- It is simple enough to explain clearly in interviews.
- It lets the project focus on MLOps and risk metrics rather than hiding everything inside a black-box library.

Training:

- initialize weights at zero
- calculate predicted probabilities using sigmoid
- compute prediction error
- update weights with gradient descent
- apply L2 regularization

Output:

- `models/credit_risk_model.json`

The model file contains:

- weights
- bias
- feature means
- feature standard deviations
- feature order

## Metrics

The project does not rely on plain accuracy because credit datasets can be imbalanced and ranking quality matters more.

### AUC

AUC measures whether risky borrowers are ranked above safer borrowers. AUC around 0.74 means the model has useful separation.

### Gini

Gini is widely used in credit risk and is computed as:

```text
Gini = 2 * AUC - 1
```

The current Gini is about 0.49.

### KS

KS measures the maximum separation between cumulative distributions of defaults and non-defaults. It is common in scorecard-style evaluation.

### Capture Rate

Capture rate answers: if we review the riskiest 10% or 20% of applications, what percentage of all defaults do we catch?

This is more business-friendly than accuracy.

### Calibration

Calibration compares predicted default probability against observed default rate by score bucket. It answers whether a predicted 20% risk group actually defaults around 20% of the time.

### PSI

PSI stands for Population Stability Index. It measures whether the current application population has drifted from the training/reference population.

Rough interpretation:

- below 0.10: stable
- 0.10 to 0.20: watch
- above 0.20: investigate

In this project, the generated current dataset intentionally shifts interest rate and DTI slightly so the drift report has something meaningful to show.

## FastAPI

`app/main.py` exposes:

- `GET /health`: checks service health
- `GET /metrics`: returns training metrics
- `GET /drift`: returns drift table
- `POST /score`: scores one applicant
- `POST /score-batch`: scores multiple applicants

The API loads the saved model and uses the same feature engineering statistics from training, which is important because inference must transform inputs exactly like training.

## Streamlit Dashboard

`app/dashboard.py` is the public-facing demo.

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

This is what recruiters or hiring managers can open quickly to understand the project.

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

If MLflow is missing, training still runs. This makes the project robust in lightweight environments like Streamlit Cloud.

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

This is useful because generated artifacts are not committed to GitHub, but Streamlit Cloud still needs them at runtime.

## Files Not Committed

The project intentionally does not commit generated files such as:

- `data/*.csv`
- `data/*.json`
- `models/*.json`
- `mlruns/`

These are reproducible artifacts generated from source code.

## How To Run Locally

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

## What To Say In Interviews

Short answer:

> I built a credit-risk MLOps project that predicts probability of default from loan application features. I implemented the model in NumPy, evaluated it with credit-risk metrics like AUC, Gini, KS, capture rate, and calibration, added PSI drift monitoring, exposed scoring through FastAPI, logged experiments with MLflow, and deployed a Streamlit dashboard.

Longer answer:

> I wanted to build something different from my RAG work, so I chose a fintech ML problem. The project simulates a lending portfolio, trains a logistic probability-of-default model, and focuses on production concerns: reproducible training, model artifacts, probability calibration, drift monitoring, and API serving. I also added a Streamlit dashboard so non-technical stakeholders can inspect metrics and drift without reading code.

## Strong Talking Points

- I implemented logistic regression from scratch to show fundamentals.
- I used credit-risk metrics rather than generic accuracy.
- I separated train-time artifacts from inference-time logic.
- I added PSI to show monitoring after deployment.
- I exposed both single and batch scoring endpoints.
- I made the Streamlit app deployable by generating artifacts automatically.
- I kept the project self-contained with synthetic data so anyone can run it.

## Limitations

Be honest if asked:

- The dataset is synthetic, not a real lending dataset.
- The model is logistic regression, not CatBoost/XGBoost.
- There is no real database or production scheduler.
- The monitoring report is batch-based, not real-time.
- Thresholds are illustrative and not regulated credit policy.

These limitations are acceptable because the project is built to demonstrate engineering workflow, not to be a production lending decision system.

## Best Next Improvements

High-value future upgrades:

1. Add XGBoost or CatBoost comparison.
2. Add SHAP explainability.
3. Add a model registry promotion step in MLflow.
4. Add GitHub Actions CI.
5. Add Docker Compose for API + dashboard.
6. Add a Postgres-backed scoring history table.
7. Add threshold optimization by expected profit/loss.
8. Add fairness checks across synthetic demographic segments.

## Resume Version

Use this resume bullet:

> Built an end-to-end credit default modeling and MLOps system with synthetic lending data, feature engineering, from-scratch NumPy logistic regression, calibration analysis, AUC/Gini/KS/capture-rate reporting, PSI drift monitoring, MLflow experiment tracking, FastAPI scoring endpoints, and a deployed Streamlit dashboard.

