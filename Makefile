PYTHON ?= python

.PHONY: data train drift score test

data:
	$(PYTHON) src/generate_data.py

train: data
	$(PYTHON) src/train.py

drift:
	$(PYTHON) src/drift.py

score:
	$(PYTHON) src/score.py '{"annual_income": 900000, "loan_amount": 350000, "interest_rate": 13.5, "credit_score": 690, "dti": 0.32, "employment_years": 4, "delinquencies_2y": 1, "open_credit_lines": 6, "prior_defaults": 0, "loan_purpose": "debt_consolidation"}'

api:
	uvicorn app.main:app --reload

dashboard:
	streamlit run app/dashboard.py

mlflow:
	mlflow ui --backend-store-uri ./mlruns

test:
	pytest -q
