# credit-logic

Credit Logic — local end-to-end ML pipeline for loan scoring and model evaluation.

Purpose
- Build and evaluate a reproducible credit/loan approval model (feature engineering, preprocessing, baseline & CV training, calibration, inference + explainability).

Quickstart (Windows PowerShell)
```powershell
cd 'c:\Users\prese\Downloads\credit-logic'
python -m pip install -r requirements.txt
# If no requirements file exists, install common deps:
python -m pip install pandas numpy scikit-learn joblib matplotlib seaborn shap xgboost pytest

# Build features and preprocessor
python -u src/preprocess.py

# Train baseline
python -u src/train_baseline.py

# Cross-validated model selection
python -u src/train_cv.py

# Calibrate chosen model
python -u src/calibrate_and_register.py

# Score and explain
python -u src/infer.py --input-file data/processed/Loan_with_features.csv --limit 5 --explain-method auto

# Run tests
python -m pytest -q
```

Repository layout (high-level)
- `data/processed/` — cleaned and engineered datasets (`Loan_cleaned.csv`, `Loan_with_features.csv`)
- `src/` — source code (preprocessing, feature engineering, training, inference, explainability)
- `models/` — saved preprocessors, model artifacts, and `feature_map.json` / `registry.json`
- `reports/` — evaluation plots, EDA outputs, inference SHAP values
- `tests/` — pytest unit tests

Next steps & recommendations
- Run `src/calibrate_and_register.py` to produce a calibrated model and update `models/registry.json`.
- Add a GitHub Actions workflow to run tests and lint on PRs.
- Expand unit tests to cover edge cases and add a small monitoring script to detect drift post-deployment.

License & contribution
- Follow repository CONTRIBUTING and standard GitHub PR workflow. Open a PR from feature branches into `main` and run CI before merging.

Contact
- Repo owner: `PanasheMakoni27` (see the GitHub repo for issues / PRs).
