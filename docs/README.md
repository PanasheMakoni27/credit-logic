# Documentation

This folder contains human-readable guidance and notes for the `credit-logic` repository.

Overview
- The project is organized as an end-to-end local pipeline. Key responsibilities:
  - `src/preprocess.py`: build features and save the preprocessor
  - `src/feature_engineering.py`: feature factory and preprocessor fit helper
  - `src/train_baseline.py`, `src/train_cv.py`: training and model selection
  - `src/calibrate_and_register.py`: calibrate model and update `models/registry.json`
  - `src/infer.py`: inference CLI with SHAP explanations
  - `src/eda.py`: exploratory data analysis scripts

How to use
- See the top-level `README.md` for quickstart commands and further instructions.

Notes
- Keep `reports/` and `models/` out of source control in long-running projects; they are useful here for traceability and reproducibility.
