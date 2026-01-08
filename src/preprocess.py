"""Feature Factory / Preprocessing entrypoint.

This script loads the cleaned dataset, applies feature engineering (re-using
`feature_engineering.py`), then adds higher-level "Super Clues":

- `DTI_Risk`: coarse risk band ('Safe','Caution','High Risk')
- `Total_Liquid`: sum of `Savings` + `Checking` when available
- `CreditUtilization`: `CurrentBalance / TotalAvailableCredit` (safe-handled)

It fits and saves the preprocessor (`models/preprocessor.joblib`) by calling
the existing `build_and_save_preprocessor` and writes the final table to
`data/processed/Loan_with_features.csv`.

Run:
    python -u src/preprocess.py
"""
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEANED = ROOT / "data" / "processed" / "Loan_cleaned.csv"
OUT_FEATURES = ROOT / "data" / "processed" / "Loan_with_features.csv"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.joblib"


def add_super_clues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure DTI exists
    if "TotalDebtToIncomeRatio" in df.columns:
        dti = df["TotalDebtToIncomeRatio"].astype(float).fillna(0).clip(0, 1)
        # Coarse risk bands: Safe <0.20, Caution [0.20,0.35), High Risk >=0.35
        bins = [-0.01, 0.2, 0.35, 1.01]
        labels = ["Safe", "Caution", "High Risk"]
        df["DTI_Risk"] = pd.cut(dti, bins=bins, labels=labels, include_lowest=True)

    # Wealth indicator: sum of Savings and Checking if present
    savings_col = None
    checking_col = None
    # Look for common savings/checking column names
    for c in ["Savings", "Checking", "SavingsBalance", "CheckingBalance", "SavingsAccountBalance", "CheckingAccountBalance"]:
        if c in df.columns and savings_col is None and "Savings" in c:
            savings_col = c
        if c in df.columns and checking_col is None and "Checking" in c:
            checking_col = c

    if savings_col or checking_col:
        s = df[savings_col] if savings_col in df.columns else 0
        c = df[checking_col] if checking_col in df.columns else 0
        df["Total_Liquid"] = pd.to_numeric(s, errors="coerce").fillna(0) + pd.to_numeric(c, errors="coerce").fillna(0)

    # Credit Utilization: prefer an existing utilization column, else compute if possible
    if "CreditUtilization" in df.columns:
        df["CreditUtilization"] = pd.to_numeric(df["CreditUtilization"], errors="coerce").fillna(0)
    elif "CreditCardUtilizationRate" in df.columns:
        # assume this is already a 0-1 rate
        df["CreditUtilization"] = pd.to_numeric(df["CreditCardUtilizationRate"], errors="coerce").fillna(0)
    else:
        # fallback if fields for computation exist
        if "CurrentBalance" in df.columns and "TotalAvailableCredit" in df.columns:
            cur = pd.to_numeric(df["CurrentBalance"], errors="coerce").fillna(0)
            avail = pd.to_numeric(df["TotalAvailableCredit"], errors="coerce").replace(0, np.nan)
            df["CreditUtilization"] = (cur / avail).replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def main(cleaned_path: str = None):
    cleaned_path = Path(cleaned_path) if cleaned_path else DEFAULT_CLEANED
    if not cleaned_path.exists():
        print(f"Cleaned dataset not found at {cleaned_path}. Run cleaning step first.")
        return 1

    print(f"Loading cleaned dataset from {cleaned_path}")
    df = pd.read_csv(cleaned_path)

    # Re-use existing feature engineering
    try:
        from feature_engineering import build_feature_engineering, build_and_save_preprocessor
    except Exception:
        # If running as module from different working dir, try src import fallback
        sys.path.insert(0, str(Path(__file__).parent))
        from feature_engineering import build_feature_engineering, build_and_save_preprocessor

    print("Applying base engineered features...")
    df_fe = build_feature_engineering(df)

    print("Adding Master's-level Super Clues (DTI bands, Total_Liquid, CreditUtilization)...")
    df_fe = add_super_clues(df_fe)

    # Save engineered dataset
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df_fe.to_csv(OUT_FEATURES, index=False)
    print(f"Saved dataset with engineered features to {OUT_FEATURES} (rows={len(df_fe)})")

    # Fit and save preprocessor
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    preproc = build_and_save_preprocessor(df_fe, str(PREPROCESSOR_PATH))

    print("Preprocessing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
