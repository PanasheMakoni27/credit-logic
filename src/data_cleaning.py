from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / 'Loan.csv'
OUT_DIR = ROOT / 'data' / 'processed'
OUT_PATH = OUT_DIR / 'Loan_cleaned.csv'

def main():
    if not CSV_PATH.exists():
        print(f"Error: data file not found at {CSV_PATH}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # Parse dates
    if 'ApplicationDate' in df.columns:
        df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'], errors='coerce')

    # 1) Create Is_Insolvent flag BEFORE capping (preserve signal)
    # Consider someone insolvent if monthly debt > monthly income or DTI > 1
    if set(['MonthlyDebtPayments', 'MonthlyIncome', 'TotalDebtToIncomeRatio']).issubset(df.columns):
        insolvent_mask = (df['MonthlyDebtPayments'] > df['MonthlyIncome']) | (df['TotalDebtToIncomeRatio'] > 1.0)
        df['Is_Insolvent'] = insolvent_mask.astype(int)
        print(f"Is_Insolvent flagged: {df['Is_Insolvent'].sum()} rows")
    else:
        df['Is_Insolvent'] = 0
        print("Warning: Required columns for insolvency flag not present. Set Is_Insolvent=0 for all rows.")

    # 2) Cap TotalDebtToIncomeRatio at 1.0
    if 'TotalDebtToIncomeRatio' in df.columns:
        df['TotalDebtToIncomeRatio'] = df['TotalDebtToIncomeRatio'].clip(upper=1.0)

    # 3) Impute missing values (median for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print('\nNumeric columns to impute (median):')
    for col in numeric_cols:
        missing_before = df[col].isna().sum()
        if missing_before > 0:
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"- {col}: filled {missing_before} missing with median={median}")

    # 4) Impute categoricals with mode and encode to integers
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print('\nCategorical columns to impute/encode:')
    for col in cat_cols:
        missing_before = df[col].isna().sum()
        if missing_before > 0:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                fill = mode.iloc[0]
            else:
                fill = 'MISSING'
            df[col].fillna(fill, inplace=True)
            print(f"- {col}: filled {missing_before} missing with mode='{fill}'")
        # Convert to categorical codes (integers)
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

    # 5) Summary of remaining missing values
    total_missing = df.isnull().sum().sum()
    print(f"\nTotal missing values after imputation: {total_missing}")

    # 6) Save cleaned CSV
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned dataset to {OUT_PATH}")

if __name__ == '__main__':
    main()
