from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / 'Loan.csv'

def main():
    if not CSV_PATH.exists():
        print(f"Error: data file not found at {CSV_PATH}")
        return

    print(f"Loading: {CSV_PATH}\n")
    df = pd.read_csv(CSV_PATH)

    # 1. Missing values per column
    print("--- Missing Values Per Column (count, % of rows) ---")
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(3)
    miss_df = pd.DataFrame({'missing_count': missing, 'missing_pct': pct}).sort_values('missing_pct', ascending=False)
    print(miss_df)

    # Flag columns with notable missingness
    threshold_pct = 1.0  # 1% threshold to flag for attention
    flagged = miss_df[miss_df['missing_pct'] > threshold_pct]
    if not flagged.empty:
        print(f"\nColumns with > {threshold_pct}% missing values: {list(flagged.index)}")
    else:
        print(f"\nNo columns exceed {threshold_pct}% missing values.")

    # 2. Data types
    print("\n--- Data Types ---")
    print(df.dtypes)

    # Attempt to parse ApplicationDate if present
    if 'ApplicationDate' in df.columns:
        print("\nChecking `ApplicationDate` parsing...")
        parsed = pd.to_datetime(df['ApplicationDate'], errors='coerce')
        nat_count = parsed.isna().sum()
        print(f"Parsed datetimes: {len(parsed) - nat_count}, Failed parses (NaT): {nat_count}")
        if nat_count > 0:
            print(parsed.head(10))
        else:
            df['ApplicationDate'] = parsed

    # 3. Coerce important numeric columns and report non-numeric values
    numeric_candidates = [
        'CreditScore','AnnualIncome','MonthlyDebtPayments','LoanAmount','MonthlyIncome',
        'Experience','Age','LoanDuration','NumberOfOpenCreditLines','NumberOfCreditInquiries',
        'PaymentHistory','LengthOfCreditHistory','SavingsAccountBalance','CheckingAccountBalance',
        'TotalAssets','TotalLiabilities','NetWorth'
    ]
    print("\n--- Numeric Coercion Check ---")
    for col in numeric_candidates:
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors='coerce')
            non_numeric = coerced.isna() & ~df[col].isna()
            cnt = non_numeric.sum()
            if cnt:
                print(f"Column `{col}` has {cnt} non-numeric values (shown as NaN after coercion).")
            # replace with coerced where safe
            df[col] = coerced.fillna(df[col])

    # 4. Summary statistics
    print("\n--- Statistical Summary (numeric columns) ---")
    print(df.describe(include=[np.number]).T[['count','mean','std','min','25%','50%','75%','max']])

    # 5. Logical consistency checks
    print("\n--- Logical Consistency Checks ---")
    checks = []

    if set(['Experience','Age']).issubset(df.columns):
        bad_exp = df[df['Experience'] > df['Age']]
        checks.append(('Experience > Age', len(bad_exp), bad_exp.head(5)))

    # MonthlyDebtPayments compared to income
    if 'MonthlyDebtPayments' in df.columns and 'AnnualIncome' in df.columns:
        # If MonthlyDebtPayments * 12 > AnnualIncome that's suspicious
        over_annual = df[df['MonthlyDebtPayments'] * 12 > df['AnnualIncome']]
        checks.append(('MonthlyDebtPayments*12 > AnnualIncome', len(over_annual), over_annual.head(5)))

    if 'MonthlyDebtPayments' in df.columns and 'MonthlyIncome' in df.columns:
        over_monthly = df[df['MonthlyDebtPayments'] > df['MonthlyIncome']]
        checks.append(('MonthlyDebtPayments > MonthlyIncome', len(over_monthly), over_monthly.head(5)))

    if set(['TotalAssets','SavingsAccountBalance','CheckingAccountBalance']).issubset(df.columns):
        assets_incons = df[df['TotalAssets'] < (df['SavingsAccountBalance'] + df['CheckingAccountBalance'])]
        checks.append(('TotalAssets < Savings+Checking', len(assets_incons), assets_incons.head(5)))

    # Print check summaries
    for name, count, sample in checks:
        print(f"{name}: {count} rows")
        if count > 0:
            print(sample)

    # 6. High-cardinality / unexpected categories for important categorical fields
    cat_candidates = ['EmploymentStatus','EducationLevel','MaritalStatus','HomeOwnershipStatus','LoanPurpose']
    print("\n--- Categorical Cardinality / Unusual Values ---")
    for c in cat_candidates:
        if c in df.columns:
            vals = df[c].value_counts(dropna=False)
            print(f"\n`{c}` (top 10):")
            print(vals.head(10))

    # 7. Quick advice based on missingness and logical checks
    print("\n--- Quick Recommendations ---")
    if not flagged.empty:
        print("- Some columns show >1% missingness. Investigate whether missingness is informative (e.g., 'not applicable') or systematic.")
    else:
        print("- Missingness appears minimal (>1% threshold not exceeded).")

    print("- For banking/risk data, prefer careful imputation (median for skewed numerics, mode for categoricals) over blanket deletion to avoid selection bias.")
    print("- For fields where missing means 'no previous record' (e.g., PreviousLoanDefaults), consider domain logic to map to 0 vs NA.")
    print("- Fix any numeric coercion issues before modeling; investigate rows flagged by logical consistency checks.")

if __name__ == '__main__':
    main()
