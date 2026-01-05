from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / 'Loan.csv'
CLEAN = ROOT / 'data' / 'processed' / 'Loan_cleaned.csv'

def summarize():
    if not ORIG.exists() or not CLEAN.exists():
        print(f"Missing files. Orig: {ORIG.exists()}, Clean: {CLEAN.exists()}")
        return

    orig = pd.read_csv(ORIG)
    clean = pd.read_csv(CLEAN)

    print("=== Basic shape ===")
    print(f"Original: rows={len(orig):,}, cols={len(orig.columns)}")
    print(f"Cleaned : rows={len(clean):,}, cols={len(clean.columns)}")

    print('\n=== Column differences ===')
    orig_cols = set(orig.columns)
    clean_cols = set(clean.columns)
    added = sorted(list(clean_cols - orig_cols))
    removed = sorted(list(orig_cols - clean_cols))
    print(f"Added columns ({len(added)}): {added}")
    print(f"Removed columns ({len(removed)}): {removed}")

    print('\n=== Missing values before/after (top 10 change) ===')
    miss_orig = orig.isnull().sum()
    miss_clean = clean.isnull().sum()
    miss_change = (miss_orig - miss_clean).abs().sort_values(ascending=False)
    for col in miss_change.head(10).index:
        print(f"{col}: orig_missing={miss_orig[col]}, clean_missing={miss_clean.get(col,0)}")

    total_missing_before = miss_orig.sum()
    total_missing_after = miss_clean.sum()
    print(f"\nTotal missing values: before={total_missing_before}, after={total_missing_after}")

    print('\n=== Data type changes (sample) ===')
    for col in sorted(list(orig_cols & clean_cols))[:20]:
        o = orig[col].dtype
        c = clean[col].dtype
        if o != c:
            print(f"{col}: orig={o} -> clean={c}")

    # DTI capping
    if 'TotalDebtToIncomeRatio' in orig.columns and 'TotalDebtToIncomeRatio' in clean.columns:
        above_before = (orig['TotalDebtToIncomeRatio'] > 1.0).sum()
        above_after = (clean['TotalDebtToIncomeRatio'] > 1.0).sum()
        print(f"\nTotalDebtToIncomeRatio > 1: before={above_before}, after={above_after}")
        if above_before>0:
            changed = orig['TotalDebtToIncomeRatio'] > 1.0
            sample_idx = orig[changed].index[:5]
            print('\nSample of capped values (index, orig, clean)')
            for i in sample_idx:
                print(i, orig.at[i,'TotalDebtToIncomeRatio'], clean.at[i,'TotalDebtToIncomeRatio'])

    # Insolvency flag
    if 'Is_Insolvent' in clean.columns:
        print(f"\nIs_Insolvent count in cleaned: {clean['Is_Insolvent'].sum()} (of {len(clean)})")

    # Example: show columns encoded
    print('\n=== Example encoded categorical columns (first 5 rows) ===')
    cat_candidates = ['EmploymentStatus','EducationLevel','MaritalStatus','HomeOwnershipStatus','LoanPurpose']
    for c in cat_candidates:
        if c in orig.columns and c in clean.columns:
            print(f"\nColumn: {c}")
            print(" original sample:")
            print(orig[c].head(5).to_string(index=False))
            print(" cleaned sample:")
            print(clean[c].head(5).to_string(index=False))

    print('\n=== Top-level summary ===')
    print('- Added `Is_Insolvent` flag; capped `TotalDebtToIncomeRatio` at 1.0')
    print('- Filled missing numerics with median and filled categoricals with mode; converted categoricals to integer codes')

if __name__ == "__main__":
    summarize()
