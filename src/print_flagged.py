from pathlib import Path
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / 'Loan.csv'

def main():
    if not CSV_PATH.exists():
        print(f"Error: data file not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    sus = df[df['MonthlyDebtPayments'] > df['MonthlyIncome']]
    print(f"Flagged rows: {len(sus)}")
    if len(sus) > 0:
        print(sus.to_string(index=False))

if __name__ == '__main__':
    main()
