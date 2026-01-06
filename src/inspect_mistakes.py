import pathlib
import pandas as pd


def detective_report():
    path = "reports/baseline/baseline_mistakes.csv"
    df = pd.read_csv(path)

    # False Positives: Model said "Yes" (1) but it was "No" (0)
    fps = df[df["error_type"] == "FP"].sort_values(by="prob_approved", ascending=False)

    # False Negatives: Model said "No" (0) but it was "Yes" (1)
    fns = df[df["error_type"] == "FN"].sort_values(by="prob_approved", ascending=True)

    print("=== THE DETECTIVE REPORT ===")
    print(f"\nTotal Mistakes Found: {len(df)}")
    print(f"Number of 'Dangerous' Mistakes (False Positives): {len(fps)}")
    print(f"Number of 'Missed Opportunity' Mistakes (False Negatives): {len(fns)}")

    print("\n--- TOP 3 DANGEROUS MISTAKES (The 'Liars') ---")
    print("These people looked so good the model was sure they'd be approved, but they weren't.")
    cols_to_show = [
        "AnnualIncome",
        "CreditScore",
        "TotalDebtToIncomeRatio",
        "prob_approved",
        "LoanApproved",
    ]
    if len(fps) == 0:
        print("No false positives found.")
    else:
        print(fps[cols_to_show].head(3).to_string(index=False))

    print("\n--- TOP 3 MISSED OPPORTUNITIES (The 'Hidden Gems') ---")
    print("The model thought these people were bad, but they actually got approved.")
    if len(fns) == 0:
        print("No false negatives found.")
    else:
        print(fns[cols_to_show].head(3).to_string(index=False))

    # Save the top examples for further inspection
    outdir = pathlib.Path("reports/baseline")
    outdir.mkdir(parents=True, exist_ok=True)
    fps.head(50).to_csv(outdir / "top_false_positives.csv", index=False)
    fns.head(50).to_csv(outdir / "top_false_negatives.csv", index=False)

    print(f"\nSaved top false positives to: {outdir / 'top_false_positives.csv'}")
    print(f"Saved top false negatives to: {outdir / 'top_false_negatives.csv'}")


if __name__ == "__main__":
    detective_report()
