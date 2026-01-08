"""Train a Logistic Regression baseline using the preprocessed features.

Saves model to `models/baseline_model.joblib` and evaluation plots to
`reports/baseline/`.

Run:
    python -u src/train_baseline.py
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "Loan_with_features.csv"
PREPROCESSOR = ROOT / "models" / "preprocessor.joblib"
MODEL_OUT = ROOT / "models" / "baseline_model.joblib"
REPORT_DIR = ROOT / "reports" / "baseline"


def load_features_and_target(path: Path):
    df = pd.read_csv(path)
    if "LoanApproved" not in df.columns:
        raise ValueError("Target column 'LoanApproved' not found in features CSV")
    y = df["LoanApproved"].astype(int)
    X = df.drop(columns=["LoanApproved", "ApplicationDate"]) if "ApplicationDate" in df.columns else df.drop(columns=["LoanApproved"])
    return X, y


def main():
    if not FEATURES_CSV.exists():
        print(f"Features CSV not found at {FEATURES_CSV}. Run preprocessing first.")
        return 1
    if not PREPROCESSOR.exists():
        print(f"Preprocessor not found at {PREPROCESSOR}. Run preprocessing first.")
        return 1

    X_df, y = load_features_and_target(FEATURES_CSV)

    # Load preprocessor and transform
    preproc = joblib.load(PREPROCESSOR)
    # Determine input feature names expected by preprocessor
    feature_names = None
    try:
        feature_names = list(preproc.feature_names_in_)
    except Exception:
        # fallback to selecting all columns present in X_df that are listed in models/feature_map.json
        fmap = ROOT / "models" / "feature_map.json"
        if fmap.exists():
            import json
            with open(fmap, "r", encoding="utf-8") as fh:
                fm = json.load(fh)
                feature_names = [n for n in fm.get("numeric_features", []) + list(fm.get("categorical_features", {}).keys()) if n in X_df.columns]
        else:
            # last resort: use all columns except the ones obviously non-features
            feature_names = [c for c in X_df.columns if c not in ["ApplicationDate"]]

    X_in = X_df[feature_names].copy()
    X = preproc.transform(X_in)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)
    clf.fit(X_train, y_train)

    # Predict probabilities
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    roc = roc_auc_score(y_test, y_prob)
    print(f"Baseline ROC-AUC: {roc:.4f}")

    # Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"Saved baseline model to {MODEL_OUT}")

    # Save ROC and PR plots
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Baseline ROC")
    plt.legend()
    plt.tight_layout()
    roc_path = REPORT_DIR / "baseline_roc.png"
    plt.savefig(roc_path)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_path = REPORT_DIR / "baseline_pr.png"
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    import json
    with open(REPORT_DIR / "baseline_classification_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"Saved evaluation artifacts to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
