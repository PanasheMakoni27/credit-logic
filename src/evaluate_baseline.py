import os
import pathlib
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)


def evaluate_baseline(
    data_path: str = "data/processed/Loan_with_features.csv",
    preprocessor_path: str = "models/preprocessor.joblib",
    model_path: str = "models/baseline_model.joblib",
    outdir: str = "reports/baseline",
):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"Loading preprocessor from {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    target = "LoanApproved"
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {data_path}")

    # Determine features used by preprocessor
    try:
        features = list(preprocessor.feature_names_in_)
    except Exception:
        features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    print("Transforming features with preprocessor...")
    X_trans = preprocessor.transform(X)

    print("Predicting with baseline model...")
    probs = model.predict_proba(X_trans)[:, 1]
    preds = model.predict(X_trans)

    # ROC
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - Baseline")
    plt.legend(loc="lower right")
    roc_path = os.path.join(outdir, "baseline_roc.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"AP = {ap:.4f}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Baseline")
    plt.legend(loc="lower left")
    pr_path = os.path.join(outdir, "baseline_pr.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    # Classification report
    clf_report = classification_report(y, preds)
    report_txt = os.path.join(outdir, "baseline_classification.txt")
    with open(report_txt, "w") as fh:
        fh.write(clf_report)

    # Mistakes CSV: add predictions, probs, and error type
    results = df.copy()
    results["predicted"] = preds
    results["prob_approved"] = probs
    results["correct"] = results["predicted"] == results[target]

    def error_type(row):
        if row["predicted"] == 1 and row[target] == 0:
            return "FP"
        if row["predicted"] == 0 and row[target] == 1:
            return "FN"
        if row["predicted"] == 1 and row[target] == 1:
            return "TP"
        return "TN"

    results["error_type"] = results.apply(error_type, axis=1)
    mistakes_path = os.path.join(outdir, "baseline_mistakes.csv")
    results.to_csv(mistakes_path, index=False)

    summary = {
        "roc_path": roc_path,
        "pr_path": pr_path,
        "classification_report": report_txt,
        "mistakes_csv": mistakes_path,
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
    }

    summary_path = os.path.join(outdir, "baseline_eval_summary.json")
    import json

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Evaluation artifacts saved:")
    print(f" - ROC curve: {roc_path}")
    print(f" - PR curve: {pr_path}")
    print(f" - Classification report: {report_txt}")
    print(f" - Mistakes CSV: {mistakes_path}")
    print(f" - Summary JSON: {summary_path}")

    return summary


if __name__ == "__main__":
    evaluate_baseline()
