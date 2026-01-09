"""Cross-validated model selection and stronger baseline training.

This script performs randomized hyperparameter search for a Logistic
Regression baseline and for a tree-based classifier (XGBoost if available,
otherwise sklearn's GradientBoostingClassifier). It uses a stratified
train/test split, runs RandomizedSearchCV on the training fold, evaluates on
the hold-out test fold, and saves best models and metadata to `models/`.

Run:
    python -u src/train_cv.py
"""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "Loan_with_features.csv"
PREPROCESSOR = ROOT / "models" / "preprocessor.joblib"
OUT_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "model_selection"


def load_data():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Features CSV not found at {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    if "LoanApproved" not in df.columns:
        raise ValueError("Target column 'LoanApproved' not found in features CSV")
    y = df["LoanApproved"].astype(int)
    X = df.drop(columns=["LoanApproved", "ApplicationDate"]) if "ApplicationDate" in df.columns else df.drop(columns=["LoanApproved"])
    return X, y


def get_feature_matrix(preproc_path, X_df):
    preproc = joblib.load(preproc_path)
    # choose feature names
    try:
        feature_names = list(preproc.feature_names_in_)
    except Exception:
        fmap = ROOT / "models" / "feature_map.json"
        if fmap.exists():
            with open(fmap, "r", encoding="utf-8") as fh:
                fm = json.load(fh)
                feature_names = [n for n in fm.get("numeric_features", []) + list(fm.get("categorical_features", {}).keys()) if n in X_df.columns]
        else:
            feature_names = [c for c in X_df.columns]
    X_in = X_df[feature_names].copy()
    X = preproc.transform(X_in)
    return X, feature_names


def build_logistic_search():
    clf = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42, max_iter=1000)
    param_dist = {
        "C": np.logspace(-4, 4, 50),
        "penalty": ["l1", "l2"],
    }
    return clf, param_dist


def build_tree_search():
    # Prefer XGBoost if available
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        param_dist = {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
    except Exception:
        clf = GradientBoostingClassifier(random_state=42)
        param_dist = {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
        }
    return clf, param_dist


def run_search(name, estimator, param_dist, X_train, y_train, n_iter=24):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=n_iter, scoring="roc_auc", cv=cv, verbose=1, n_jobs=-1, random_state=42)
    t0 = time.time()
    search.fit(X_train, y_train)
    t1 = time.time()
    print(f"{name} search done in {t1-t0:.1f}s. Best ROC-AUC (cv): {search.best_score_:.4f}")
    return search


def evaluate_and_save(name, search, X_test, y_test, feature_names):
    best = search.best_estimator_
    y_prob = best.predict_proba(X_test)[:, 1]
    y_pred = best.predict(X_test)
    roc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_name = f"{name}--{timestamp}.joblib"
    meta_name = f"{name}--{timestamp}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best, OUT_DIR / model_name)

    metadata = {
        "model_file": str(OUT_DIR / model_name),
        "name": name,
        "timestamp": timestamp,
        "best_params": search.best_params_,
        "cv_score": float(search.best_score_),
        "test_roc_auc": float(roc),
        "test_pr_auc": float(pr_auc),
    }
    with open(OUT_DIR / meta_name, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    # Save simple text report
    with open(REPORT_DIR / f"{name}--{timestamp}.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Name: {name}\n")
        fh.write(f"Best CV ROC-AUC: {search.best_score_:.4f}\n")
        fh.write(f"Test ROC-AUC: {roc:.4f}\n")
        fh.write(f"Test PR AUC: {pr_auc:.4f}\n")
        fh.write("\nClassification report:\n")
        fh.write(json.dumps(classification_report(y_test, y_pred, output_dict=True), indent=2))

    print(f"Saved {name} model to {OUT_DIR / model_name} and metadata to {OUT_DIR / meta_name}")
    return metadata


def main():
    X_df, y = load_data()

    # hold-out split
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)

    X_train, feature_names = get_feature_matrix(PREPROCESSOR, X_train_df)
    X_test, _ = get_feature_matrix(PREPROCESSOR, X_test_df)

    # Logistic
    log_clf, log_params = build_logistic_search()
    log_search = run_search("logistic", log_clf, log_params, X_train, y_train)
    log_meta = evaluate_and_save("logistic", log_search, X_test, y_test, feature_names)

    # Tree-based
    tree_clf, tree_params = build_tree_search()
    tree_search = run_search("tree", tree_clf, tree_params, X_train, y_train)
    tree_meta = evaluate_and_save("tree", tree_search, X_test, y_test, feature_names)

    # Save comparison summary
    summary = {"logistic": log_meta, "tree": tree_meta}
    with open(REPORT_DIR / "comparison_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Model selection complete. Summary written to reports/model_selection/comparison_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
