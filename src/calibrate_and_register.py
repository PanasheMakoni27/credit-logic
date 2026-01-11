"""Calibrate the selected model and register it.

This script:
- loads the engineered dataset and preprocessor
- selects the best model (by metadata) and fits a calibration wrapper using
  a held-out calibration split with `cv='prefit'`
- selects an operational threshold by maximizing F1 on a test split
- saves the calibrated model and updates `models/registry.json`

Run:
    python -u src/calibrate_and_register.py
"""
from pathlib import Path
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, brier_score_loss


ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "Loan_with_features.csv"
PREPROCESSOR = ROOT / "models" / "preprocessor.joblib"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "model_selection"


def choose_best_model(models_dir: Path):
    metas = list(models_dir.glob("*.json"))
    best = None
    best_auc = -1.0
    best_meta = {}
    for m in metas:
        try:
            with open(m, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            test_auc = float(data.get("test_roc_auc") or -1)
            if test_auc > best_auc:
                best_auc = test_auc
                best = data.get("model_file")
                best_meta = data
        except Exception:
            continue
    if best:
        return Path(best), best_meta
    # fallback
    joblibs = sorted(models_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return (joblibs[0], {}) if joblibs else (None, {})


def main():
    if not FEATURES_CSV.exists():
        print("Features CSV missing; run preprocessing first.")
        return 1
    if not PREPROCESSOR.exists():
        print("Preprocessor missing; run preprocessing first.")
        return 1

    df = pd.read_csv(FEATURES_CSV)
    if "LoanApproved" not in df.columns:
        raise ValueError("LoanApproved target not found in features CSV")

    X_df = df.drop(columns=["LoanApproved", "ApplicationDate"]) if "ApplicationDate" in df.columns else df.drop(columns=["LoanApproved"])
    y = df["LoanApproved"].astype(int)

    # split into calibration and test (use 60/20/20 logic: here use 60 train (unused), 20 cal, 20 test)
    # Since model is already trained, we only need calibration and test splits
    _, X_temp, _, y_temp = train_test_split(X_df, y, test_size=0.4, random_state=42, stratify=y)
    X_cal_df, X_test_df, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    preproc = joblib.load(PREPROCESSOR)
    # Determine feature inputs
    try:
        feature_names = list(preproc.feature_names_in_)
    except Exception:
        fmap = MODELS_DIR / "feature_map.json"
        if fmap.exists():
            with open(fmap, "r", encoding="utf-8") as fh:
                fm = json.load(fh)
                feature_names = fm.get("numeric_features", []) + list(fm.get("categorical_features", {}).keys())
        else:
            feature_names = [c for c in X_df.columns]

    # ensure columns
    for c in feature_names:
        if c not in X_cal_df.columns:
            X_cal_df[c] = np.nan
        if c not in X_test_df.columns:
            X_test_df[c] = np.nan

    X_cal = preproc.transform(X_cal_df[feature_names])
    X_test = preproc.transform(X_test_df[feature_names])

    model_path, meta = choose_best_model(MODELS_DIR)
    if model_path is None:
        print("No model found to calibrate")
        return 1

    print(f"Using base model: {model_path}")
    base = joblib.load(model_path)

    # Calibrate
    cal = CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv='prefit')
    cal.fit(X_cal, y_cal)

    # Evaluate on test
    if hasattr(cal, "predict_proba"):
        probs = cal.predict_proba(X_test)[:, 1]
    else:
        probs = cal.decision_function(X_test)
        probs = 1 / (1 + np.exp(-probs))

    roc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    # choose threshold maximizing F1
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_f1 = -1
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cal_model_path = MODELS_DIR / f"calibrated--{timestamp}.joblib"
    joblib.dump(cal, cal_model_path)

    # update registry
    registry_path = MODELS_DIR / "registry.json"
    entry = {
        "model_file": str(cal_model_path),
        "base_model_file": str(model_path),
        "timestamp": timestamp,
        "calibration": "sigmoid",
        "test_roc_auc": float(roc),
        "test_brier_score": float(brier),
        "selected_threshold": best_t,
        "selected_f1": float(best_f1),
    }
    registry = []
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as fh:
                registry = json.load(fh)
        except Exception:
            registry = []
    registry.insert(0, entry)
    with open(registry_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = REPORT_DIR / f"calibration_summary_{timestamp}.json"
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2)

    print(f"Calibrated model saved to {cal_model_path}")
    print(f"Selected threshold={best_t:.3f} (F1={best_f1:.3f}), ROC-AUC={roc:.4f}, Brier={brier:.4f}")
    print(f"Registry updated: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
