"""Inference utility: score applicants with the chosen model and preprocessor.

Usage examples:
  # score first row of the engineered features CSV
  python -u src/infer.py --input-file data/processed/Loan_with_features.csv --limit 1

  # score a single applicant provided as JSON
  python -u src/infer.py --row '{"Age":45, "AnnualIncome":40000, ...}'

Outputs probabilities and decisions and optionally SHAP explanations saved
to `reports/inference/`.
"""
from pathlib import Path
import argparse
import json
import sys
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR = ROOT / "models" / "preprocessor.joblib"
MODELS_DIR = ROOT / "models"
FEATURES_CSV = ROOT / "data" / "processed" / "Loan_with_features.csv"
REPORT_DIR = ROOT / "reports" / "inference"


def choose_best_model(models_dir: Path):
    # scan for metadata json files created by train_cv.py
    metas = list(models_dir.glob("*.json"))
    best = None
    best_auc = -1.0
    for m in metas:
        try:
            with open(m, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            test_auc = float(data.get("test_roc_auc") or data.get("test_auc") or -1)
            if test_auc > best_auc:
                best_auc = test_auc
                best = data.get("model_file") or None
        except Exception:
            continue
    if best:
        return Path(best)
    # fallback: pick the latest joblib in models
    joblibs = sorted(models_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return joblibs[0] if joblibs else None


def get_feature_input_names(preproc):
    try:
        return list(preproc.feature_names_in_)
    except Exception:
        fmap = ROOT / "models" / "feature_map.json"
        if fmap.exists():
            with open(fmap, "r", encoding="utf-8") as fh:
                fm = json.load(fh)
                numeric = fm.get("numeric_features", [])
                cat = list(fm.get("categorical_features", {}).keys())
                return numeric + cat
        return None


def load_input(args):
    if args.row:
        obj = json.loads(args.row)
        df = pd.DataFrame([obj])
    else:
        path = Path(args.input_file) if args.input_file else FEATURES_CSV
        if not path.exists():
            print(f"Input file not found: {path}")
            sys.exit(1)
        df = pd.read_csv(path)
        if args.limit:
            df = df.head(args.limit)
    return df


def explain_with_shap(preproc, model, X_raw, feature_names, method="auto", out_png=None, out_csv=None):
    try:
        import shap
    except Exception:
        print("SHAP not installed; skipping explanations")
        return None

    # Build a pipeline that accepts raw DataFrame rows and returns probabilities
    try:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([("preprocessor", preproc), ("model", model)])
    except Exception:
        pipeline = None

    # Use provided method or let SHAP pick the best explainer
    try:
        if pipeline is not None:
            explainer = shap.Explainer(pipeline, X_raw, algorithm=method if method != "auto" else None)
            vals = explainer(X_raw)
            # shap returns arrays matching input features when given a pipeline with a DataFrame
            shap_values = vals.values
            # ensure shape matches feature_names
            if shap_values.shape[1] == len(feature_names):
                shap_df = pd.DataFrame(shap_values, columns=feature_names)
            else:
                # fallback: flatten per-feature contributions into numbered cols
                cols = [f"shap_{i}" for i in range(shap_values.shape[1])]
                shap_df = pd.DataFrame(shap_values, columns=cols)
        else:
            # As a fallback, explain on preprocessed arrays and map back if possible
            X_pre = preproc.transform(X_raw)
            explainer = shap.Explainer(model.predict_proba if hasattr(model, "predict_proba") else model, X_pre)
            vals = explainer(X_pre)
            shap_values = vals.values
            cols = [f"shap_{i}" for i in range(shap_values.shape[1])]
            shap_df = pd.DataFrame(shap_values, columns=cols)

        if out_csv:
            shap_df.to_csv(out_csv, index=False)
        return shap_df
    except Exception as e:
        print(f"SHAP explanation failed inside explain_with_shap: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", help="CSV of input rows to score")
    parser.add_argument("--row", help="Single JSON row to score (as string)")
    parser.add_argument("--model-file", help="Explicit model joblib to use")
    parser.add_argument("--limit", type=int, default=1, help="Limit rows when reading CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive class")
    parser.add_argument("--explain-method", default="auto", choices=["auto", "kernel", "linear", "tree", "none"], help="SHAP explainer method to use")
    args = parser.parse_args()

    df = load_input(args)
    if df.empty:
        print("No input rows provided")
        return 1

    if not PREPROCESSOR.exists():
        print("Preprocessor not found. Run preprocessing first.")
        return 1

    preproc = joblib.load(PREPROCESSOR)
    feature_in = get_feature_input_names(preproc)
    if feature_in is None:
        feature_in = [c for c in df.columns if c != "LoanApproved"]

    # ensure all required columns present, fill missing with NA
    missing = [c for c in feature_in if c not in df.columns]
    if missing:
        print(f"Warning: missing expected columns {missing}; filling with NaN")
        for c in missing:
            df[c] = np.nan

    X_raw = df[feature_in].copy()
    X = preproc.transform(X_raw)

    # Choose model
    model_path = Path(args.model_file) if args.model_file else choose_best_model(MODELS_DIR)
    if model_path is None or not model_path.exists():
        print("No model file found to load. Run training first.")
        return 1

    model = joblib.load(model_path)

    # Get probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.decision_function(X)
        probs = 1 / (1 + np.exp(-probs))

    decisions = (probs >= args.threshold).astype(int)

    out = df.copy()
    out["score"] = probs
    out["decision"] = decisions

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "scores.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved scores to {out_path}")

    # Try SHAP explanations for first row(s)
    try:
        if args.explain_method != "none":
            feat_names = feature_in
            shap_csv = REPORT_DIR / "shap_values.csv"
            shap_df = explain_with_shap(preproc, model, X_raw, feat_names, method=args.explain_method, out_csv=shap_csv)
            if shap_df is not None:
                print(f"Saved SHAP values to {shap_csv}")
    except Exception as e:
        print(f"SHAP explanation failed: {e}")

    # Print small summary
    for i, row in out.head(5).iterrows():
        print(f"row={i} score={row['score']:.4f} decision={row['decision']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
