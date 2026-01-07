import json
import pathlib
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_explanations(sample_size: int = 500):
    reports_dir = pathlib.Path("reports/baseline")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and preprocessor...")
    model = joblib.load("models/baseline_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

    # Load feature map if available (provides encoded feature names)
    fmap_path = pathlib.Path("models") / "feature_map.json"
    feature_names_encoded = None
    if fmap_path.exists():
        with open(fmap_path, "r", encoding="utf-8") as fh:
            fmap = json.load(fh)
        feature_names_encoded = fmap.get("feature_names")

    print("Loading data...")
    df = pd.read_csv("data/processed/Loan_with_features.csv")

    # Determine input features to preprocessor
    try:
        input_features = list(preprocessor.feature_names_in_)
    except Exception:
        # fallback to feature_map numeric + categorical keys
        input_features = []
        if fmap_path.exists():
            numeric = fmap.get("numeric_features", [])
            cats = list(fmap.get("categorical_features", {}).keys())
            input_features = numeric + cats

    if len(input_features) == 0:
        raise RuntimeError("Could not determine preprocessor input feature names")

    # Sample data for speed
    rng = np.random.RandomState(42)
    if sample_size is not None and sample_size < len(df):
        X = df[input_features].sample(n=sample_size, random_state=42)
    else:
        X = df[input_features]

    print(f"Transforming {len(X)} rows with preprocessor (this may take a moment)...")
    X_trans = preprocessor.transform(X)

    # Ensure we have encoded feature names for plotting
    if feature_names_encoded is None:
        # try to read from preprocessor if it exposes get_feature_names_out
        try:
            feature_names_encoded = list(preprocessor.get_feature_names_out())
        except Exception:
            # fallback to numeric+categorical raw names
            feature_names_encoded = input_features

    # Convert transformed array to DataFrame (columns = encoded names)
    try:
        X_trans_df = pd.DataFrame(X_trans, columns=feature_names_encoded)
    except Exception:
        # If shapes mismatch, create generic column names
        cols = [f"f_{i}" for i in range(X_trans.shape[1])]
        X_trans_df = pd.DataFrame(X_trans, columns=cols)
        feature_names_encoded = cols

    # Compute SHAP values
    try:
        import shap

        print("Computing SHAP values (LinearExplainer)...")
        # For linear models, LinearExplainer is appropriate
        explainer = shap.LinearExplainer(model, X_trans, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_trans)

        # shap_values may be a list for multi-output; ensure we have array
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        # Summary plot (beeswarm)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_trans_df, feature_names=feature_names_encoded, show=False)
        outpath = reports_dir / "shap_summary.png"
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary to {outpath}")
        return str(outpath)
    except Exception as e:
        print("SHAP explainability failed:", e)
        raise


if __name__ == "__main__":
    generate_explanations()
