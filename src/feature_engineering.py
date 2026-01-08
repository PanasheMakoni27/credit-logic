import os
import sys
import pathlib
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to dataframe (in-place copy returned).

    Features added:
    - `AgeBucket`: categorical age groups
    - `DTI_Bucket`: categorical TotalDebtToIncomeRatio bucket
    - `Log_AnnualIncome`, `Log_TotalAssets`
    - `CreditScore_DTI_interaction`
    - `High_MonthlyDebt_to_Income`: boolean flag
    """
    df = df.copy()

    # Age bucket
    if "Age" in df.columns:
        bins = [0, 25, 35, 45, 55, 65, 200]
        labels = ["<=25", "26-35", "36-45", "46-55", "56-65", ">=66"]
        df["AgeBucket"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)

    # DTI bucket
    if "TotalDebtToIncomeRatio" in df.columns:
        dti_bins = [0.0, 0.2, 0.35, 0.5, 0.75, 1.0]
        dti_labels = ["0-0.2", "0.2-0.35", "0.35-0.5", "0.5-0.75", "0.75-1.0"]
        df["DTI_Bucket"] = pd.cut(df["TotalDebtToIncomeRatio"].clip(0, 1.0), bins=dti_bins, labels=dti_labels, include_lowest=True)

    # Log transforms for heavy-tailed numeric fields
    for col in ["AnnualIncome", "TotalAssets"]:
        if col in df.columns:
            safe = df[col].fillna(0).astype(float)
            df[f"Log_{col}"] = np.log1p(safe.clip(lower=0))

    # Interaction
    if "CreditScore" in df.columns and "TotalDebtToIncomeRatio" in df.columns:
        df["CreditScore_DTI_interaction"] = df["CreditScore"].astype(float) * df["TotalDebtToIncomeRatio"].astype(float)

    # High monthly-debt-to-income indicator
    if "MonthlyDebtPayments" in df.columns and "MonthlyIncome" in df.columns:
        df["High_MonthlyDebt_to_Income"] = (df["MonthlyDebtPayments"] > (0.5 * df["MonthlyIncome"])).astype(int)

    return df


def build_and_save_preprocessor(df: pd.DataFrame, output_path: str):
    """Fit and save preprocessing pipeline and a feature map JSON.

    Returns the fitted preprocessor.
    """
    # Select numeric and categorical features
    numeric_feats = [
        c for c in [
            "AnnualIncome", "TotalAssets", "MonthlyDebtPayments", "MonthlyIncome", "CreditScore", "TotalDebtToIncomeRatio",
        ] if c in df.columns
    ]

    for engineered in [
        "Log_AnnualIncome",
        "Log_TotalAssets",
        "CreditScore_DTI_interaction",
        "Total_Liquid",
        "CreditUtilization",
    ]:
        if engineered in df.columns and engineered not in numeric_feats:
            numeric_feats.append(engineered)

    categorical_feats = [c for c in ["AgeBucket", "DTI_Bucket", "DTI_Risk"] if c in df.columns]

    # Build transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # Create a OneHotEncoder in a version-compatible way
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore")

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats),
    ], remainder="drop")

    # Fit preprocessor
    X_sample = df[numeric_feats + categorical_feats].copy()
    preprocessor.fit(X_sample)

    # Persist preprocessor
    outdir = pathlib.Path(output_path).parent
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)

    # Build a feature map for reproducibility and explainability
    feature_map = {
        "numeric_features": numeric_feats,
        "categorical_features": {},
        "feature_names": list(numeric_feats),
    }

    # Try to extract categories from fitted OneHotEncoder
    try:
        cat_pipeline = preprocessor.named_transformers_.get("cat")
        ohe_fitted = None
        if cat_pipeline is not None and hasattr(cat_pipeline, "named_steps"):
            ohe_fitted = cat_pipeline.named_steps.get("onehot")
        # fallback: inspect steps
        if ohe_fitted is None:
            for step in getattr(cat_pipeline, "steps", []):
                if isinstance(step[1], OneHotEncoder):
                    ohe_fitted = step[1]
                    break

        if ohe_fitted is not None and hasattr(ohe_fitted, "categories_"):
            for col, cats in zip(categorical_feats, ohe_fitted.categories_):
                cat_list = [str(x) for x in list(cats)]
                feature_map["categorical_features"][col] = cat_list
                for cat in cat_list:
                    feature_map["feature_names"].append(f"{col}__{cat}")
        else:
            for col in categorical_feats:
                feature_map["categorical_features"][col] = []
    except Exception:
        # Non-fatal; continue without categorical mapping
        pass

    # Save feature map JSON
    fmap_path = outdir.parent / "models" / "feature_map.json"
    try:
        fmap_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(fmap_path, "w", encoding="utf-8") as fh:
            json.dump(feature_map, fh, indent=2, ensure_ascii=False)
        print(f"Saved feature map to {fmap_path}")
    except Exception:
        print("Warning: could not save feature_map.json")

    print(f"Saved preprocessor to {output_path}")
    return preprocessor


def main(cleaned_csv_path: str = "data/processed/Loan_cleaned.csv"):
    if not os.path.exists(cleaned_csv_path):
        print(f"Cleaned dataset not found at {cleaned_csv_path}. Run the cleaning step first.")
        return 1

    print(f"Loading cleaned dataset from {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path)

    print("Building engineered features...")
    df_fe = build_feature_engineering(df)

    # Save a copy of engineered full table
    features_path = "data/processed/Loan_with_features.csv"
    pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_fe.to_csv(features_path, index=False)
    print(f"Saved dataset with engineered features to {features_path} (rows={len(df_fe)})")

    # Fit and save preprocessor
    preprocessor_path = "models/preprocessor.joblib"
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    preprocessor = build_and_save_preprocessor(df_fe, preprocessor_path)

    # Optionally transform a small sample and show shape
    try:
        feature_names = list(preprocessor.feature_names_in_)
        X = df_fe[feature_names].head(5)
        Xt = preprocessor.transform(X)
        print(f"Example transformed shape for 5 rows: {Xt.shape}")
    except Exception:
        # feature_names_in_ may not exist for older sklearn versions
        pass

    print("Feature engineering pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
