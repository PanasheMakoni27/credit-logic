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
    """Fit a preprocessing pipeline on the provided dataframe and save it.

    The pipeline imputes numeric features with median, scales them, and one-hot encodes
    categorical features (including newly created buckets). The fitted pipeline is saved
    as `output_path` using joblib.
    """
    # Choose some sensible numeric and categorical features (best-effort)
    numeric_feats = [
        c for c in [
            "AnnualIncome", "TotalAssets", "MonthlyDebtPayments", "MonthlyIncome", "CreditScore", "TotalDebtToIncomeRatio",
        ] if c in df.columns
    ]

    # Add engineered numeric columns if present
    for engineered in ["Log_AnnualIncome", "Log_TotalAssets", "CreditScore_DTI_interaction"]:
        if engineered in df.columns and engineered not in numeric_feats:
            numeric_feats.append(engineered)

    categorical_feats = [c for c in ["AgeBucket", "DTI_Bucket"] if c in df.columns]

    # Build transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats),
    ], remainder="drop")

    # Fit preprocessor
    X_sample = df[numeric_feats + categorical_feats].copy()
    preprocessor.fit(X_sample)

    # Ensure parent dir exists
    outdir = pathlib.Path(output_path).parent
    outdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, output_path)

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
