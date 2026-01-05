import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def train_baseline():
    # 1. Load the data we prepared in the last step
    data_path = "data/processed/Loan_with_features.csv"
    preprocessor_path = "models/preprocessor.joblib"
    
    df = pd.read_csv(data_path)
    preprocessor = joblib.load(preprocessor_path)

    # 2. Separate the Clues (X) from the Answer (y)
    # We drop the 'Target' and the 'Cheat Sheets' (RiskScore, InterestRate)
    target = "LoanApproved"
    leaky_cols = ["RiskScore", "InterestRate", "BaseInterestRate", "MonthlyLoanPayment"]
    
    # We only keep the features the preprocessor expects
    try:
        features = list(preprocessor.feature_names_in_)
    except Exception:
        # Fallback: assume the preprocessor was fit on a DataFrame and we can infer numeric+cat
        features = [c for c in df.columns if c not in ([target] + leaky_cols)]

    X = df[features]
    y = df[target]

    # 3. The "Three-Box Trick" (Split into Study and Exam groups)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Prepare the ingredients using our factory
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 5. Build the "Simple Brain"
    # We use 'class_weight=balanced' because rejections are more common than approvals
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    
    print("Training the Simple Brain (Logistic Regression)...")
    model.fit(X_train_transformed, y_train)

    # 6. Take the Final Exam
    predictions = model.predict(X_test_transformed)
    probs = model.predict_proba(X_test_transformed)[:, 1]

    # 7. Print the Report Card
    print("\n--- Model Report Card ---")
    print(classification_report(y_test, predictions))
    print(f"Overall Score (ROC-AUC): {roc_auc_score(y_test, probs):.4f}")
    
    print("\n--- Confusion Matrix (The 'Mistake' Table) ---")
    print(confusion_matrix(y_test, predictions))

    # 8. Save the Brain
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/baseline_model.joblib")
    print("\nBaseline model saved to models/baseline_model.joblib")


if __name__ == "__main__":
    train_baseline()
