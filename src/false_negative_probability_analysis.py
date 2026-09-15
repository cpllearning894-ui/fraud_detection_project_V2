import pandas as pd
import joblib

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

X_test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/X_test_engineered.csv"
)

y_test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/y_test_engineered.csv"
)

predictions_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/feature_engineered_predictions.csv"
)

model_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/feature_engineered_random_forest_fraud_model.pkl"
)

output_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/false_negative_probability_analysis.csv"
)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

predictions_df = pd.read_csv(predictions_path)

model = joblib.load(model_path)

print("Data and model loaded successfully!")

# --------------------------------------------------
# 3. Calculate fraud probabilities
# --------------------------------------------------

fraud_probabilities = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 4. Combine everything
# --------------------------------------------------

analysis_df = X_test.copy()

analysis_df["actual"] = y_test.values
analysis_df["predicted"] = predictions_df["predicted"].values
analysis_df["fraud_probability"] = fraud_probabilities

# --------------------------------------------------
# 5. Select false negatives
# --------------------------------------------------

false_negatives = analysis_df[
    (analysis_df["actual"] == 1) &
    (analysis_df["predicted"] == 0)
].copy()

# --------------------------------------------------
# 6. Sort by fraud probability
# --------------------------------------------------

false_negatives = false_negatives.sort_values(
    by="fraud_probability",
    ascending=False
)

# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\nFALSE NEGATIVE PROBABILITY ANALYSIS")
print("=" * 70)

print(false_negatives[
    [
        "transaction_amount",
        "transaction_hour",
        "account_age_days",
        "previous_transactions",
        "failed_attempts",
        "failed_attempt_rate",
        "is_new_device",
        "is_international",
        "fraud_probability"
    ]
].to_string(index=False))

# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

false_negatives.to_csv(
    output_path,
    index=False
)

print("\nSaved successfully!")
print(output_path)