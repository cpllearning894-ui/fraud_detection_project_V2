import pandas as pd


# --------------------------------
# File paths
# --------------------------------

test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/X_test_engineered.csv"
)

target_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/y_test_engineered.csv"
)

prediction_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/feature_engineered_predictions.csv"
)


# --------------------------------
# Load test data
# --------------------------------

X_test = pd.read_csv(test_path)

y_test = pd.read_csv(target_path).squeeze()

predictions = pd.read_csv(prediction_path)


# --------------------------------
# Combine actual and predicted values
# --------------------------------

analysis_df = X_test.copy()

analysis_df["actual"] = y_test.values

analysis_df["predicted"] = predictions["predicted"].values


# --------------------------------
# Find False Negatives
# --------------------------------

false_negatives = analysis_df[
    (analysis_df["actual"] == 1) &
    (analysis_df["predicted"] == 0)
].copy()


# --------------------------------
# Display False Negatives
# --------------------------------

print()
print("FALSE NEGATIVE ANALYSIS")
print("=" * 70)
print()

print(
    "Number of False Negatives:",
    len(false_negatives)
)

print()

print(false_negatives.to_string(index=False))


# --------------------------------
# Calculate average values
# --------------------------------

features = [
    "transaction_amount",
    "transaction_hour",
    "account_age_days",
    "previous_transactions",
    "is_international",
    "is_new_device",
    "failed_attempts",
    "failed_attempt_rate",
    "is_night",
    "is_high_amount",
    "transactions_per_day"
]


fn_average = false_negatives[features].mean()


# --------------------------------
# Display averages
# --------------------------------

print()
print("=" * 70)
print("AVERAGE VALUES OF MISSED FRAUD TRANSACTIONS")
print("=" * 70)
print()

print(fn_average.round(3))


# --------------------------------
# Save False Negatives
# --------------------------------

output_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/false_negative_deep_analysis.csv"
)

false_negatives.to_csv(
    output_path,
    index=False
)


# --------------------------------
# Confirmation
# --------------------------------

print()
print("=" * 70)
print("FALSE NEGATIVE ANALYSIS SAVED SUCCESSFULLY!")
print("=" * 70)
print()

print(output_path)