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
# Load data
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
# Create result labels
# --------------------------------

analysis_df["result"] = "Other"

analysis_df.loc[
    (analysis_df["actual"] == 1) &
    (analysis_df["predicted"] == 1),
    "result"
] = "True Positive"

analysis_df.loc[
    (analysis_df["actual"] == 1) &
    (analysis_df["predicted"] == 0),
    "result"
] = "False Negative"


# --------------------------------
# Select only fraud transactions
# --------------------------------

fraud_analysis = analysis_df[
    analysis_df["actual"] == 1
].copy()


# --------------------------------
# Select features to compare
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


# --------------------------------
# Calculate average values
# --------------------------------

comparison = (
    fraud_analysis
    .groupby("result")[features]
    .mean()
)


# --------------------------------
# Display comparison
# --------------------------------

print()
print("FALSE NEGATIVES VS TRUE POSITIVES")
print("=" * 70)
print()

print(comparison.round(3))


# --------------------------------
# Calculate differences
# --------------------------------

difference = (
    comparison.loc["True Positive"]
    - comparison.loc["False Negative"] 
)


# --------------------------------
# Display differences
# --------------------------------

print()
print("=" * 70)
print("DIFFERENCE: TRUE POSITIVE - FALSE NEGATIVE")
print("=" * 70)
print()

print(difference.round(3))


# --------------------------------
# Save comparison report
# --------------------------------

output_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/false_negative_vs_true_positive.csv"
)

comparison.round(3).to_csv(output_path)


# --------------------------------
# Confirmation
# --------------------------------

print()
print("=" * 70)
print("COMPARISON SAVED SUCCESSFULLY!")
print("=" * 70)
print()

print(output_path)