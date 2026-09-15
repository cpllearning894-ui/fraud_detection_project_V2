import pandas as pd


# --------------------------------
# Load feature-engineered test data
# --------------------------------

test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "X_test_engineered.csv"
)

target_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "y_test_engineered.csv"
)


X_test = pd.read_csv(test_path)

y_test = pd.read_csv(
    target_path
).squeeze()


print("Feature-engineered test data loaded successfully!")
print()


# --------------------------------
# Load predictions
# --------------------------------

prediction_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "feature_engineered_predictions.csv"
)

predictions = pd.read_csv(
    prediction_path
)


print("Predictions loaded successfully!")
print()


# --------------------------------
# Combine test data and predictions
# --------------------------------

analysis_df = X_test.copy()

analysis_df["actual"] = y_test.values

analysis_df["predicted"] = predictions["predicted"].values


# --------------------------------
# Create error groups
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


analysis_df.loc[
    (analysis_df["actual"] == 0) &
    (analysis_df["predicted"] == 1),
    "result"
] = "False Positive"


analysis_df.loc[
    (analysis_df["actual"] == 0) &
    (analysis_df["predicted"] == 0),
    "result"
] = "True Negative"


# --------------------------------
# Compare feature averages
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


comparison = (
    analysis_df
    .groupby("result")[features]
    .mean()
)


print("QUANTITATIVE ERROR ANALYSIS")
print("=" * 70)
print()

print(
    comparison.round(3)
)


# --------------------------------
# Save comparison
# --------------------------------

output_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "quantitative_error_analysis.csv"
)


comparison.to_csv(
    output_path
)


print()
print("Quantitative error analysis saved successfully!")
print(output_path)