import pandas as pd

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

input_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/false_negative_probability_analysis.csv"
)

output_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/compare_probability_groups.csv"
)

# --------------------------------------------------
# 2. Load false negative report
# --------------------------------------------------

false_negatives = pd.read_csv(input_path)

print("False negative report loaded successfully!")

# --------------------------------------------------
# 3. Create probability groups
# --------------------------------------------------

false_negatives["probability_group"] = "Confidently Wrong"

false_negatives.loc[
    false_negatives["fraud_probability"] >= 0.30,
    "probability_group"
] = "Near Threshold"

# --------------------------------------------------
# 4. Count each group
# --------------------------------------------------

group_counts = (
    false_negatives["probability_group"]
    .value_counts()
)

print("\nPROBABILITY GROUP COUNTS")
print("=" * 50)
print(group_counts)

# --------------------------------------------------
# 5. Calculate average features
# --------------------------------------------------

features = [
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

comparison = (
    false_negatives
    .groupby("probability_group")[features]
    .mean()
)

print("\nAVERAGE FEATURES BY PROBABILITY GROUP")
print("=" * 70)
print(comparison.round(3))

# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

false_negatives.to_csv(
    output_path,
    index=False
)

print("\nSaved successfully!")
print(output_path)