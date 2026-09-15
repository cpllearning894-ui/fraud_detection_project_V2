import pandas as pd
import numpy as np


# --------------------------------
# Load processed dataset
# --------------------------------

data_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "fraud_transactions_processed.csv"
)

df = pd.read_csv(data_path)

print("Processed dataset loaded successfully!")
print()

print("Original dataset shape:")
print(df.shape)
print()


# --------------------------------
# Create new features
# --------------------------------

# 1. Failed attempts rate
df["failed_attempt_rate"] = (
    df["failed_attempts"]
    / (df["previous_transactions"] + 1)
)


# 2. Night transaction
df["is_night"] = (
    (df["transaction_hour"] < 6)
    |
    (df["transaction_hour"] >= 22)
).astype(int)


# 3. High transaction amount
amount_threshold = df["transaction_amount"].quantile(0.75)

df["is_high_amount"] = (
    df["transaction_amount"] >= amount_threshold
).astype(int)


# 4. Transaction activity
df["transactions_per_day"] = (
    df["previous_transactions"]
    / (df["account_age_days"] + 1)
)


# --------------------------------
# Display new features
# --------------------------------

print("New features created:")
print()

print(
    df[
        [
            "failed_attempt_rate",
            "is_night",
            "is_high_amount",
            "transactions_per_day"
        ]
    ].head()
)

print()


# --------------------------------
# Save feature-engineered dataset
# --------------------------------

output_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "fraud_transactions_feature_engineered.csv"
)

df.to_csv(
    output_path,
    index=False
)


print("Feature-engineered dataset saved successfully!")
print(output_path)
print()

print("Final dataset shape:")
print(df.shape)