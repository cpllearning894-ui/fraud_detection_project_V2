import pandas as pd
import numpy as np


# --------------------------------
# Make random data reproducible
# --------------------------------

np.random.seed(42)


# --------------------------------
# Number of transactions
# --------------------------------

n_transactions = 5000


# --------------------------------
# Create transaction features
# --------------------------------

transaction_amount = np.random.exponential(
    scale=100,
    size=n_transactions
)

transaction_hour = np.random.randint(
    0,
    24,
    n_transactions
)

account_age_days = np.random.randint(
    1,
    3650,
    n_transactions
)

previous_transactions = np.random.randint(
    0,
    500,
    n_transactions
)

is_international = np.random.choice(
    [0, 1],
    size=n_transactions,
    p=[0.85, 0.15]
)

is_new_device = np.random.choice(
    [0, 1],
    size=n_transactions,
    p=[0.80, 0.20]
)

failed_attempts = np.random.randint(
    0,
    6,
    n_transactions
)


# --------------------------------
# Create fraud risk score
# --------------------------------

fraud_score = (
    0.002 * transaction_amount
    + 0.8 * is_international
    + 1.2 * is_new_device
    + 0.5 * failed_attempts
    - 0.0003 * account_age_days
    - 0.002 * previous_transactions
    + np.random.normal(
        0,
        1,
        n_transactions
    )
)


# --------------------------------
# Create fraud target
# --------------------------------

is_fraud = (
    fraud_score > 2.8
).astype(int)


# --------------------------------
# Create DataFrame
# --------------------------------

df = pd.DataFrame({
    "transaction_amount": transaction_amount,
    "transaction_hour": transaction_hour,
    "account_age_days": account_age_days,
    "previous_transactions": previous_transactions,
    "is_international": is_international,
    "is_new_device": is_new_device,
    "failed_attempts": failed_attempts,
    "is_fraud": is_fraud
})


# --------------------------------
# Save dataset
# --------------------------------

file_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/raw/"
    "fraud_transactions.csv"
)

df.to_csv(
    file_path,
    index=False
)


# --------------------------------
# Show results
# --------------------------------

print("Fraud dataset created successfully!")
print()

print("First 5 transactions:")
print(df.head())
print()

print("Dataset shape:")
print(df.shape)
print()

print("Fraud distribution:")
print(df["is_fraud"].value_counts())
print()

print("Fraud percentage:")
print(
    round(
        df["is_fraud"].mean() * 100,
        2
    ),
    "%"
)

print()

print("Dataset saved successfully!")
print(file_path)