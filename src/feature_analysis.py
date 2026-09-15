import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# --------------------------------
# Load dataset
# --------------------------------

df = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/raw/"
    "fraud_transactions.csv"
)

print("Fraud dataset loaded successfully!")
print()


# --------------------------------
# Compare average feature values
# --------------------------------

print("Average feature values by fraud status:")
print()

feature_means = df.groupby("is_fraud")[
    [
        "transaction_amount",
        "account_age_days",
        "previous_transactions",
        "failed_attempts"
    ]
].mean()

print(feature_means)
print()


# --------------------------------
# Create readable fraud labels
# --------------------------------

df["fraud_status"] = df["is_fraud"].map({
    0: "Normal",
    1: "Fraud"
})


# --------------------------------
# Transaction amount vs fraud
# --------------------------------

amount_means = df.groupby("fraud_status")[
    "transaction_amount"
].mean()

plt.figure(figsize=(8, 5))

amount_means.plot(kind="bar")

plt.title("Average Transaction Amount by Fraud Status")
plt.xlabel("Transaction Status")
plt.ylabel("Average Transaction Amount")
plt.xticks(rotation=0)

plt.tight_layout()

amount_chart = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/figures/"
    "transaction_amount_vs_fraud.png"
)

plt.savefig(
    amount_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved:", amount_chart)


# --------------------------------
# Failed attempts vs fraud
# --------------------------------

failed_means = df.groupby("fraud_status")[
    "failed_attempts"
].mean()

plt.figure(figsize=(8, 5))

failed_means.plot(kind="bar")

plt.title("Average Failed Attempts by Fraud Status")
plt.xlabel("Transaction Status")
plt.ylabel("Average Failed Attempts")
plt.xticks(rotation=0)

plt.tight_layout()

failed_chart = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/figures/"
    "failed_attempts_vs_fraud.png"
)

plt.savefig(
    failed_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved:", failed_chart)


# --------------------------------
# Account age vs fraud
# --------------------------------

account_age_means = df.groupby("fraud_status")[
    "account_age_days"
].mean()

plt.figure(figsize=(8, 5))

account_age_means.plot(kind="bar")

plt.title("Average Account Age by Fraud Status")
plt.xlabel("Transaction Status")
plt.ylabel("Average Account Age (Days)")
plt.xticks(rotation=0)

plt.tight_layout()

account_age_chart = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/figures/"
    "account_age_vs_fraud.png"
)

plt.savefig(
    account_age_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved:", account_age_chart)


# --------------------------------
# Previous transactions vs fraud
# --------------------------------

previous_means = df.groupby("fraud_status")[
    "previous_transactions"
].mean()

plt.figure(figsize=(8, 5))

previous_means.plot(kind="bar")

plt.title("Average Previous Transactions by Fraud Status")
plt.xlabel("Transaction Status")
plt.ylabel("Average Previous Transactions")
plt.xticks(rotation=0)

plt.tight_layout()

previous_chart = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/figures/"
    "previous_transactions_vs_fraud.png"
)

plt.savefig(
    previous_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved:", previous_chart)


# --------------------------------
# Check generated files
# --------------------------------

print()
print("Checking generated reports:")

for file_path in [
    amount_chart,
    failed_chart,
    account_age_chart,
    previous_chart
]:
    print(
        os.path.basename(file_path),
        "exists:",
        os.path.exists(file_path)
    )


print()
print("Feature analysis completed successfully!")