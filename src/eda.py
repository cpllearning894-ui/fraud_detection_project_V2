import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os


# --------------------------------
# Load fraud dataset
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
# Show first 5 rows
# --------------------------------

print("First 5 rows:")
print(df.head())
print()


# --------------------------------
# Dataset information
# --------------------------------

print("Dataset information:")
df.info()
print()


# --------------------------------
# Dataset shape
# --------------------------------

print("Dataset shape:")
print(df.shape)
print()


# --------------------------------
# Missing values
# --------------------------------

print("Missing values:")
print(df.isnull().sum())
print()


# --------------------------------
# Duplicate rows
# --------------------------------

print("Duplicate rows:")
print(df.duplicated().sum())
print()


# --------------------------------
# Statistical summary
# --------------------------------

print("Statistical summary:")
print(df.describe())
print()


# --------------------------------
# Fraud distribution
# --------------------------------

print("Fraud distribution:")
print(df["is_fraud"].value_counts())
print()


# --------------------------------
# Create readable labels
# --------------------------------

fraud_counts = df["is_fraud"].value_counts().sort_index()

fraud_counts.index = [
    "Normal Transaction",
    "Fraud Transaction"
]


# --------------------------------
# Create bar chart
# --------------------------------

print("Creating fraud distribution chart...")

plt.figure(figsize=(8, 5))

fraud_counts.plot(kind="bar")

plt.title("Fraud Transaction Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=0)

plt.tight_layout()


# --------------------------------
# Save chart
# --------------------------------

chart_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/figures/"
    "fraud_distribution.png"
)

plt.savefig(
    chart_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------
# Check chart
# --------------------------------

print("Chart saved successfully!")
print("Chart location:")
print(chart_path)

print(
    "Chart exists:",
    os.path.exists(chart_path)
)

print()
print("EDA completed successfully!")