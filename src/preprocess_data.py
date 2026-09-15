import pandas as pd


# --------------------------------
# Load raw dataset
# --------------------------------

df = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/raw/"
    "fraud_transactions.csv"
)

print("Raw dataset loaded successfully!")
print()


# --------------------------------
# Check missing values
# --------------------------------

print("Missing values:")
print(df.isnull().sum())
print()


# --------------------------------
# Remove duplicate rows
# --------------------------------

df = df.drop_duplicates()

print("Duplicates removed.")
print("Dataset shape after removing duplicates:")
print(df.shape)
print()


# --------------------------------
# Separate features and target
# --------------------------------

X = df.drop("is_fraud", axis=1)

y = df["is_fraud"]

print("Features:")
print(X.head())
print()

print("Target:")
print(y.head())
print()


# --------------------------------
# Save processed dataset
# --------------------------------

processed_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "fraud_transactions_processed.csv"
)

df.to_csv(
    processed_path,
    index=False
)

print("Processed dataset saved successfully!")
print(processed_path)
print()


# --------------------------------
# Final dataset information
# --------------------------------

print("Final dataset shape:")
print(df.shape)
print()

print("Preprocessing completed successfully!")