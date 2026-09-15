import pandas as pd
from sklearn.model_selection import train_test_split


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

print("Dataset shape:", df.shape)
print()


# --------------------------------
# Separate features and target
# --------------------------------

X = df.drop(
    "is_fraud",
    axis=1
)

y = df["is_fraud"]


# --------------------------------
# Split data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Data split successfully!")
print()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print()


# --------------------------------
# Calculate threshold from TRAINING
# data only
# --------------------------------

amount_threshold = X_train[
    "transaction_amount"
].quantile(0.75)


print("High amount threshold:")
print(amount_threshold)
print()


# --------------------------------
# Feature engineering function
# --------------------------------

def create_features(data):

    data = data.copy()

    # 1. Failed attempt rate
    data["failed_attempt_rate"] = (
        data["failed_attempts"]
        / (data["previous_transactions"] + 1)
    )

    # 2. Night transaction
    data["is_night"] = (
        (data["transaction_hour"] < 6)
        |
        (data["transaction_hour"] >= 22)
    ).astype(int)

    # 3. High transaction amount
    data["is_high_amount"] = (
        data["transaction_amount"]
        >= amount_threshold
    ).astype(int)

    # 4. Transactions per day
    data["transactions_per_day"] = (
        data["previous_transactions"]
        / (data["account_age_days"] + 1)
    )

    return data


# --------------------------------
# Apply feature engineering
# --------------------------------

X_train = create_features(X_train)

X_test = create_features(X_test)


print("Feature engineering completed!")
print()

print("X_train new shape:")
print(X_train.shape)

print()

print("X_test new shape:")
print(X_test.shape)

print()


# --------------------------------
# Save feature-engineered data
# --------------------------------

X_train_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "X_train_engineered.csv"
)

X_test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "X_test_engineered.csv"
)

y_train_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "y_train_engineered.csv"
)

y_test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "y_test_engineered.csv"
)


X_train.to_csv(
    X_train_path,
    index=False
)

X_test.to_csv(
    X_test_path,
    index=False
)

y_train.to_csv(
    y_train_path,
    index=False
)

y_test.to_csv(
    y_test_path,
    index=False
)


print("Feature-engineered train/test data saved successfully!")
print()

print(X_train_path)
print(X_test_path)
print(y_train_path)
print(y_test_path)