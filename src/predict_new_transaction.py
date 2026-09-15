import pandas as pd
import joblib
from pathlib import Path


# 1. Define project paths
project_path = Path(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2"
)

model_path = (
    project_path
    / "models"
    / "final_fraud_model.pkl"
)

config_path = (
    project_path
    / "reports"
    / "final_model_config.csv"
)


# 2. Load the final model
model = joblib.load(model_path)


# 3. Load the final configuration
config = pd.read_csv(config_path)

final_threshold = config.loc[
    0,
    "threshold"
]


# 4. Create a new transaction
new_transaction = pd.DataFrame({
    "transaction_amount": [500],
    "transaction_hour": [2],
    "account_age_days": [30],
    "previous_transactions": [5],
    "is_international": [1],
    "is_new_device": [1],
    "failed_attempts": [6]
})


# 5. Get fraud probability
fraud_probability = model.predict_proba(
    new_transaction
)[:, 1][0]


# 6. Make the final prediction
prediction = int(
    fraud_probability >= final_threshold
)


# 7. Display the result
print("\nNEW TRANSACTION PREDICTION")
print("Fraud probability:", round(fraud_probability, 4))
print("Threshold:", final_threshold)


if prediction == 1:
    print("Prediction: FRAUD 🚨")
else:
    print("Prediction: NORMAL ✅")