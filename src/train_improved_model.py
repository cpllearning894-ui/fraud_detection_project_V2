import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# --------------------------------
# Load training data
# --------------------------------

X_train = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/X_train.csv"
)

y_train = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/y_train.csv"
).squeeze()


print("Training data loaded successfully!")
print()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print()


# --------------------------------
# Create improved Random Forest
# --------------------------------

improved_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# --------------------------------
# Train improved model
# --------------------------------

improved_model.fit(
    X_train,
    y_train
)

print("Improved model training completed successfully!")
print()


# --------------------------------
# Save improved model
# --------------------------------

model_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/improved_random_forest_fraud_model.pkl"
)

joblib.dump(
    improved_model,
    model_path
)

print("Improved model saved successfully!")
print(model_path)