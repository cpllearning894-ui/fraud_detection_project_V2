import pandas as pd
import joblib

from imblearn.over_sampling import SMOTE
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


# --------------------------------
# Show original class distribution
# --------------------------------

print("Original training fraud distribution:")
print(y_train.value_counts())
print()


# --------------------------------
# Apply SMOTE
# --------------------------------

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("SMOTE applied successfully!")
print()

print("Training fraud distribution after SMOTE:")
print(y_train_smote.value_counts())
print()


# --------------------------------
# Create Random Forest model
# --------------------------------

smote_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# --------------------------------
# Train model with SMOTE data
# --------------------------------

smote_model.fit(
    X_train_smote,
    y_train_smote
)


print("SMOTE model training completed successfully!")
print()


# --------------------------------
# Save model
# --------------------------------

model_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/smote_random_forest_fraud_model.pkl"
)

joblib.dump(
    smote_model,
    model_path
)


print("SMOTE model saved successfully!")
print(model_path)