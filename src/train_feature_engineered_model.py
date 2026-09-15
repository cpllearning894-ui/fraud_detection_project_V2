import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# --------------------------------
# Load feature-engineered training data
# --------------------------------

X_train = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "X_train_engineered.csv"
)

y_train = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "y_train_engineered.csv"
).squeeze()


print("Feature-engineered training data loaded successfully!")
print()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print()


# --------------------------------
# Create Random Forest model
# --------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# --------------------------------
# Train model
# --------------------------------

model.fit(
    X_train,
    y_train
)


print("Feature-engineered Random Forest training completed!")
print()


# --------------------------------
# Feature importance
# --------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("Feature importance:")
print(feature_importance)
print()


# --------------------------------
# Save model
# --------------------------------

model_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/"
    "feature_engineered_random_forest_fraud_model.pkl"
)

joblib.dump(
    model,
    model_path
)


print("Feature-engineered model saved successfully!")
print(model_path)