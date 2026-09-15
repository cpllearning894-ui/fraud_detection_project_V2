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

print("Model training completed successfully!")
print()


# --------------------------------
# Show feature importance
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
# Save trained model
# --------------------------------

model_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/"
    "random_forest_fraud_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print("Model saved successfully!")
print(model_path)
print()

print("Machine Learning training completed!")