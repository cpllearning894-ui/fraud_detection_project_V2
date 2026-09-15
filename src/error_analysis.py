import pandas as pd
import joblib


# --------------------------------
# Load feature-engineered test data
# --------------------------------

X_test = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "X_test_engineered.csv"
)

y_test = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "y_test_engineered.csv"
).squeeze()


print("Feature-engineered test data loaded successfully!")
print()


# --------------------------------
# Load trained model
# --------------------------------

model = joblib.load(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/"
    "feature_engineered_random_forest_fraud_model.pkl"
)


print("Feature-engineered model loaded successfully!")
print()


# --------------------------------
# Create predictions
# --------------------------------

y_pred = model.predict(X_test)


print("Predictions created successfully!")
print()


# --------------------------------
# Create error-analysis DataFrame
# --------------------------------

analysis_df = X_test.copy()

analysis_df["actual"] = y_test.values

analysis_df["predicted"] = y_pred

# --------------------------------
# Save all predictions
# --------------------------------

analysis_df.to_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "feature_engineered_predictions.csv",
    index=False
)

print("All predictions saved successfully!")
print()


# --------------------------------
# Find False Negatives
# --------------------------------

false_negatives = analysis_df[
    (analysis_df["actual"] == 1)
    &
    (analysis_df["predicted"] == 0)
]


print("FALSE NEGATIVES")
print("-" * 50)

print(
    "Number of missed fraud transactions:",
    len(false_negatives)
)

print()

print(
    false_negatives
)


# --------------------------------
# Find False Positives
# --------------------------------

false_positives = analysis_df[
    (analysis_df["actual"] == 0)
    &
    (analysis_df["predicted"] == 1)
]


print()
print("FALSE POSITIVES")
print("-" * 50)

print(
    "Number of incorrectly flagged transactions:",
    len(false_positives)
)

print()

print(
    false_positives
)


# --------------------------------
# Save errors
# --------------------------------

false_negatives.to_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "false_negatives.csv",
    index=False
)

false_positives.to_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "false_positives.csv",
    index=False
)


print()
print("Error analysis completed!")
print()

print(
    "False negatives saved to:"
)

print(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "false_negatives.csv"
)

print()

print(
    "False positives saved to:"
)

print(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "false_positives.csv"
)