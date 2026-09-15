import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


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


print("Feature-engineered testing data loaded successfully!")
print()

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print()


# --------------------------------
# Load feature-engineered model
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


print("Feature-engineered predictions created successfully!")
print()

print("First 10 predictions:")
print(y_pred[:10])
print()

print("First 10 actual values:")
print(y_test[:10].values)
print()


# --------------------------------
# Calculate metrics
# --------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# --------------------------------
# Print results
# --------------------------------

print("FEATURE-ENGINEERED MODEL EVALUATION RESULTS")
print("-" * 50)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print()

print("Classification Report:")
print()

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------
# Confusion Matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print()

print(cm)

print()

print(
    "Feature-engineered model evaluation completed!"
)