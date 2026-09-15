import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------
# Load testing data
# --------------------------------

X_test = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/X_test.csv"
)

y_test = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/y_test.csv"
).squeeze()


print("Testing data loaded successfully!")
print()


# --------------------------------
# Load best current model
# --------------------------------

model = joblib.load(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/random_forest_fraud_model.pkl"
)

print("Best current model loaded successfully!")
print()


# --------------------------------
# Get fraud probabilities
# --------------------------------

fraud_probabilities = model.predict_proba(
    X_test
)[:, 1]


print("Fraud probabilities created successfully!")
print()

print("First 10 fraud probabilities:")
print(
    fraud_probabilities[:10]
)
print()


# --------------------------------
# Define thresholds
# --------------------------------

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]


# --------------------------------
# Test each threshold
# --------------------------------

results = []


for threshold in thresholds:

    y_pred = (
        fraud_probabilities >= threshold
    ).astype(int)


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


    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# --------------------------------
# Create results DataFrame
# --------------------------------

results_df = pd.DataFrame(
    results
)


print("THRESHOLD TUNING RESULTS")
print("=" * 60)
print()

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------
# Find best threshold
# --------------------------------

best_result = results_df.loc[
    results_df["F1 Score"].idxmax()
]


print()
print("=" * 60)
print("BEST THRESHOLD BASED ON F1 SCORE")
print("=" * 60)
print()

print(
    "Threshold:",
    best_result["Threshold"]
)

print(
    "Precision:",
    round(
        best_result["Precision"],
        4
    )
)

print(
    "Recall:",
    round(
        best_result["Recall"],
        4
    )
)

print(
    "F1 Score:",
    round(
        best_result["F1 Score"],
        4
    )
)


# --------------------------------
# Create predictions using best threshold
# --------------------------------

best_threshold = best_result["Threshold"]

best_predictions = (
    fraud_probabilities >= best_threshold
).astype(int)


# --------------------------------
# Create confusion matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    best_predictions
)


print()
print("CONFUSION MATRIX")
print()
print(cm)


# --------------------------------
# Save threshold results
# --------------------------------

results_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/threshold_tuning_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print()
print("Threshold tuning results saved successfully!")
print(results_path)