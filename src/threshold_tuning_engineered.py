import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

X_test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/X_test_engineered.csv"
)

y_test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/y_test_engineered.csv"
)

model_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/feature_engineered_random_forest_fraud_model.pkl"
)

output_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/threshold_tuning_engineered.csv"
)

# --------------------------------------------------
# 2. Load data and model
# --------------------------------------------------

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

model = joblib.load(model_path)

print("Data and model loaded successfully!")

# --------------------------------------------------
# 3. Get fraud probabilities
# --------------------------------------------------

fraud_probabilities = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 4. Define thresholds
# --------------------------------------------------

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

# --------------------------------------------------
# 5. Store results
# --------------------------------------------------

results = []

# --------------------------------------------------
# 6. Test every threshold
# --------------------------------------------------

for threshold in thresholds:

    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    })

# --------------------------------------------------
# 7. Convert results into a DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)

# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\nTHRESHOLD TUNING RESULTS")
print("=" * 80)

print(
    results_df.round(4).to_string(index=False)
)

# --------------------------------------------------
# 9. Find the best threshold by F1
# --------------------------------------------------

best_result = results_df.loc[
    results_df["f1_score"].idxmax()
]

print("\nBEST THRESHOLD BY F1 SCORE")
print("=" * 80)
print(best_result)

# --------------------------------------------------
# 10. Save results
# --------------------------------------------------

results_df.to_csv(
    output_path,
    index=False
)

print("\nSaved successfully!")
print(output_path)