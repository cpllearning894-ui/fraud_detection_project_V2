import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix


# 1. File paths
X_test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "X_test_engineered.csv"
)

y_test_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/"
    "y_test_engineered.csv"
)

model_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/"
    "feature_engineered_random_forest_fraud_model.pkl"
)

output_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "cost_sensitive_threshold.csv"
)


# 2. Load test data
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()


# 3. Load trained model
model = joblib.load(model_path)


# 4. Get fraud probabilities
fraud_probabilities = model.predict_proba(X_test)[:, 1]


# 5. Define business costs
false_negative_cost = 100
false_positive_cost = 5


# 6. Define thresholds
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]


# 7. Store results
results = []


# 8. Test every threshold
for threshold in thresholds:

    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    total_cost = (
        (fn * false_negative_cost)
        +
        (fp * false_positive_cost)
    )

    results.append({
        "threshold": threshold,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "total_cost": total_cost
    })


# 9. Convert results to DataFrame
results_df = pd.DataFrame(results)


# 10. Display results
print("\nCOST-SENSITIVE THRESHOLD ANALYSIS")
print(results_df.to_string(index=False))


# 11. Find the cheapest threshold
best_result = results_df.loc[
    results_df["total_cost"].idxmin()
]

print("\nBEST THRESHOLD BY BUSINESS COST")
print(best_result)


# 12. Save results
results_df.to_csv(
    output_path,
    index=False
)

print("\nSaved:", output_path)