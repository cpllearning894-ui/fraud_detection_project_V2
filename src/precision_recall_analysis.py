import pandas as pd
import joblib

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

# ==================================================
# PROJECT PATHS
# ==================================================

project_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2"
)

X_test_path = (
    project_path
    + "/data/processed/X_test.csv"
)

y_test_path = (
    project_path
    + "/data/processed/y_test.csv"
)

model_path = (
    project_path
    + "/models/final_fraud_model.pkl"
)

output_path = (
    project_path
    + "/reports/precision_recall_analysis.csv"
)

# ==================================================
# LOAD DATA AND MODEL
# ==================================================

X_test = pd.read_csv(X_test_path)

y_test = pd.read_csv(
    y_test_path
).squeeze()

model = joblib.load(model_path)

# ==================================================
# FRAUD PROBABILITIES
# ==================================================

fraud_probabilities = model.predict_proba(
    X_test
)[:, 1]

# ==================================================
# PRECISION-RECALL CURVE
# ==================================================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    fraud_probabilities
)

# ==================================================
# PR-AUC
# ==================================================

pr_auc = average_precision_score(
    y_test,
    fraud_probabilities
)

print("\nPRECISION-RECALL ANALYSIS")
print("PR-AUC:", round(pr_auc, 4))

# ==================================================
# SAVE CURVE DATA
# ==================================================

pr_df = pd.DataFrame({
    "precision": precision[:-1],
    "recall": recall[:-1],
    "threshold": thresholds
})

pr_df.to_csv(
    output_path,
    index=False
)

print("Saved:", output_path)