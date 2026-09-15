import pandas as pd
import joblib

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
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
    + "/reports/roc_auc_analysis.csv"
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
# CALCULATE FRAUD PROBABILITIES
# ==================================================

fraud_probabilities = model.predict_proba(
    X_test
)[:, 1]

# ==================================================
# CALCULATE AUC
# ==================================================

auc_score = roc_auc_score(
    y_test,
    fraud_probabilities
)

# ==================================================
# CALCULATE ROC CURVE VALUES
# ==================================================

false_positive_rate, true_positive_rate, thresholds = roc_curve(
    y_test,
    fraud_probabilities
)

print("\nROC-AUC ANALYSIS")
print("AUC:", round(auc_score, 4))

# ==================================================
# SAVE ROC DATA
# ==================================================

roc_df = pd.DataFrame({
    "false_positive_rate": false_positive_rate,
    "true_positive_rate": true_positive_rate,
    "threshold": thresholds
})

roc_df.to_csv(
    output_path,
    index=False
)

print("Saved:", output_path)