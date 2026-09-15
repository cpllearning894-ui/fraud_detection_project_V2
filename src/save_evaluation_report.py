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


# --------------------------------
# Load trained model
# --------------------------------

model = joblib.load(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/random_forest_fraud_model.pkl"
)


# --------------------------------
# Create predictions
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# Calculate evaluation metrics
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

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

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# --------------------------------
# Create evaluation report
# --------------------------------

evaluation_report = f"""
FRAUD DETECTION MODEL EVALUATION REPORT
========================================

MODEL PERFORMANCE
----------------------------------------

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}


CLASSIFICATION REPORT
----------------------------------------

{report}


CONFUSION MATRIX
----------------------------------------

{cm}


MODEL INTERPRETATION
----------------------------------------

The model was evaluated using testing data.

Accuracy shows the overall percentage of correct predictions.

Precision shows how often predicted fraud cases
were actually fraud.

Recall shows how many real fraud cases
were successfully detected.

F1 Score provides a balance between
Precision and Recall.

"""


# --------------------------------
# Save report
# --------------------------------

report_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/model_evaluation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(evaluation_report)


print("Evaluation report saved successfully!")
print()
print(report_path)