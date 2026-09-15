import pandas as pd

# --------------------------------
# Create model comparison data
# --------------------------------

comparison_data = {
    "Model": [
        "Original Random Forest",
        "Improved Random Forest",
        "SMOTE Random Forest",
        "Tuned Random Forest",
        "Feature-Engineered Random Forest"
    ],

    "Accuracy": [
        0.9040,
        0.9040,
        0.8290,
        0.9180,
        0.9040
    ],

    "Precision": [
        0.4948,
        0.4948,
        0.2791,
        0.6667,
        0.4945
    ],

    "Recall": [
        0.5053,
        0.5053,
        0.5053,
        0.2737,
        0.4737
    ],

    "F1 Score": [
        0.5000,
        0.5000,
        0.3596,
        0.3881,
        0.4839
    ]
}


# --------------------------------
# Convert dictionary to DataFrame
# --------------------------------

comparison_df = pd.DataFrame(comparison_data)


# --------------------------------
# Display model comparison
# --------------------------------

print()
print("FRAUD DETECTION MODEL COMPARISON")
print("=" * 70)
print()

print(comparison_df.to_string(index=False))


# --------------------------------
# Find best model based on F1
# --------------------------------

best_model = comparison_df.loc[
    comparison_df["F1 Score"].idxmax()
]


print()
print("=" * 70)
print("BEST MODEL BASED ON F1 SCORE")
print("=" * 70)
print()

print("Model:", best_model["Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))


# --------------------------------
# Save CSV report
# --------------------------------

csv_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/model_comparison.csv"
)

comparison_df.to_csv(
    csv_path,
    index=False
)


# --------------------------------
# Create text report
# --------------------------------

text_report = f"""
FRAUD DETECTION MODEL COMPARISON REPORT
========================================

MODEL COMPARISON
----------------------------------------

{comparison_df.to_string(index=False)}


BEST MODEL
----------------------------------------

Best model based on F1 Score:

Model: {best_model["Model"]}
F1 Score: {best_model["F1 Score"]:.4f}


PROFESSIONAL INTERPRETATION
----------------------------------------

The Original Random Forest and Improved
Random Forest achieved the highest F1 Score
among the evaluated models.

The SMOTE Random Forest produced lower
overall performance and more false positive
predictions.

The Tuned Random Forest achieved the highest
Accuracy and Precision, but its Recall was
much lower.

The Feature-Engineered Random Forest did not
improve the baseline model's F1 Score.

For the current fraud detection objective,
the Original Random Forest provides the best
balance between Precision and Recall.

However, fraud detection places strong
importance on Recall because missing an
actual fraudulent transaction can be costly.

Therefore, further False Negative analysis
should be performed before selecting the
final production model.
"""


# --------------------------------
# Save text report
# --------------------------------

text_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/model_comparison_report.txt"
)

with open(
    text_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(text_report)


# --------------------------------
# Confirmation
# --------------------------------

print()
print("Reports saved successfully!")
print()

print("CSV Report:")
print(csv_path)

print()

print("Text Report:")
print(text_path)