import pandas as pd


# 1. File paths
input_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "model_comparison.csv"
)

output_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "reports/"
    "final_model_selection.csv"
)


# 2. Load model comparison results
comparison_df = pd.read_csv(input_path)


# 3. Display all models
print("\nMODEL COMPARISON")
print(comparison_df.to_string(index=False))


# 4. Find the model with the highest F1 score
best_model = comparison_df.loc[
    comparison_df["F1 Score"].idxmax()
]


# 5. Display the selected model
print("\nFINAL MODEL SELECTED")
print(best_model)


# 6. Save the selected model information
best_model.to_frame().T.to_csv(
    output_path,
    index=False
)


# 7. Display confirmation
print("\nSaved:", output_path)