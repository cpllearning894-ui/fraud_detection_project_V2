import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# --------------------------------
# Load training data
# --------------------------------

X_train = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/X_train.csv"
)

y_train = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "data/processed/y_train.csv"
).squeeze()


print("Training data loaded successfully!")
print()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print()


# --------------------------------
# Create base Random Forest model
# --------------------------------

rf_model = RandomForestClassifier(
    random_state=42
)


# --------------------------------
# Define hyperparameters
# --------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# --------------------------------
# Create Grid Search
# --------------------------------

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=2
)


# --------------------------------
# Start hyperparameter tuning
# --------------------------------

grid_search.fit(
    X_train,
    y_train
)


print()
print("Hyperparameter tuning completed!")
print()


# --------------------------------
# Show best settings
# --------------------------------

print("Best Hyperparameters:")
print(
    grid_search.best_params_
)
print()

print("Best Cross-Validation F1 Score:")
print(
    round(
        grid_search.best_score_,
        4
    )
)
print()


# --------------------------------
# Get best model
# --------------------------------

best_model = grid_search.best_estimator_


# --------------------------------
# Save best model
# --------------------------------

model_path = (
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2/"
    "models/tuned_random_forest_fraud_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)


print("Tuned model saved successfully!")
print(model_path)