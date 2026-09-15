import pandas as pd
import joblib
from pathlib import Path


# 1. Define project paths
project_path = Path(
    "D:/DataScience_Portfolio/"
    "fraud_detection_project_V2"
)

model_path = (
    project_path
    / "models"
    / "random_forest_fraud_model.pkl"
)

final_model_path = (
    project_path
    / "models"
    / "final_fraud_model.pkl"
)

config_path = (
    project_path
    / "reports"
    / "final_model_config.csv"
)


# 2. Define the final threshold
final_threshold = 0.30


# 3. Load the selected trained model
model = joblib.load(model_path)


# 4. Save it with the final model name
joblib.dump(model, final_model_path)


# 5. Create the final configuration
config = pd.DataFrame({
    "model_name": ["Original Random Forest"],
    "model_file": ["final_fraud_model.pkl"],
    "threshold": [final_threshold]
})


# 6. Save the configuration
config.to_csv(
    config_path,
    index=False
)


# 7. Display confirmation
print("\nFINAL MODEL SAVED")
print("Model:", final_model_path)
print("Threshold:", final_threshold)
print("Config:", config_path)