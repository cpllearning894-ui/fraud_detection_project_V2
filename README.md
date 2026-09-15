# Fraud Detection Machine Learning Project

A machine learning project that detects potentially fraudulent
transactions using a Random Forest classifier.

The project includes data preprocessing, model training,
model comparison, threshold analysis, error analysis,
ROC-AUC and Precision-Recall evaluation, and an interactive
Streamlit application for predicting new transactions.

## Project Objective

The objective of this project is to build a machine learning
system that can identify potentially fraudulent transactions
and provide a risk assessment to support further investigation.

The project focuses on:

- Detecting potentially fraudulent transactions
- Evaluating model performance
- Understanding false positives and false negatives
- Selecting an appropriate decision threshold
- Applying business cost considerations
- Building an interactive prediction application

## Dataset

The dataset contains 5,000 transaction records.

Each transaction contains information about transaction
behavior and account characteristics.

### Features

| Feature | Description |
|---|---|
| transaction_amount | Amount of the transaction |
| transaction_hour | Hour when the transaction occurred |
| account_age_days | Age of the account in days |
| previous_transactions | Number of previous transactions |
| is_international | Whether the transaction is international |
| is_new_device | Whether a new device was used |
| failed_attempts | Number of failed attempts |
| is_fraud | Target variable indicating whether the transaction is fraudulent |

The target variable is is_fraud, where:

- 0 = Normal transaction
- 1 = Fraudulent transaction

## Train/Test Split

The dataset was divided into training and testing sets using an
80/20 split.

- Training data: 4,000 transactions
- Testing data: 1,000 transactions
- Split ratio: 80% training / 20% testing
- Stratification: Used to preserve the fraud/non-fraud class distribution
- Random state: 42

The training set was used to train the machine learning models,
while the test set was kept separate and used to evaluate how
well the models perform on unseen transactions.

## Data Preprocessing

Before training the machine learning models, the dataset was
checked and prepared to ensure that the data was suitable for
model training.

The preprocessing steps included:

- Checking for missing values
- Checking for duplicate records
- Separating input features from the target variable
- Preparing the training and testing datasets
- Saving the processed dataset for reproducibility

The dataset was checked for missing values and duplicate records,
and the processed data was saved in the data/processed/ directory.

The original input features were kept consistent between training
and testing so that the model could make predictions correctly
on unseen transactions.

## Model Development

Several Random Forest approaches were tested during the project
to compare different strategies for fraud detection.

The experiments included:

1. Original Random Forest
   - Used as the baseline model.
   - Provided the strongest F1 score among the tested models.

2. Class-Weighted Random Forest
   - Applied class weighting to give more importance to the
     minority fraud class.

3. SMOTE Random Forest
   - Used SMOTE to create synthetic examples of the minority
     fraud class in the training data.

4. Hyperparameter-Tuned Random Forest
   - Used GridSearchCV with 5-fold cross-validation to search
     for better model parameters.

5. Feature-Engineered Random Forest
   - Added derived features such as failed attempt rate,
     night-time activity, high transaction amount, and
     transactions per day.

These experiments were compared using accuracy, precision,
recall, and F1 score.

The original Random Forest was selected as the final model
because it achieved the highest F1 score among the tested
approaches.

## Model Performance

The tested models were evaluated on the same unseen test dataset.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Original Random Forest | 90.4% | 49.5% | 50.5% | 50.0% |
| Class-Weighted Random Forest | 90.4% | 49.5% | 50.5% | 50.0% |
| SMOTE Random Forest | 82.9% | 27.9% | 50.5% | 36.0% |
| Tuned Random Forest | 91.8% | 66.7% | 27.4% | 38.8% |
| Feature-Engineered Random Forest | 90.4% | 49.5% | 47.4% | 48.4% |

The Original Random Forest achieved the highest F1 score among
the tested approaches and was selected as the final model.

The final model achieved:

- Accuracy: 90.4%
- Precision: 49.5%
- Recall: 50.5%
- F1 Score: 50.0%
- ROC-AUC: 87.54%
- PR-AUC: 49.18%

Because fraud detection is an imbalanced classification problem,
accuracy alone is not sufficient to evaluate the model.
Precision, recall, F1 score, ROC-AUC, and PR-AUC were also
considered.

## Confusion Matrix

The final Random Forest model was evaluated using a confusion
matrix on the unseen test dataset.

The results were:

| | Predicted Normal | Predicted Fraud |
|---|---:|---:|
| Actual Normal | 856 | 49 |
| Actual Fraud | 47 | 48 |

This gives:

- True Negatives (TN): 856 — normal transactions correctly identified as normal
- False Positives (FP): 49 — normal transactions incorrectly flagged as fraud
- False Negatives (FN): 47 — fraudulent transactions incorrectly classified as normal
- True Positives (TP): 48 — fraudulent transactions correctly identified

The confusion matrix helps explain the types of mistakes made by
the fraud detection model.

In a fraud detection system:

- False negatives can represent missed fraudulent transactions.
- False positives can result in unnecessary investigations or
  inconvenience for legitimate customers.

Therefore, the balance between false positives and false
negatives is an important business consideration.

## Threshold Analysis

The Random Forest model produces a probability of fraud for each
transaction.

Instead of automatically using a probability of 0.50 as the
decision threshold, different thresholds were evaluated to
understand the trade-off between detecting fraud and generating
false alarms.

For example:

- A lower threshold identifies more potentially fraudulent
  transactions but can increase false positives.
- A higher threshold reduces false positives but can increase
  false negatives.

The feature-engineered model was evaluated at several thresholds,
including 0.30, 0.40, 0.50, 0.60, and 0.70.

Based on an illustrative business-cost analysis, where a missed
fraudulent transaction was assigned a higher cost than a false
alarm, a threshold of 0.30 produced the lowest estimated
cost among the tested thresholds.

The final application therefore uses a 0.30 decision threshold.

The costs used in this analysis are illustrative assumptions for
demonstrating business decision-making and do not represent
actual financial losses.

## ROC-AUC and Precision-Recall Evaluation

Because fraud detection is a classification problem with an
imbalanced target, additional evaluation metrics were used to
understand the model's performance across different probability
thresholds.

### ROC-AUC

The final Random Forest model achieved a ROC-AUC score of
0.8754.

ROC-AUC measures how well the model can distinguish between
fraudulent and normal transactions across different classification
thresholds.

A higher ROC-AUC generally indicates better ability to rank
fraudulent transactions above normal transactions.

### Precision-Recall

The model achieved a PR-AUC score of 0.4918.

Precision-Recall evaluation focuses specifically on the trade-off
between precision and recall for the fraud class.

This is especially useful for fraud detection because fraudulent
transactions are the minority class.

The project includes both ROC and Precision-Recall analysis files
in the reports/ directory.

## Error Analysis

Model evaluation does not stop at accuracy and other performance
metrics. The incorrect predictions were also analyzed to understand
where the fraud detection model struggles.

The test predictions were separated into:

- False Positives (FP) — normal transactions incorrectly
  predicted as fraud
- False Negatives (FN) — fraudulent transactions incorrectly
  predicted as normal
- True Positives (TP) — fraudulent transactions correctly
  detected
- True Negatives (TN) — normal transactions correctly
  identified

The project saved detailed error-analysis results in the
reports/ directory, including:

- false_positives.csv
- false_negatives.csv
- false_negative_deep_analysis.csv
- false_negative_vs_true_positive.csv
- false_negative_probability_analysis.csv
- compare_probability_groups.csv
- quantitative_error_analysis.csv

### False Negative Analysis

The model produced 50 false negatives for the feature-engineered
experiment.

The analysis showed that some missed fraudulent transactions
looked similar to normal transactions.

Compared with detected fraud, missed fraud tended to have:

- Older accounts
- More previous transaction history
- Fewer failed attempts
- Lower failed-attempt rates
- More activity during night-time hours
- Higher transactions per day

This suggests that some fraudulent transactions are difficult to
detect because their behavior is less obviously suspicious.

The analysis describes patterns in the model's errors and should
not be interpreted as proof that these characteristics cause fraud.

## Streamlit Application

The project includes an interactive Streamlit application that
allows users to enter transaction information and receive a fraud
risk assessment.

The application allows the user to enter:

- Transaction amount
- Transaction hour
- Account age
- Previous transactions
- International transaction status
- New device status
- Failed attempts

After the user submits the transaction, the application:

1. Sends the transaction information to the trained Random Forest
   model.
2. Calculates the estimated fraud probability.
3. Compares the probability with the decision threshold.
4. Produces a fraud or normal prediction.
5. Displays a risk level.

The application uses three risk levels:

- 🟢 LOW — probability below the decision threshold
- 🟡 MEDIUM — probability at or above the threshold but below 70%
- 🔴 HIGH — probability of 70% or higher

The application also provides:

- Model information
- Project performance metrics
- Confusion matrix
- ROC curve
- Business interpretation
- Model limitations
- Project summary
- Downloadable prediction reports

## Project Structure

```text
fraud_detection_project_V2/
│
├── data/
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── fraud_transactions_processed.csv
│
├── models/
│   ├── random_forest_fraud_model.pkl
│   ├── improved_random_forest_fraud_model.pkl
│   ├── feature_engineered_random_forest_fraud_model.pkl
│   └── final_fraud_model.pkl
│
├── reports/
│   ├── final_model_selection.csv
│   ├── quantitative_error_analysis.csv
│   ├── false_negative_deep_analysis.csv
│   ├── false_negative_vs_true_positive.csv
│   ├── false_negative_probability_analysis.csv
│   ├── compare_probability_groups.csv
│   ├── threshold_tuning_engineered.csv
│   ├── roc_auc_analysis.csv
│   └── precision_recall_analysis.csv
│
├── src/
│   ├── app.py
│   ├── predict_new_transaction.py
│   ├── final_model_selection.py
│   ├── save_final_model.py
│   ├── roc_auc_analysis.py
│   ├── precision_recall_analysis.py
│   └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```

### 🧠 What each main folder means

| Folder | Purpose |
|---|---|
| data/ | Dataset and processed data |
| models/ | Trained machine-learning models |
| reports/ | Evaluation and analysis results |
| src/ | Python source code |
| README.md | Project documentation |
| requirements.txt | Python libraries required to run the project |
| .gitignore | Files Git should not track |

### 💼 Why employers like this

Imagine opening a GitHub repository and seeing:

```textS
data
models
reports
src
README.md
requirements.txt

 ## Installation and Setup
``` 

### 1. Clone the Repository

Clone this repository to your computer:

```bash
git clone <your-github-repository-url>
cd fraud_detection_project_V2
```

## How to Use the Application

1. Start the Streamlit application using:

   ```bash
   streamlit run src/app.py

## Model Limitations

This project is a portfolio machine learning project and has
several limitations.

- The model produces an estimated probability of fraud; it does
  not prove that a transaction is fraudulent.
- The model was trained on the available historical dataset and
  may not generalize to every real-world fraud scenario.
- Fraud patterns can change over time, so the model would need
  regular monitoring and retraining in a production environment.
- False positives may cause legitimate transactions to be
  investigated or temporarily flagged.
- False negatives may allow fraudulent transactions to go
  undetected.
- The decision threshold can be adjusted depending on business
  requirements and the relative cost of different types of errors.
- The illustrative cost analysis in this project does not
  represent actual financial losses from a real financial
  institution.
- The model should be treated as a decision-support tool rather
  than a replacement for human investigation and additional
  security controls.

## Future Improvements

The project could be further improved in several ways:

- Train the model on a larger and more realistic fraud dataset.
- Add additional transaction and customer behavior features.
- Monitor model performance after deployment.
- Retrain the model when fraud patterns change.
- Optimize the decision threshold using real business costs.
- Compare Random Forest with other algorithms such as XGBoost,
  LightGBM, or Logistic Regression.
- Perform more extensive hyperparameter optimization.
- Add model explainability techniques to help investigators
  understand why a transaction was flagged.
- Add data and model monitoring for a production environment.
- Deploy the Streamlit application to a cloud platform.
- Add authentication and access control for production use.

## Results and Key Findings

The final Random Forest model achieved an accuracy of 90.4%
and an F1 score of 50.0% on the unseen test dataset.

Additional evaluation produced:

- Precision: 49.5%
- Recall: 50.5%
- ROC-AUC: 0.8754
- PR-AUC: 0.4918

The project showed that accuracy alone is not enough for evaluating
a fraud detection system because fraud is the minority class.

Several modeling strategies were tested, including class
weighting, SMOTE, hyperparameter tuning, and feature engineering.
The original Random Forest achieved the highest F1 score among
the tested approaches and was selected as the final model.

Error analysis also showed that some fraudulent transactions were
difficult to distinguish from normal transactions, demonstrating
the importance of analyzing false negatives and false positives.

The project also demonstrated how a machine learning model can be
connected to business decision-making through probability
threshold selection and cost-sensitive analysis.

