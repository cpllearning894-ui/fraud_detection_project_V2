import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #F8FAFC;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #1E3A8A,
            #2563EB
        );
        padding: 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        margin-bottom: 0.5rem;
    }

    .hero p {
        margin-bottom: 0;
        color: #DBEAFE;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #1E293B;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .fraud-card {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
    }

    .normal-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
    }

    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-description {
        color: #475569;
        margin-bottom: 0;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# PROJECT PATHS
# ==================================================

project_path = Path(__file__).resolve().parent.parent

model_path = (
    project_path
    / "models"
    / "final_fraud_model.pkl"
)

config_path = (
    project_path
    / "reports"
    / "final_model_config.csv"
)

# ==================================================
# LOAD MODEL AND THRESHOLD
# ==================================================

model = joblib.load(model_path)

config = pd.read_csv(config_path)

final_threshold = config.loc[
    0,
    "threshold"
]

# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Fraud Detection System</h1>
        <p>
            AI-powered transaction risk assessment
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Enter transaction information to check "
    "whether the transaction looks suspicious."
)

# ==================================================
# INPUT SECTION
# ==================================================

st.markdown(
    '<div class="section-title">📝 Transaction Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    transaction_amount = st.number_input(
        "Transaction amount ($)",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

    transaction_hour = st.number_input(
        "Transaction hour",
        min_value=0,
        max_value=23,
        value=2
    )

    account_age_days = st.number_input(
        "Account age (days)",
        min_value=0,
        value=30
    )

    previous_transactions = st.number_input(
        "Previous transactions",
        min_value=0,
        value=5
    )

with col2:

    is_international = st.selectbox(
        "Is international?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_new_device = st.selectbox(
        "Is new device?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    failed_attempts = st.number_input(
        "Failed attempts",
        min_value=0,
        value=6
    )

# ==================================================
# PREDICTION BUTTON
# ==================================================

st.write("")

# Prediction button
if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
):

    # Create a DataFrame
    new_transaction = pd.DataFrame({
        "transaction_amount": [transaction_amount],
        "transaction_hour": [transaction_hour],
        "account_age_days": [account_age_days],
        "previous_transactions": [previous_transactions],
        "is_international": [is_international],
        "is_new_device": [is_new_device],
        "failed_attempts": [failed_attempts]
    })

    # Calculate fraud probability
    fraud_probability = model.predict_proba(
        new_transaction
    )[:, 1][0]

    # Apply the final threshold
    prediction = int(
        fraud_probability >= final_threshold
    )
    
    # ==================================================
    # RISK LEVEL
    # ==================================================

    if fraud_probability >= 0.70:
        risk_level = "HIGH"
        risk_icon = "🔴"

    elif fraud_probability >= final_threshold:
        risk_level = "MEDIUM"
        risk_icon = "🟡"

    else:
        risk_level = "LOW"
        risk_icon = "🟢"
        
        
    # ==================================================
    # DOWNLOAD PREDICTION REPORT
    # ==================================================

    report_text = f"""
FRAUD DETECTION REPORT

Transaction amount: ${transaction_amount:,.2f}
Transaction hour: {transaction_hour}
Account age: {account_age_days} days
Previous transactions: {previous_transactions}

International: {"Yes" if is_international == 1 else "No"}
New device: {"Yes" if is_new_device == 1 else "No"}
Failed attempts: {failed_attempts}

Fraud probability: {fraud_probability:.1%}
Risk level: {risk_level}
Prediction: {"FRAUD" if prediction == 1 else "NORMAL"}
Decision threshold: {final_threshold:.1%}
"""

    st.download_button(
        label="📥 Download Prediction Report",
        data=report_text,
        file_name="fraud_prediction_report.txt",
        mime="text/plain"
    )    
        
    # ==================================================
    # WHAT DOES THIS RESULT MEAN?
    # ==================================================

    st.markdown(
        '<div class="section-title">💡 What does this result mean?</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:
        st.write(
            f"The model estimated a "
            f"{fraud_probability:.1%} probability of fraud. "
            f"This is above the "
            f"{final_threshold:.1%} decision threshold, "
            "so the transaction was flagged for further review."
        )

    else:
        st.write(
            f"The model estimated a "
            f"{fraud_probability:.1%} probability of fraud. "
            f"This is below the "
            f"{final_threshold:.1%} decision threshold, "
            "so the transaction was not flagged by the current rule."
        ) 
        
        
    # ==================================================
    # MODEL INFORMATION
    # ==================================================

    st.markdown(
        '<div class="section-title">🤖 Model Information</div>',
        unsafe_allow_html=True
    )

    st.write("Model: Random Forest Classifier")
    st.write("Purpose: Detect suspicious transactions")
    st.write(
        f"Decision threshold: {final_threshold:.1%}"
    )

    st.caption(
        "The prediction is an estimate from the trained model, "
        "not proof that a transaction is fraudulent."
    )       

    # Display risk assessment
    st.markdown(
        '<div class="section-title">📊 Risk Assessment</div>',
        unsafe_allow_html=True
    )
    
    # Risk level indicator

    st.write(
        f"Risk level: {risk_icon} {risk_level}"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Fraud probability",
            f"{fraud_probability:.1%}",
            help="Estimated probability of fraud from the trained model."
        )

    with metric_col2:
        st.metric(
            "Decision threshold",
            f"{final_threshold:.1%}",
            help="Probability boundary used to classify the transaction."
        )

    # Probability progress bar
    st.progress(
        fraud_probability,
        text=f"Fraud probability: {fraud_probability:.1%}"
    )

    # Final decision
    if prediction == 1:

        st.markdown(
            """
            <div class="result-card fraud-card">
                <div class="result-title">
                    🚨 HIGH RISK — FRAUD
                </div>
                <p class="result-description">
                    This transaction has been flagged for
                    further review.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "The fraud probability is above the "
            "decision threshold."
        )

    else:

        st.markdown(
            """
            <div class="result-card normal-card">
                <div class="result-title">
                    ✅ LOW RISK — NORMAL
                </div>
                <p class="result-description">
                    This transaction has not been flagged
                    by the current decision rule.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "The fraud probability is below the "
            "decision threshold."
        )
        
# ==================================================
    # TRANSACTION SUMMARY
    # ==================================================

    st.markdown(
        '<div class="section-title">📋 Transaction Summary</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write(
            f"Transaction amount: "
            f"${transaction_amount:,.2f}"
        )

        st.write(
            f"Transaction hour: "
            f"{transaction_hour}"
        )

        st.write(
            f"Account age: "
            f"{account_age_days} days"
        )

        st.write(
            f"Previous transactions: "
            f"{previous_transactions}"
        )

    with summary_col2:

        st.write(
            f"International: "
            f"{'Yes' if is_international == 1 else 'No'}"
        )

        st.write(
            f"New device: "
            f"{'Yes' if is_new_device == 1 else 'No'}"
        )

        st.write(
            f"Failed attempts: "
            f"{failed_attempts}"
        )
        
# ==================================================
# HOW THE SYSTEM WORKS
# ==================================================

st.markdown(
    '<div class="section-title">⚙️ How the system works</div>',
    unsafe_allow_html=True
)

st.write(
    "1. Enter transaction details  \n"
    "The user provides information about the transaction."
)

st.write(
    "2. Machine learning analysis  \n"
    "The trained Random Forest model analyzes the input."
)

st.write(
    "3. Calculate fraud probability  \n"
    "The model estimates how suspicious the transaction may be."
)

st.write(
    "4. Apply the decision threshold  \n"
    "The probability is compared with the selected threshold."
)

st.write(
    "5. Display the result  \n"
    "The app shows the prediction and risk level."
)

# ==================================================
# PROJECT PERFORMANCE
# ==================================================

st.markdown(
    '<div class="section-title">📊 Project Performance</div>',
    unsafe_allow_html=True
)

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric("Accuracy", "90.4%")
    st.metric("Precision", "49.5%")
    st.metric("ROC-AUC", "87.54%")


with metric_col2:
    st.metric("Recall", "50.5%")
    st.metric("F1 Score", "50.0%")
    st.metric("PR-AUC", "49.18%")

st.caption(
    "These metrics describe the performance of the selected "
    "Random Forest model on the test dataset. ROC-AUC measures "
    "how well the model ranks fraudulent transactions above "
    "normal transactions across different thresholds."
    "ROC-AUC measures the model's ranking ability across "
    "different thresholds. PR-AUC summarizes the precision-"
    "recall trade-off for the fraud class."
)

# ==================================================
# ROC CURVE
# ==================================================

st.markdown(
    '<div class="section-title">📈 ROC Curve</div>',
    unsafe_allow_html=True
)

roc_path = (
    project_path
    / "reports"
    / "roc_auc_analysis.csv"
)

roc_data = pd.read_csv(roc_path)

st.line_chart(
    roc_data,
    x="false_positive_rate",
    y="true_positive_rate"
)

st.caption(
    "The ROC curve shows the trade-off between detecting "
    "fraud and incorrectly flagging normal transactions "
    "across different probability thresholds."
)

# ==================================================
# CONFUSION MATRIX
# ==================================================

st.markdown(
    '<div class="section-title">🧩 Confusion Matrix</div>',
    unsafe_allow_html=True
)

st.write(
    "The confusion matrix shows how the model's predictions "
    "compare with the actual transaction labels."
)

cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    st.metric("Correct normal transactions", "856")
    st.metric("Missed fraud transactions", "47")

with cm_col2:
    st.metric("False alarms", "49")
    st.metric("Detected fraud transactions", "48")

st.caption(
    "These values come from the original Random Forest model "
    "evaluated on the test dataset."
)        

# ==================================================
# BUSINESS INTERPRETATION
# ==================================================

st.markdown(
    '<div class="section-title">💼 Business Interpretation</div>',
    unsafe_allow_html=True
)

st.write(
    "A false negative means the model missed a fraudulent "
    "transaction. This is important because the business "
    "may lose money."
)

st.write(
    "A false positive means the model flagged a normal "
    "transaction. This may cause unnecessary review or "
    "inconvenience for a customer."
)

st.write(
    "The business should balance these two types of mistakes "
    "when choosing the decision threshold."
)

# ==================================================
# MODEL LIMITATIONS
# ==================================================

st.markdown(
    '<div class="section-title">⚠️ Model Limitations</div>',
    unsafe_allow_html=True
)

st.write(
    "• The model's prediction is an estimate, not proof of fraud."
)

st.write(
    "• The model was trained on historical transaction data."
)

st.write(
    "• The model may miss some fraudulent transactions."
)

st.write(
    "• The model may sometimes flag normal transactions."
)

st.write(
    "• The decision threshold can be adjusted depending on business needs."
)

# ==================================================
# PROJECT SUMMARY
# ==================================================

st.markdown(
    '<div class="section-title">📋 Project Summary</div>',
    unsafe_allow_html=True
)

st.write(
    "This project uses machine learning to identify "
    "potentially fraudulent transactions."
)

st.write(
    "A Random Forest model was trained on transaction data "
    "and evaluated on unseen test data."
)

st.write(
    "The application allows users to enter transaction "
    "details and receive a fraud probability, risk level, "
    "and prediction."
)

# ==================================================
# MODEL DEVELOPMENT JOURNEY
# ==================================================

st.markdown(
    '<div class="section-title">📈 Model Development Journey</div>',
    unsafe_allow_html=True
)

st.write(
    "We started with a baseline Random Forest model "
    "and tested several approaches to improve fraud detection."
)

st.write(
    "The experiments included class weighting, SMOTE, "
    "hyperparameter tuning, and feature engineering."
)

st.write(
    "After comparing the results, we selected the original "
    "Random Forest as the final baseline model because it "
    "achieved the highest F1 score among the tested models."
)

# ==================================================
# FOOTER
# ==================================================

# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <hr>
    <div style="text-align: center; color: gray;">
        Built with Python, Pandas, Scikit-learn, and Streamlit<br>
        Fraud Detection Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        Fraud Detection System • Random Forest Model<br>
        Risk assessment based on the trained model and selected threshold.
    </div>
    """,
    unsafe_allow_html=True
)