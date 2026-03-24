import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="ChurnGuard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Load Model and Columns
# -------------------------------
model = pickle.load(open("xgb_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# -------------------------------
# Header
# -------------------------------
st.title("📊 ChurnGuard: A Machine Learning System for Customer Retention in Telecoms")

st.markdown("""
**ChurnGuard** is a machine learning-powered system designed to predict customer churn in the telecommunications sector.

By analyzing customer attributes such as **subscription patterns, billing behavior, and service usage**, the system identifies customers who are most likely to leave a telecom service provider.

The predictive model was optimized using the **F1-score**, balancing precision and recall to ensure accurate churn detection.  
The solution is deployed using **Streamlit**, allowing users to interact with the model and generate real-time predictions through an intuitive interface.

By helping telecom companies identify at-risk customers early, **ChurnGuard supports proactive retention strategies, reduces revenue loss, and promotes financial inclusion by ensuring continued access to essential communication services.**
""")

st.divider()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📋 Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

SeniorCitizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

Partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])

MultipleLines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

InternetService = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

Contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0)

# -------------------------------
# Prediction Section
# -------------------------------
st.header("🔎 Churn Prediction")

if st.button("Predict Customer Churn"):

    input_df = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[SeniorCitizen],
        "Partner":[Partner],
        "Dependents":[Dependents],
        "tenure":[tenure],
        "PhoneService":[PhoneService],
        "MultipleLines":[MultipleLines],
        "InternetService":[InternetService],
        "OnlineSecurity":[OnlineSecurity],
        "OnlineBackup":[OnlineBackup],
        "DeviceProtection":[DeviceProtection],
        "TechSupport":[TechSupport],
        "StreamingTV":[StreamingTV],
        "StreamingMovies":[StreamingMovies],
        "Contract":[Contract],
        "PaperlessBilling":[PaperlessBilling],
        "PaymentMethod":[PaymentMethod],
        "MonthlyCharges":[MonthlyCharges],
        "TotalCharges":[TotalCharges]
    })

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Align columns with training data
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 1:
            st.error("⚠️ High Risk of Customer Churn")
        else:
            st.success("✅ Customer Likely to Stay")

    with col2:

        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )

    st.progress(probability)

    if prediction == 1:

        st.warning("""
        **Recommended Action**

        The customer shows a high probability of churn. Telecom providers may consider:

        • Offering loyalty incentives  
        • Providing service upgrades  
        • Addressing service dissatisfaction  
        """)

    else:

        st.info("""
        **Customer Status**

        The customer currently shows a low risk of churn. Continue maintaining good service experience.
        """)

# -------------------------------
# Footer
# -------------------------------
st.divider()

st.caption("ChurnGuard | Machine Learning for Telecom Customer Retention | Built with Streamlit")