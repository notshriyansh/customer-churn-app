import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Telco Customer Churn Predictor", layout="wide")

st.title("📊 Telco Customer Churn Prediction")
st.markdown(
    """
    Welcome to the **Customer Churn Prediction App**.  
    Provide customer details in the form below and check the probability of churn.  
    """
)

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    """
    - **Dataset**: Telco Customer Churn (Kaggle)  
    - **Model**: Logistic Regression / Decision Tree  
    - **Target**: Predict whether a customer will churn  
    """
)

# -------------------------------
# Input Sections
# -------------------------------
st.subheader("📌 Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    Partner = st.selectbox("Partner", ["Yes", "No"])

with col2:
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col3:
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

# -------------------------------
st.subheader("📌 Entertainment & Billing")
col4, col5 = st.columns(2)

with col4:
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col5:
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🚀 Predict Churn"):
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges],
        "gender": [gender],
        "SeniorCitizen": [1 if SeniorCitizen == "Yes" else 0],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod]
    })

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # -------------------------------
    # Result Display
    # -------------------------------
    st.markdown("### 🎯 Prediction Result")
    if pred == 1:
        st.error(f"❌ The customer is **likely to Churn**")
    else:
        st.success(f"✅ The customer is **likely to Stay**")

    st.progress(int(prob * 100))
    st.write(f"**Churn Probability:** {prob:.2f}")
