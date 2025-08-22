import streamlit as st
import pandas as pd
import joblib

# Load the model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Telco Customer Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Prediction")
st.write("Fill in all the details below to check churn probability:")

# -------------------------------
# Numeric features
# -------------------------------
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Predict"):
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges],
        "gender": [gender],
        "SeniorCitizen": [1 if SeniorCitizen=="Yes" else 0],
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

    st.write(f"**Prediction:** {'Churn' if pred == 1 else 'No Churn'}")
    st.write(f"**Churn Probability:** {prob:.2f}")
