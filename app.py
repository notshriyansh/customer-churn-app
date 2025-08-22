import streamlit as st
import pandas as pd
import joblib
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Model file not found. Make sure model.pkl is in the same folder as app.py.")
    st.stop()


st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer details below to check churn probability:")

# Input features
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Make predictions
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {'Churn' if pred == 1 else 'No Churn'}")
    st.write(f"**Churn Probability:** {prob:.2f}")
