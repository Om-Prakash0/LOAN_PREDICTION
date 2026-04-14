import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model
# -------------------------------
data = joblib.load("loan_model.pkl")
model = data["model"]
feature_columns = data["feature_columns"]

st.title("🏦 Loan Approval Prediction (ANN)")

st.sidebar.header("Enter Applicant Details")

# -------------------------------
# Inputs (MATCH YOUR DATASET)
# -------------------------------
no_of_dependents = st.sidebar.number_input("Dependents", 0)

education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

income = st.sidebar.number_input("Income (Annual)", 0)
loan_amount = st.sidebar.number_input("Loan Amount", 0)
loan_term = st.sidebar.number_input("Loan Term", 12)

cibil_score = st.sidebar.number_input("CIBIL Score", 300, 900)

res_assets = st.sidebar.number_input("Residential Assets", 0)
comm_assets = st.sidebar.number_input("Commercial Assets", 0)
lux_assets = st.sidebar.number_input("Luxury Assets", 0)
bank_assets = st.sidebar.number_input("Bank Assets", 0)

# -------------------------------
# Encoding
# -------------------------------
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# -------------------------------
# Create DataFrame (IMPORTANT)
# -------------------------------
input_data = pd.DataFrame([{
    "no_of_dependents": no_of_dependents,
    "education": education,
    "self_employed": self_employed,
    "income_annum": income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": res_assets,
    "commercial_assets_value": comm_assets,
    "luxury_assets_value": lux_assets,
    "bank_asset_value": bank_assets
}])

# Ensure correct column order
input_data = input_data[feature_columns]

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")