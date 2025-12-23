import streamlit as st
import requests
import json

# Page Configuration
st.set_page_config(page_title="Bank Loan Risk Analyzer", layout="wide")

st.title("🏦 Credit Scoring & Risk Prediction")
st.markdown("Enter the applicant's details below to get an instant risk assessment.")

# 1. Setup Input Fields (Matching your GermanCredit dataset)
with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = st.selectbox("Checking Account Status", ["... < 100 DM", "0 <= ... < 200 DM", "no checking account", "... >= 200 DM / salary for at least 1 year"])
        duration = st.number_input("Duration (Months)", min_value=1, max_value=72, value=12)
        amount = st.number_input("Credit Amount", min_value=100, max_value=20000, value=2000)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    
    with col2:
        purpose = st.selectbox("Purpose", ["car (new)", "car (used)", "furniture/equipment", "radio/television", "domestic appliances", "repairs", "education", "retraining", "business", "others"])
        savings = st.selectbox("Savings Account", ["... < 100 DM", "100 <= ... < 500 DM", "500 <= ... < 1000 DM", "... >= 1000 DM", "unknown/no savings account"])
        employment_duration = st.selectbox("Employment Duration", ["unemployed", "... < 1 year", "1 <= ... < 4 years", "4 <= ... < 7 years", "... >= 7 years"])
        job = st.selectbox("Job Type", ["unemployed/unskilled - non-resident", "unskilled - resident", "skilled employee/official", "management/self-employed/highly qualified employee/officer"])

    with col3:
        housing = st.selectbox("Housing Status", ["own", "rent", "for free"])
        credit_history = st.selectbox("Credit History", ["critical account/other credits existing", "existing credits paid back duly till now", "delay in paying off in the past", "no credits taken/all credits paid back duly", "all credits at this bank paid back duly"])
        telephone = st.radio("Telephone", ["yes", "no"])
        foreign_worker = st.radio("Foreign Worker", ["yes", "no"])

    # Hardcoded/Default values for remaining fields required by API
    submitted = st.form_submit_button("Predict Risk Level")

# 2. Handle Prediction
if submitted:
    # Prepare the payload for FastAPI
    payload = {
        "status": status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "amount": amount,
        "savings": savings,
        "employment_duration": employment_duration,
        "installment_rate": 4,  # Default
        "personal_status_sex": "male : single",  # Default
        "other_debtors": "none",
        "present_residence": 3,
        "property": "real estate",
        "age": age,
        "other_installment_plans": "none",
        "housing": housing,
        "number_credits": 1,
        "job": job,
        "people_liable": 1,
        "telephone": telephone,
        "foreign_worker": foreign_worker
    }

    try:
        # Call the FastAPI /predict endpoint
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        st.divider()
        
        # 3. Display Results
        if result["risk_classification"] == "Low Risk (Approved)":
            st.success(f"✅ Prediction: {result['risk_classification']}")
        else:
            st.error(f"❌ Prediction: {result['risk_classification']}")
            
        st.info(f"Confidence Score: {result['confidence_score'] * 100:.2f}%")
        
    except Exception as e:
        st.error(f"Error: Could not connect to the API. Make sure FastAPI is running. {e}")