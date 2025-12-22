import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Telecom Churn AI", layout="wide")

# Load saved model and columns
@st.cache_resource
def load_assets():
    model = joblib.load('models/churn_model.pkl')
    cols = joblib.load('models/model_columns.pkl')
    return model, cols

model, model_columns = load_assets()

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges ($)", 10.0, 150.0, 70.0)
total = st.sidebar.number_input("Total Charges ($)", 10.0, 8000.0, 500.0)
cltv = st.sidebar.number_input("CLTV Score", 2000, 7000, 4000)

# Prediction Logic
if st.sidebar.button("Predict Churn"):
    # Create a blank dataframe matching training columns
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Fill basic numerical values
    input_data['Tenure'] = tenure
    input_data['MonthlyCharges'] = monthly
    input_data['TotalCharges'] = total
    input_data['CLTV'] = cltv
    
    # Fill tenure group dummy
    if tenure <= 12: group = "TenureGroup_0-1 Year"
    elif tenure <= 24: group = "TenureGroup_1-2 Years"
    elif tenure <= 48: group = "TenureGroup_2-4 Years"
    else: group = "TenureGroup_4+ Years"
    
    if group in input_data.columns:
        input_data[group] = 1

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
    with col2:
        if prob > 0.5:
            st.error("🚨 High Risk: This customer is likely to leave.")
        else:
            st.success("✅ Low Risk: This customer is likely to stay.")