import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_model():
    # Ensure these paths match your folder structure exactly
    model = joblib.load('models/churn_model.pkl')
    cols = joblib.load('models/model_columns.pkl')
    return model, cols

try:
    model, model_columns = load_model()
except:
    st.error("⚠️ Model files not found. Run your notebook cells to save 'churn_model.pkl' and 'model_columns.pkl' in the 'models' folder first!")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📋 Customer Information")

def user_input_features():
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 8000.0, 500.0)
    cltv = st.sidebar.number_input("Customer Lifetime Value (CLTV)", 2000, 7000, 4000)
    
    if tenure <= 12: t_group = '0-1 Year'
    elif tenure <= 24: t_group = '1-2 Years'
    elif tenure <= 48: t_group = '2-4 Years'
    else: t_group = '4+ Years'
    
    data = {
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'CLTV': cltv,
        'TenureGroup': t_group
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- PREDICT BUTTON ---
predict_btn = st.sidebar.button("🚀 Predict Churn Risk")

# --- MAIN PAGE ---
st.title("📞 Customer Churn Prediction Dashboard")
st.write("Adjust the customer details in the sidebar and click **Predict** to see results.")
st.divider()

if predict_btn:
    # 1. Preprocessing for prediction
    df_pred = pd.DataFrame(columns=model_columns)
    df_pred.loc[0] = 0 # Initialize with zeros
    
    df_pred['Tenure'] = input_df['Tenure']
    df_pred['MonthlyCharges'] = input_df['MonthlyCharges']
    df_pred['TotalCharges'] = input_df['TotalCharges']
    df_pred['CLTV'] = input_df['CLTV']

    t_col = f"TenureGroup_{input_df['TenureGroup'][0]}"
    if t_col in df_pred.columns:
        df_pred[t_col] = 1

    # 2. Run Prediction
    prediction = model.predict(df_pred)
    prediction_proba = model.predict_proba(df_pred)[0] # [Stay_Prob, Churn_Prob]

    # 3. Display Results in Columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Analysis Result")
        if prediction[0] == 1:
            st.error("### 🚨 HIGH RISK\nCustomer is likely to leave (Churn).")
        else:
            st.success("### ✅ LOW RISK\nCustomer is likely to stay.")

    with col2:
        st.subheader("Churn Probability")
        churn_percent = prediction_proba[1] * 100
        st.write(f"The model is **{churn_percent:.1f}%** confident in this prediction.")
        st.progress(prediction_proba[1])

    st.divider()
    st.info(f"💡 **Key Observation:** At a {churn_percent:.1f}% risk level, the company should consider a retention offer.")
else:
    st.info("👈 Fill in the customer details and click the **Predict Churn Risk** button in the sidebar.")