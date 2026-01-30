import streamlit as st
import pandas as pd
import joblib
import time
import os

# Set Page Config
st.set_page_config(page_title="Fraud Guard: E-Commerce Detection", layout="wide")

st.title("🛡️ E-Commerce Fraud Detection Dashboard")

# --- SIDEBAR: Load Model & Data ---
st.sidebar.header("Project Settings")
model_path = 'models/fraud_model.pkl'
data_path = 'data/X_test.csv'

if os.path.exists(model_path) and os.path.exists(data_path):
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    st.sidebar.success("✅ Model & Data Loaded")
else:
    st.sidebar.error("❌ Files missing. Please run your training script first.")
    st.stop()

# --- MAIN SECTION: KPIs ---
st.header("📊 Business Impact (KPIs)")
col1, col2, col3 = st.columns(3)

# Logic for KPIs
preds = model.predict(data)
fraud_caught = data[(preds == 1)]['Amount'].sum()
false_positives = len(data[(preds == 1)]) # Simplified for demo

col1.metric("Total Fraud Caught", f"${fraud_caught:,.2f}")
col2.metric("False Alarms", f"{false_positives}")
col3.metric("Estimated Savings", f"${fraud_caught - (false_positives * 5):,.2f}")

# --- SECTION: Stream Simulation ---
st.header("🕒 Real-Time Transaction Stream")
if st.button("Start Live Simulation"):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Show the last 10 transactions one by one
    for i in range(10):
        row = data.iloc[[i]]
        prediction = model.predict(row)
        
        if prediction[0] == 1:
            st.error(f"⚠️ ALERT: Transaction {i} flagged as FRAUD! (Amount: ${row['Amount'].values[0]})")
        else:
            st.success(f"✅ Transaction {i} Cleared. (Amount: ${row['Amount'].values[0]})")
        
        progress_bar.progress((i + 1) * 10)
        time.sleep(0.8)