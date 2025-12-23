import pandas as pd
import joblib
from fastapi import FastAPI
from api.schema import LoanApplication

app = FastAPI(title="Credit Scoring API")

# Load model and feature columns at startup
model = joblib.load("models/credit_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

@app.get("/")
def home():
    return {"message": "Credit Scoring API is Live"}

@app.post("/predict")
def predict_risk(data: LoanApplication):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Preprocess (One-hot encoding to match training)
    input_encoded = pd.get_dummies(input_df)
    
    # Reindex to ensure all columns exist and are in the same order
    input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Predict
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]
    
    return {
        "risk_classification": "Low Risk (Approved)" if prediction == 1 else "High Risk (Denied)",
        "confidence_score": round(float(probability), 2)
    }