import joblib
import re  # Added for regex cleaning
from xgboost import XGBClassifier
from feature_engineering import get_prepared_data

def train():
    # Load data
    X_train, X_test, y_train, y_test, feature_names = get_prepared_data("data/processed/cleaned_credit.csv")
    
    # --- FIX START: Sanitize Column Names ---
    # This regex removes [, ], and < to satisfy XGBoost requirements
    X_train.columns = [re.sub(r'[\[\]<]', '_', str(col)) for col in X_train.columns]
    X_test.columns = [re.sub(r'[\[\]<]', '_', str(col)) for col in X_test.columns]
    
    # Update the feature_names list to match the new sanitized columns
    feature_names = list(X_train.columns)
    # --- FIX END ---

    # Day 3 Goal: Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        use_label_encoder=False
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Save model and feature list for the API
    joblib.dump(model, "models/credit_model.pkl")
    joblib.dump(feature_names, "models/feature_columns.pkl")
    print("✅ Model and Feature List saved to models/")

if __name__ == "__main__":
    train()