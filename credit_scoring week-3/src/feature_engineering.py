import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def get_prepared_data(file_path):
    df = pd.read_csv(file_path)
    
    # Target: credit_risk (1=Good, 0=Bad)
    X = df.drop(columns=['credit_risk'])
    y = df['credit_risk']
    
    # One-Hot Encoding for categorical variables (Required for SMOTE/XGBoost)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Split BEFORE SMOTE to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Day 2 Goal: Handle Class Imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test, X_encoded.columns.tolist()

if __name__ == "__main__":
    # Test loading
    X_tr, X_te, y_tr, y_te, cols = get_prepared_data("data/processed/cleaned_credit.csv")
    print(f"✅ Features engineered. Training set size after SMOTE: {len(X_tr)}")