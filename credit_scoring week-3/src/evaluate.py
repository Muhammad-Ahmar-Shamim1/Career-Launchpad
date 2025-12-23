import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # <--- ADD THIS LINE FIRST
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os # Added to ensure directory exists
from sklearn.metrics import classification_report, confusion_matrix
from feature_engineering import get_prepared_data

def evaluate():
    # Load model
    model = joblib.load("models/credit_model.pkl")
    
    # Load data
    _, X_test, _, y_test, _ = get_prepared_data("data/processed/cleaned_credit.csv")
    
    # Clean columns
    X_test.columns = [re.sub(r'[\[\]<]', '_', str(col)) for col in X_test.columns]
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Print Results
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create the reports folder if it doesn't exist
    os.makedirs('reports/visuals/', exist_ok=True)
    
    # Generate Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Credit Risk Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the file
    plt.savefig('reports/visuals/confusion_matrix.png')
    plt.close() # Close the figure to free up memory
    print("✅ Evaluation complete. Visuals saved to reports/visuals/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()