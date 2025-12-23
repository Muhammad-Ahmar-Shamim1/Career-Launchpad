import pandas as pd
import os

def clean_data(input_file, output_file):
    # Load data
    df = pd.read_csv(input_file)
    
    # 1. Basic Cleaning: Check for nulls
    df = df.dropna()
    
    # 2. Binary Encoding for simple categories
    # Based on GermanCredit.csv structure
    df['telephone'] = df['telephone'].map({'yes': 1, 'no': 0})
    df['foreign_worker'] = df['foreign_worker'].map({'yes': 1, 'no': 0})
    
    # Save to processed folder
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Data cleaned and saved to {output_file}")

if __name__ == "__main__":
    clean_data("data/raw/GermanCredit.csv", "data/processed/cleaned_credit.csv")