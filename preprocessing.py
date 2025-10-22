import pandas as pd
import numpy as np

def preprocess(df, mode='Online'):
    """
    Preprocess input data for prediction
    mode: 'Online' for single prediction, 'Batch' for multiple
    """
    
    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Binary encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Gender encoding
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # One-hot encoding for categorical columns
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod', 
                       'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies']
    
    # Apply one-hot encoding
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    return df
