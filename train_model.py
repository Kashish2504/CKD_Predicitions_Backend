import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import os
import io
import json

def train_and_save():
    print("Loading actual dataset...")
    file_path = 'Chronic_Kidney_Disease/chronic_kidney_disease_full.arff'
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please check the exact file name.")
        return

    print("Parsing messy ARFF file using Pandas...")
    col_names = []
    data_lines = []
    is_data = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue  # Skip empty lines and comments
        
        if not is_data:
            if line.lower().startswith('@attribute'):
                # Extract the column name (e.g., from "@attribute 'age' numeric")
                parts = line.split()
                col_name = parts[1].replace("'", "").replace('"', "")
                col_names.append(col_name)
            elif line.lower() == '@data':
                is_data = True
        else:
            # Clean up spaces around commas to fix the "yes " vs "yes" issue
            clean_line = ','.join([val.strip() for val in line.split(',')])
            data_lines.append(clean_line)

    # Convert our cleaned lines directly into a Pandas DataFrame
    # na_values=['?'] tells Pandas that "?" means missing data
    csv_data = '\n'.join(data_lines)
    df = pd.read_csv(io.StringIO(csv_data), names=col_names, na_values=['?'])

    print("Preprocessing data...")
    
    # Extra safety: Clean up any hidden spaces left in text columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', np.nan) # Fix pandas string casting

    # The target column in this dataset is named 'class'
    target_col = 'class'
    if target_col not in df.columns:
        print(f"Error: Could not find '{target_col}' column.")
        return

    # Map target to 1 (CKD) and 0 (Not CKD)
    y = df[target_col].map({'ckd': 1, 'notckd': 0})
    X = df.drop(target_col, axis=1)

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # --- Handle Missing Values ---
    print("Filling missing values...")
    # Fill missing numbers with the average (mean)
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

    # Fill missing text categories with the most frequent value (mode)
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    # --- Encode Text to Numbers ---
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save feature names for the FastAPI backend
    feature_names = X.columns.tolist()

    print("Training Gradient Boosting Model (XGBoost)...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Check Accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

    print("Saving model to 'model' directory...")
    os.makedirs('model', exist_ok=True)
    
    # Save files
    model.save_model("model/ckd_xgb_model.json")
    with open("model/model_features.json", "w") as f:
        json.dump(feature_names, f)

    print("Success! Check the 'model' folder for your trained files.")

if __name__ == "__main__":
    train_and_save()