import io
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import os
import json

def train_and_save():
    print("Loading actual dataset...")
    file_path = 'Chronic_Kidney_Disease/chronic_kidney_disease_full.arff'
    
    try:
        # 1. Read the file manually first
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # 2. Clean out the sneaky spaces and tabs from the data rows
        cleaned_lines = []
        is_data = False
        for line in lines:
            if is_data:
                # Remove spaces and tabs that cause the ValueError
                line = line.replace(' ', '').replace('\t', '')
            if line.strip().lower() == '@data':
                is_data = True
            cleaned_lines.append(line)
            
        # 3. Feed the cleaned data to scipy
        cleaned_file = io.StringIO(''.join(cleaned_lines))
        data, meta = arff.loadarff(cleaned_file)
        
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please check the exact file name inside the folder.")
        return

    df = pd.DataFrame(data)

    # Decode byte strings to normal strings (ARFF quirk)
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode('utf-8')

    # ... [THE REST OF YOUR CODE REMAINS EXACTLY THE SAME FROM HERE] ...
    # Replace '?' (missing values in ARFF) with NaN
    df.replace('?', np.nan, inplace=True)

    print("Preprocessing data...")
    # The target column in this dataset is named 'class'
    y = df['class']
    X = df.drop('class', axis=1)

    # Encode the target variable (ckd = 1, notckd = 0)
    y = y.map({'ckd': 1, 'notckd': 0})

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Convert numerical columns that might have been read as strings back to floats
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # --- Handle Missing Values (Imputation) ---
    num_imputer = SimpleImputer(strategy='mean')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    # --- Encode Categorical Features ---
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

    accuracy = model.score(X_test, y_test)
    print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

    print("Saving model to 'model' directory...")
    # Ensure the model directory exists (like in your previous code)
    os.makedirs('model', exist_ok=True)

    # Save the model
    model.save_model("model/ckd_xgb_model.json")

    # Save the feature names so FastAPI knows what order to feed the data
    with open("model/model_features.json", "w") as f:
        json.dump(feature_names, f)

    print("Success! Check the 'model' folder for your trained files.")

if __name__ == "__main__":
    train_and_save()