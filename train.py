# train.py
# Run this script once locally to generate the model files.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
import joblib

print("Starting model training process...")

# --- 1. Load Data ---
df = pd.read_csv('predictive_maintenance.csv')

# --- 2. Preprocess Data ---
df = df.drop(['UDI', 'Product ID'], axis=1)

# Define feature columns and target column
# 'Target' is included as a feature because it correlates with failure types
feature_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Target']
target_column = 'Failure Type'

X = df[feature_columns]
y = df[target_column]

# --- 3. Encode Features and Target ---
type_encoder = LabelEncoder()
target_encoder = LabelEncoder()

# Fit and transform the 'Type' column
X['Type'] = type_encoder.fit_transform(X['Type'])
y_encoded = target_encoder.fit_transform(y)

# --- 4. Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled.")

# --- 5. Train Model ---
print("Training the LGBM Classifier...")
lgbm = LGBMClassifier(random_state=33)
lgbm.fit(X_scaled, y_encoded)
print("✅ Model training complete!")

# --- 6. Save Objects ---
joblib.dump(lgbm, 'lgbm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(type_encoder, 'type_encoder.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

print("\n✅ Model and preprocessing objects saved successfully!")
