# train.py
# Run this script once locally to generate the model files.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
import joblib

print("Starting model training process...")

# --- 1. Load the Dataset ---
df = pd.read_csv('predictive_maintenance.csv')

# --- 2. Data Preprocessing ---
df = df.drop(['UDI', 'Product ID'], axis=1)

# Define feature columns and target column
feature_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target_column = 'Failure Type'

# Separate features (X) and the target variable (y)
X = df[feature_columns + ['Target']] # Include 'Target' for now, we'll handle it
y = df[target_column]

# --- 3. Encode Categorical Features and Target ---
type_encoder = LabelEncoder()
target_encoder = LabelEncoder()

# Fit and transform the 'Type' column in features
X['Type'] = type_encoder.fit_transform(X['Type'])

# Fit and transform the 'Failure Type' column for the target
y_encoded = target_encoder.fit_transform(y)

# The 'Target' column is already 0 or 1, so no encoding needed.
# Now that we have encoded, let's ensure X only has the features for scaling and training
X_final_features = X[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Target']]


# --- 4. Feature Scaling ---
# We scale all the final features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final_features)
print("Features scaled.")

# --- 5. Train the LGBM Classifier Model ---
print("Training the LGBM Classifier...")
lgbm = LGBMClassifier(random_state=33)
lgbm.fit(X_scaled, y_encoded)
print("✅ Model training complete!")

# --- 6. Save the Model and Preprocessing Objects ---
joblib.dump(lgbm, 'lgbm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(type_encoder, 'type_encoder.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

print("\n✅ Model and preprocessing objects saved successfully!")
print("Make sure to deploy these .pkl files with your application.")
