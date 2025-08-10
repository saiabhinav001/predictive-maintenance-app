# api/index.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- Load Model and Preprocessing Objects ---
# This logic finds the files whether running locally or on Vercel
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'lgbm_model.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(base_dir, '..', 'scaler.pkl'))
    type_encoder = joblib.load(os.path.join(base_dir, '..', 'type_encoder.pkl'))
    target_encoder = joblib.load(os.path.join(base_dir, '..', 'target_encoder.pkl'))
except Exception as e:
    model, scaler, type_encoder, target_encoder = (None, None, None, None)
    error_message = f"Error loading model files: {e}. Make sure you have run train.py locally and the .pkl files are present."

# --- Prediction Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    if not all([model, scaler, type_encoder, target_encoder]):
        return jsonify({'error': 'Model components not loaded', 'message': error_message}), 500

    try:
        # Get the simple JSON data from the request
        data = request.get_json()
        
        # Create a DataFrame from the input
        input_df = pd.DataFrame([data])

        # --- Preprocess the input data ---
        # 1. Encode the 'Type' feature
        input_df['Type'] = type_encoder.transform(input_df['Type'])

        # 2. Ensure column order matches the training data
        # The 'Target' column was part of training data for the scaler, so we include it here.
        feature_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Target']
        input_df_reordered = input_df[feature_columns]

        # 3. Scale the features
        input_scaled = scaler.transform(input_df_reordered)
        
        # --- Make Prediction ---
        prediction_encoded = model.predict(input_scaled)
        prediction_probabilities = model.predict_proba(input_scaled)
        
        # --- Format the Response ---
        confidence = np.max(prediction_probabilities)
        prediction_label = target_encoder.inverse_transform(prediction_encoded)[0]
        
        return jsonify({
            'prediction': prediction_label,
            'confidence': float(confidence) # Ensure confidence is a standard float
        })

    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 400

# Root route to check if API is running
@app.route('/api', methods=['GET'])
def api_home():
    return "API is running."

# Final check for deployment