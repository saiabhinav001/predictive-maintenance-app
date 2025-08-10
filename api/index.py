# api/index.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Load the trained model and preprocessing objects ---
# This logic handles finding the files whether running locally or on Vercel
try:
    # When deployed on Vercel, files are in the root of the serverless function
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, '..', 'lgbm_model.pkl'))
    scaler = joblib.load(os.path.join(base_dir, '..', 'scaler.pkl'))
    type_encoder = joblib.load(os.path.join(base_dir, '..', 'type_encoder.pkl'))
    target_encoder = joblib.load(os.path.join(base_dir, '..', 'target_encoder.pkl'))
except FileNotFoundError:
    # If the above fails, it might be because we are running locally.
    # This provides a fallback path.
    model = joblib.load('lgbm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    type_encoder = joblib.load('type_encoder.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
except Exception as e:
    # If loading fails for any other reason, we set them to None to handle the error gracefully
    model, scaler, type_encoder, target_encoder = (None, None, None, None)
    error_message = f"Error loading model files: {e}"


# Define the prediction endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': error_message, 'message': 'Model files not found or failed to load.'}), 500

    try:
        # --- Parse the incoming JSON data from your frontend ---
        # Your frontend sends a nested structure: {"input_data": [{"fields": [...], "values": [[...]]}]}
        data = request.get_json()
        
        # Extract the field names and the single row of values
        fields = data['input_data'][0]['fields']
        values = data['input_data'][0]['values'][0]

        # Create a pandas DataFrame from the received data
        input_df = pd.DataFrame([values], columns=fields)

        # --- Preprocess the input data ---
        # 1. Encode the 'Type' feature using the loaded encoder
        input_df['Type'] = type_encoder.transform(input_df['Type'])

        # 2. Ensure the column order is exactly as the model was trained on
        # The 'Target' column is not needed for prediction, so we exclude it.
        feature_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        input_df_reordered = input_df[feature_columns]

        # 3. Scale the features using the loaded scaler
        input_scaled = scaler.transform(input_df_reordered)
        
        # --- Make a prediction and get probabilities ---
        prediction_encoded = model.predict(input_scaled)
        prediction_probabilities = model.predict_proba(input_scaled)

        # --- Format the response for your frontend ---
        # 1. Get the confidence score (the probability of the predicted class)
        confidence = np.max(prediction_probabilities)

        # 2. Decode the prediction to the original string label (e.g., "No Failure")
        prediction_label = target_encoder.inverse_transform(prediction_encoded)[0]
        
        # Return the prediction and confidence as a JSON response
        return jsonify({
            'prediction': prediction_label,
            'confidence': confidence
        })

    except Exception as e:
        # Return a detailed error if anything goes wrong
        return jsonify({'error': str(e), 'message': 'An error occurred during prediction.'}), 400

# A simple root route to check if the API is running
@app.route('/api', methods=['GET'])
def api_home():
    return "API for MachineInsight Pro is running."

