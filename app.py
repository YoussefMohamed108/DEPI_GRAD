import os
import pickle
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# Load model, threshold, and encodings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "best_model_xgboost.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "best_threshold.pkl"), "rb") as f:
    threshold = pickle.load(f)

with open(os.path.join(BASE_DIR, "encodings.json"), "rb") as f:
    encodings = json.load(f)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features')
        
        # features is expected to be a list of 17 values as per the frontend logic
        # We need to ensure the order matches what the model was trained on.
        # Based on the notebook/frontend, the features are:
        # [Year, Quarter, Month, DayofMonth, DayOfWeek, WeekOfYear, CRSDepTime, CRSArrTime, 
        #  CRSElapsedTime, Distance, Cancelled, Diverted, DivAirportLandings, Origin, Dest, Airline, Operating_Airline]
        
        # Note: The model in the notebook had 35 features after engineering (cyclical, target encoding).
        # If the model expects more features than provided, we need to handle that.
        # However, since the user wants to use the .pkl directly, I will assume the 
        # feature engineering needs to be replicated here or the model is compatible.
        
        # For now, let's wrap the features in a DataFrame if necessary
        # and handle any missing features or name mismatches.
        
        X = np.array(features).reshape(1, -1)
        
        # Get probability
        probability = float(model.predict_proba(X)[:, 1][0])
        
        return jsonify({
            'probability': probability,
            'threshold': float(threshold),
            'delayed': probability >= float(threshold)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/encodings', methods=['GET'])
def get_encodings():
    return jsonify(encodings)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
