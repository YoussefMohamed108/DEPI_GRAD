from flask import Flask, request, jsonify
import numpy as np
import json
from pathlib import Path
import onnxruntime as ort

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load ONNX model
session = ort.InferenceSession(str(BASE_DIR / "model.onnx"))
input_name = session.get_inputs()[0].name

# Load threshold
with open(BASE_DIR / "threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

FEATURE_COLUMNS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "WeekOfYear", "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
    "Distance", "Cancelled", "Diverted", "DivAirportLandings",
    "Origin", "Dest", "Airline", "Operating_Airline"
]

# IMPORTANT: You need the same encoding mappings used during training.
# If you used Label Encoding or One-Hot Encoding, load those encoders
# or recreate the mapping logic here. For now, this assumes numerical inputs.
# If your model expects encoded categoricals, you'll need to add that logic.

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Build feature array in correct order
        features = []
        for col in FEATURE_COLUMNS:
            val = data.get(col)
            if val is None:
                return jsonify({"error": f"Missing required field: {col}"}), 400
            # Convert categorical strings to numeric if needed
            # You'll need to add your encoding logic here
            features.append(float(val))

        # Reshape for ONNX: [batch_size, n_features]
        input_array = np.array([features], dtype=np.float32)

        # Run inference
        outputs = session.run(None, {input_name: input_array})
        probability = float(outputs[1][0][1])  # probability of class 1 (delayed)

        delayed = probability >= threshold
        confidence_score = abs(probability - threshold)

        if confidence_score >= 0.30:
            confidence = "High"
        elif confidence_score >= 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"

        return jsonify({
            "delayed": bool(delayed),
            "probability": round(probability * 100, 2),
            "confidence": confidence,
            "threshold": round(threshold * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
