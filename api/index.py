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

# Load encodings
with open(BASE_DIR / "encodings.json", "r") as f:
    ENCODINGS = json.load(f)

FEATURE_COLUMNS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "WeekOfYear", "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
    "Distance", "Cancelled", "Diverted", "DivAirportLandings",
    "Origin", "Dest", "Airline", "Operating_Airline"
]

CATEGORICAL_COLS = ["Origin", "Dest", "Airline", "Operating_Airline"]

@app.route("/", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        features = []
        for col in FEATURE_COLUMNS:
            val = data.get(col)
            if val is None:
                return jsonify({"error": f"Missing required field: {col}"}), 400

            # Encode categorical features
            if col in CATEGORICAL_COLS:
                mapping = ENCODINGS.get(col, {})
                if str(val) not in mapping:
                    valid = list(mapping.keys())[:5]
                    return jsonify({
                        "error": f"Unknown {col}: '{val}'. Valid examples: {valid}..."
                    }), 400
                val = mapping[str(val)]

            try:
                features.append(float(val))
            except (ValueError, TypeError):
                return jsonify({"error": f"Field '{col}' must be numeric, got: {val}"}), 400

        input_array = np.array([features], dtype=np.float32)
        outputs = session.run(None, {input_name: input_array})
        probability = float(outputs[1][0][1])

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
