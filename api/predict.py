from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load model and threshold
with open(BASE_DIR / "best_model_xgboost.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "best_threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

FEATURE_COLUMNS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "WeekOfYear", "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
    "Distance", "Cancelled", "Diverted", "DivAirportLandings",
    "Origin", "Dest", "Airline", "Operating_Airline"
]

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        df = pd.DataFrame([data])

        # Add missing columns with None
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df = df[FEATURE_COLUMNS]

        # WARNING: If your model expects encoded categorical values,
        # you MUST encode 'Origin', 'Dest', 'Airline', 'Operating_Airline'
        # here before prediction. Raw strings will likely cause errors.

        probability = float(model.predict_proba(df)[0][1])
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
        # In production, use proper logging instead of print
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)