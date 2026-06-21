from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pathlib import Path

app = Flask(**name**)

BASE_DIR = Path(**file**).resolve().parent.parent

with open(BASE_DIR / "best_model_xgboost.pkl", "rb") as f:
model = pickle.load(f)

with open(BASE_DIR / "best_threshold.pkl", "rb") as f:
threshold = pickle.load(f)

FEATURE_COLUMNS = [
"Year",
"Quarter",
"Month",
"DayofMonth",
"DayOfWeek",
"WeekOfYear",
"CRSDepTime",
"CRSArrTime",
"CRSElapsedTime",
"Distance",
"Cancelled",
"Diverted",
"DivAirportLandings",
"Origin",
"Dest",
"Airline",
"Operating_Airline"
]

@app.route("/api/predict", methods=["POST"])
def predict():
try:
data = request.get_json()

```
    df = pd.DataFrame([data])

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[FEATURE_COLUMNS]

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
        "delayed": delayed,
        "probability": round(probability * 100, 2),
        "confidence": confidence,
        "threshold": round(threshold * 100, 2)
    })

except Exception as e:
    return jsonify({"error": str(e)}), 500
```
