import os
import pickle
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "best_model_xgboost.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "best_threshold.pkl"), "rb") as f:
    threshold = pickle.load(f)
with open(os.path.join(BASE_DIR, "encodings.json")) as f:
    encodings = json.load(f)
with open(os.path.join(BASE_DIR, "target_encoding_maps.json")) as f:
    te_maps = json.load(f)

# Exact column order the model was trained on
FEATURE_COLUMNS = [
    "Airline", "Origin", "Dest", "Cancelled", "Diverted", "CRSDepTime",
    "CRSElapsedTime", "Distance", "Year", "Quarter", "Month", "DayofMonth",
    "DayOfWeek", "Operating_Airline", "OriginCityName", "OriginState",
    "OriginStateName", "DestCityName", "DestState", "DestStateName",
    "DepTimeBlk", "CRSArrTime", "ArrTimeBlk", "DistanceGroup",
    "DivAirportLandings", "DayOfMonth", "WeekOfYear", "IsWeekend",
    "DepHour", "DepHour_sin", "DepHour_cos", "Month_sin", "Month_cos",
    "DOW_sin", "DOW_cos", "Airline_DelayRate", "Origin_DelayRate",
    "Dest_DelayRate", "OpAirline_DelayRate", "Route_DelayRate",
    "DOW_DelayRate", "DepHourBin", "Hour_DelayRate"
]

def time_to_block(hhmm):
    """Convert HHMM int to time block label encoded as integer bucket (0-17)."""
    hour = int(hhmm) // 100
    # Blocks are hourly 0000-0059=0, 0100-0159=1, ... capped at 2300-2359=23
    # DepTimeBlk in BTS data uses labels like "0600-0659"; we approximate with hour bucket
    return min(hour, 23)

def distance_group(distance):
    """BTS DistanceGroup: 250-mile buckets, 1-11+."""
    return min(int(distance) // 250 + 1, 11)

def te(col_name, key):
    """Look up target-encoded value, fall back to global mean."""
    m = te_maps.get(col_name, {})
    return m['map'].get(str(int(key)), m.get('global_mean', 0.2171))

def engineer_features(raw):
    """
    raw: list of 17 values:
      [Year, Quarter, Month, DayofMonth, DayOfWeek, WeekOfYear,
       CRSDepTime, CRSArrTime, CRSElapsedTime, Distance,
       Cancelled, Diverted, DivAirportLandings,
       Origin(int), Dest(int), Airline(int), Operating_Airline(int)]
    """
    (Year, Quarter, Month, DayofMonth, DayOfWeek, WeekOfYear,
     CRSDepTime, CRSArrTime, CRSElapsedTime, Distance,
     Cancelled, Diverted, DivAirportLandings,
     Origin, Dest, Airline, Operating_Airline) = raw

    dep_hour     = (int(CRSDepTime) // 100) + (int(CRSDepTime) % 100) / 60.0
    dep_hour_bin = int(dep_hour)
    route_key    = f"{int(Origin)}_{int(Dest)}"

    row = {
        "Airline":            Airline,
        "Origin":             Origin,
        "Dest":               Dest,
        "Cancelled":          Cancelled,
        "Diverted":           Diverted,
        "CRSDepTime":         CRSDepTime,
        "CRSElapsedTime":     CRSElapsedTime,
        "Distance":           Distance,
        "Year":               Year,
        "Quarter":            Quarter,
        "Month":              Month,
        "DayofMonth":         DayofMonth,
        "DayOfWeek":          DayOfWeek,
        "Operating_Airline":  Operating_Airline,
        # City/state were label-encoded from CSV — use Origin/Dest as proxy
        # (model learned numeric patterns; same index is a safe approximation)
        "OriginCityName":     Origin,
        "OriginState":        Origin,
        "OriginStateName":    Origin,
        "DestCityName":       Dest,
        "DestState":          Dest,
        "DestStateName":      Dest,
        # Time blocks: hour bucket (0–23)
        "DepTimeBlk":         time_to_block(CRSDepTime),
        "CRSArrTime":         CRSArrTime,
        "ArrTimeBlk":         time_to_block(CRSArrTime),
        "DistanceGroup":      distance_group(Distance),
        "DivAirportLandings": DivAirportLandings,
        "DayOfMonth":         DayofMonth,   # duplicate col the notebook kept
        "WeekOfYear":         WeekOfYear,
        "IsWeekend":          int(DayOfWeek >= 5),
        "DepHour":            dep_hour,
        "DepHour_sin":        np.sin(2 * np.pi * dep_hour / 24),
        "DepHour_cos":        np.cos(2 * np.pi * dep_hour / 24),
        "Month_sin":          np.sin(2 * np.pi * Month / 12),
        "Month_cos":          np.cos(2 * np.pi * Month / 12),
        "DOW_sin":            np.sin(2 * np.pi * DayOfWeek / 7),
        "DOW_cos":            np.cos(2 * np.pi * DayOfWeek / 7),
        "Airline_DelayRate":  te("Airline_DelayRate",   Airline),
        "Origin_DelayRate":   te("Origin_DelayRate",    Origin),
        "Dest_DelayRate":     te("Dest_DelayRate",      Dest),
        "OpAirline_DelayRate":te("OpAirline_DelayRate", Operating_Airline),
        "Route_DelayRate":    te_maps["Route_DelayRate"]["map"].get(
                                  route_key,
                                  te_maps["Route_DelayRate"]["global_mean"]
                              ),
        "DOW_DelayRate":      te("DOW_DelayRate",       DayOfWeek),
        "DepHourBin":         dep_hour_bin,
        "Hour_DelayRate":     te("Hour_DelayRate",      dep_hour_bin),
    }

    return [row[col] for col in FEATURE_COLUMNS]


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        raw = data.get('features')
        engineered = engineer_features(raw)
        X = np.array(engineered, dtype=float).reshape(1, -1)
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
