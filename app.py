from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow all origins — restrict in production

# ── Load artifacts ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model   = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler  = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

with open(os.path.join(BASE_DIR, "metadata.json")) as f:
    meta = json.load(f)

FEATURES      = meta["features"]          # ordered list of 24 feature names
CAT_COLS      = meta["categorical_columns"]
LE_DICT       = meta["label_encodings"]   # {"rbc": {"abnormal":0,"normal":1}, ...}
TARGET_MAP    = meta["target_map"]        # {"0":"ckd","1":"notckd"}


# ── Helper ───────────────────────────────────────────────────────────────────
def encode_input(data: dict) -> list:
    """
    Convert raw JSON payload into an ordered feature vector.
    Categorical values are label-encoded using the saved mappings.
    Missing values default to 0 (you can improve this with imputation).
    """
    row = []
    for feat in FEATURES:
        val = data.get(feat)
        if val is None or val == "":
            row.append(0.0)
            continue

        if feat in CAT_COLS:
            encoding = LE_DICT.get(feat, {})
            # Try exact match first, then lower-cased
            encoded = encoding.get(str(val), encoding.get(str(val).lower(), 0))
            row.append(float(encoded))
        else:
            try:
                row.append(float(val))
            except (ValueError, TypeError):
                row.append(0.0)

    return row


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": meta["model_name"],
        "accuracy": meta["accuracy"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body with 24 CKD features.
    Returns prediction + probability.

    Example request body:
    {
        "age": 48, "bp": 80, "sg": 1.020, "al": 1, "su": 0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent",
        "ba": "notpresent", "bgr": 121, "bu": 36, "sc": 1.2,
        "sod": 135, "pot": 4.5, "hemo": 15.4, "pcv": 44,
        "wc": 7800, "rc": 5.2, "htn": "yes", "dm": "no",
        "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Encode & scale
        row   = encode_input(data)
        row_s = scaler.transform([row])

        # Predict
        pred_idx  = int(model.predict(row_s)[0])
        proba     = model.predict_proba(row_s)[0].tolist()
        label     = TARGET_MAP[str(pred_idx)]

        # Build response
        ckd_proba    = proba[0] if TARGET_MAP["0"] == "ckd" else proba[1]
        notckd_proba = 1 - ckd_proba

        return jsonify({
            "prediction":    label,                        # "ckd" or "notckd"
            "is_ckd":        label == "ckd",
            "confidence":    round(max(proba) * 100, 2),   # % confidence in chosen class
            "probabilities": {
                "ckd":    round(ckd_proba * 100, 2),
                "notckd": round(notckd_proba * 100, 2)
            },
            "model": meta["model_name"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/features", methods=["GET"])
def features():
    """Returns expected features and their encoding info — handy for frontend validation."""
    return jsonify({
        "features":             FEATURES,
        "categorical_columns":  CAT_COLS,
        "label_encodings":      LE_DICT,
        "numerical_columns":    meta["numerical_columns"]
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """Returns model metadata."""
    return jsonify({
        "model_name": meta["model_name"],
        "accuracy":   meta["accuracy"],
        "features":   FEATURES,
        "top_features": [
            "sc (Serum Creatinine)",
            "pcv (Packed Cell Volume)",
            "hemo (Hemoglobin)",
            "rc (Red Blood Cell Count)",
            "appet (Appetite)"
        ]
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting CKD Backend on port {port}")
    print(f"Model: {meta['model_name']} | Accuracy: {meta['accuracy']}")
    app.run(host="0.0.0.0", port=port, debug=False)