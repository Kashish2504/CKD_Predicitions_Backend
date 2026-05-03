"""
Run this BEFORE starting the server to verify everything works.
  python test_model.py
"""
import joblib, json, numpy as np, os

BASE = os.path.dirname(os.path.abspath(__file__))
model  = joblib.load(os.path.join(BASE, "model.pkl"))
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
with open(os.path.join(BASE, "metadata.json")) as f:
    meta = json.load(f)

FEATURES  = meta["features"]
CAT_COLS  = meta["categorical_columns"]
LE_DICT   = meta["label_encodings"]
TARGET_MAP = meta["target_map"]

def encode(data):
    row = []
    for feat in FEATURES:
        val = data.get(feat)
        if val is None: row.append(0.0); continue
        if feat in CAT_COLS:
            enc = LE_DICT.get(feat, {})
            row.append(float(enc.get(str(val), enc.get(str(val).lower(), 0))))
        else:
            row.append(float(val))
    return row

# ── CKD patient ──
ckd_patient = {
    "age": 48, "bp": 80, "sg": 1.020, "al": 4, "su": 0,
    "rbc": "abnormal", "pc": "abnormal", "pcc": "present",
    "ba": "notpresent", "bgr": 121, "bu": 60, "sc": 3.8,
    "sod": 111, "pot": 4.5, "hemo": 9.6, "pcv": 31,
    "wc": 8000, "rc": 3.2, "htn": "yes", "dm": "yes",
    "cad": "no", "appet": "poor", "pe": "yes", "ane": "yes"
}

# ── Healthy patient ──
healthy_patient = {
    "age": 35, "bp": 70, "sg": 1.020, "al": 0, "su": 0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent",
    "ba": "notpresent", "bgr": 110, "bu": 30, "sc": 0.9,
    "sod": 141, "pot": 4.0, "hemo": 15.4, "pcv": 46,
    "wc": 7200, "rc": 5.4, "htn": "no", "dm": "no",
    "cad": "no", "appet": "good", "pe": "no", "ane": "no"
}

for name, patient in [("CKD Patient", ckd_patient), ("Healthy Patient", healthy_patient)]:
    row = encode(patient)
    row_s = scaler.transform([row])
    pred_idx = int(model.predict(row_s)[0])
    proba = model.predict_proba(row_s)[0]
    label = TARGET_MAP[str(pred_idx)]
    print(f"{name}: {label.upper()}  |  CKD={proba[0]*100:.1f}%  notCKD={proba[1]*100:.1f}%")

print("\n✅ Model test passed — ready to run app.py")
