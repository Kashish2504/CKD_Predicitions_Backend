# CKD Prediction Backend — Setup Guide

## Folder Structure
```
ckd_backend/
├── app.py           ← Flask API
├── model.pkl        ← Trained Random Forest model
├── scaler.pkl       ← StandardScaler
├── metadata.json    ← Feature encodings & model info
├── requirements.txt
├── test_model.py    ← Quick sanity check (run before server)
└── README.md
```

---

## Step-by-Step Setup

### Step 1 — Place files in your backend folder
Copy all files from this folder into your existing backend project root.

---

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

---

### Step 3 — Test the model (no server needed)
```bash
python test_model.py
```
Expected output:
```
CKD Patient:     ckd    | CKD=99.x%  notCKD=0.x%
Healthy Patient: notckd | CKD=0.x%   notCKD=99.x%
✅ Model test passed — ready to run app.py
```

---

### Step 4 — Start the server
**Development:**
```bash
python app.py
```
Server runs on `http://localhost:5000`

**Production:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## API Endpoints

### GET `/`
Health check.
```json
{ "status": "ok", "model": "Random Forest", "accuracy": 1.0 }
```

### POST `/predict`
Main prediction endpoint. Send all 24 features as JSON.

**Request body:**
```json
{
  "age": 48,
  "bp": 80,
  "sg": 1.020,
  "al": 4,
  "su": 0,
  "rbc": "abnormal",
  "pc": "abnormal",
  "pcc": "present",
  "ba": "notpresent",
  "bgr": 121,
  "bu": 60,
  "sc": 3.8,
  "sod": 111,
  "pot": 4.5,
  "hemo": 9.6,
  "pcv": 31,
  "wc": 8000,
  "rc": 3.2,
  "htn": "yes",
  "dm": "yes",
  "cad": "no",
  "appet": "poor",
  "pe": "yes",
  "ane": "yes"
}
```

**Response:**
```json
{
  "prediction": "ckd",
  "is_ckd": true,
  "confidence": 99.0,
  "probabilities": { "ckd": 99.0, "notckd": 1.0 },
  "model": "Random Forest"
}
```

### GET `/features`
Returns feature list, categorical columns, and valid values for each.

### GET `/model-info`
Returns model name, accuracy, and top important features.

---

## Features Reference

| Feature | Type       | Description                    | Valid Values / Range       |
|---------|------------|--------------------------------|----------------------------|
| age     | Numerical  | Age in years                   | 2–90                       |
| bp      | Numerical  | Blood pressure (mm/Hg)         | 50–180                     |
| sg      | Numerical  | Specific gravity               | 1.005, 1.010, 1.015, 1.020, 1.025 |
| al      | Numerical  | Albumin                        | 0–5                        |
| su      | Numerical  | Sugar                          | 0–5                        |
| rbc     | Categorical| Red blood cells                | normal, abnormal           |
| pc      | Categorical| Pus cell                       | normal, abnormal           |
| pcc     | Categorical| Pus cell clumps                | present, notpresent        |
| ba      | Categorical| Bacteria                       | present, notpresent        |
| bgr     | Numerical  | Blood glucose random (mgs/dl)  | 70–500                     |
| bu      | Numerical  | Blood urea (mgs/dl)            | 1.5–300                    |
| sc      | Numerical  | Serum creatinine (mgs/dl)      | 0.4–76                     |
| sod     | Numerical  | Sodium (mEq/L)                 | 104–163                    |
| pot     | Numerical  | Potassium (mEq/L)              | 2.5–47                     |
| hemo    | Numerical  | Hemoglobin (gms)               | 3.1–17.8                   |
| pcv     | Numerical  | Packed cell volume             | 9–54                       |
| wc      | Numerical  | White blood cell count         | 2200–26400                 |
| rc      | Numerical  | Red blood cell count (millions)| 2.1–8.0                    |
| htn     | Categorical| Hypertension                   | yes, no                    |
| dm      | Categorical| Diabetes mellitus              | yes, no                    |
| cad     | Categorical| Coronary artery disease        | yes, no                    |
| appet   | Categorical| Appetite                       | good, poor                 |
| pe      | Categorical| Pedal edema                    | yes, no                    |
| ane     | Categorical| Anemia                         | yes, no                    |

---

## Connecting to Your Frontend

In your frontend, call the predict endpoint like this:

```javascript
const response = await fetch("http://localhost:5000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(formData)   // formData = your 24-feature object
});
const result = await response.json();
// result.prediction → "ckd" or "notckd"
// result.confidence → 0–100
// result.probabilities → { ckd: xx, notckd: xx }
```
