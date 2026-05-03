from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Enable CORS - Specific Configuration
app.add_middleware(
    CORSMiddleware,
    # Explicitly allow your Vite development server
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the model assets
try:
    with open('model/ckd_model.pkl', 'rb') as f:
        assets = pickle.load(f)
        model = assets['model']
        scaler = assets['scaler']
except FileNotFoundError:
    print("Error: Model file not found. Run train_model.py first.")
class CKDInput(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    bgr: float
    sc: float
    sod: float
    pot: float
    hemo: float

@app.get("/health")
def check_health():
    return {"status": "Backend is active"}

@app.post("/predict")
async def predict(data: CKDInput):
    try:
        features = np.array([[data.age, data.bp, data.sg, data.al, data.su, 
                              data.bgr, data.sc, data.sod, data.pot, data.hemo]])
        
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        confidence = round(np.max(probability) * 100, 2)
        
        return {
            "prediction": "CKD" if prediction == 1 else "Not CKD",
            "confidence": f"{confidence}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)