# src/api.py
# fastapi server
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import joblib

app = FastAPI(title="AI Ticket Triage v1.0")

# Load on startup
model = joblib.load("models/ticket_classifier.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

class TicketIn(BaseModel):
    document: str  # Single ticket text field

class TicketPrediction(BaseModel):
    predicted_type: str
    confidence: float
    all_probabilities: Dict[str, float]

@app.post("/predict", response_model=TicketPrediction)
def predict(ticket: TicketIn):
    proba = model.predict_proba([ticket.document])[0]
    pred_idx = proba.argmax()
    
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])
    
    probabilities = {
        label_encoder.classes_[i]: float(proba[i])
        for i in range(len(label_encoder.classes_))
    }
    
    return TicketPrediction(
        predicted_type=pred_label,
        confidence=confidence,
        all_probabilities=probabilities
    )

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
