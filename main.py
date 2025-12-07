from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np

# Input schema 
class ASLKeypoints(BaseModel):
    keypoints: conlist(float, min_length=42, max_length=42)


app = FastAPI(
    title="ASL Hand Landmark SVM API",
    description=(
        "Predict ASL characters from 21-hand-keypoints coordinates "
        "(flattened 42-length vector)."
    ),
    version="0.1.0",
)

# Load model at startup
try:
    bundle = joblib.load("models/asl_svm.joblib")
    model = bundle['model']
    class_names = bundle.get("class_names", None)
except Exception as e:
    print("Error loading model:", e)
    model = None
    class_names = None


@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "detail": "Model not found :C"}
    return {"status": "ok"}

@app.post("/predict")
def predict(data: ASLKeypoints):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded:C")

    # Convert list to array shape (1, 42)
    x = np.array(data.keypoints, dtype=float).reshape(1, -1)

    # Predict label
    try:
        probs = model.predict_proba(x)[0]
        pred_label = model.predict(x)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Map classes
    response = {
        "predicted_label": str(pred_label),
        "probabilities": probs.tolist(),
        "classes": getattr(model, "classes_", None).tolist()
        if hasattr(model, "classes_") else class_names
    }
    return response