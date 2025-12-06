from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np

# Input schema 
class ASLKeypoints(BaseModel):
    keyoints: conlist(float, min_items=42, max_items=42)

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