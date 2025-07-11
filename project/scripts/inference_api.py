import io
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "2nd_place_GISLR" / "GISLR_utils"))

from GISLR_utils.models import get_model
from GISLR_utils.keras_models.preprocess import Preprocessing

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and preprocessing config
CFG = {
    'model': 'img_v0',
    'drop_rate': 0.1,
    'deep_supervision': True,
    'encoder': 'convnext_femto',
    'num_classes': 250,
}

MODEL_PATH = "project/2nd_place_GISLR/common_weights.npy"  # Update if using a .pth checkpoint

# Load model
model = get_model(type('cfg', (), CFG))
weights = np.load(MODEL_PATH, allow_pickle=True).item()
model.load_state_dict(weights)
model.eval()

# Preprocessing
preprocessor = Preprocessing()

def preprocess_landmarks(landmarks):
    # landmarks: list of [543, 3] (frames, landmarks, xyz)
    arr = np.array(landmarks, dtype=np.float32)
    arr = torch.tensor(arr).unsqueeze(0)  # [1, frames, 543, 3]
    arr = arr.permute(0, 2, 1, 3)  # [1, 543, frames, 3] if needed
    arr = arr.squeeze(0)  # [543, frames, 3]
    arr = arr.permute(1, 0, 2)  # [frames, 543, 3]
    arr = arr.unsqueeze(0)  # [1, frames, 543, 3]
    arr = arr.float()
    with torch.no_grad():
        processed = preprocessor(arr)
    return processed.unsqueeze(0)  # [1, ...]

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    landmarks = data.get("landmarks")
    if landmarks is None:
        return JSONResponse({"error": "No landmarks provided"}, status_code=400)
    try:
        x = preprocess_landmarks(landmarks)
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred].item()
        return {"class": int(pred), "confidence": float(confidence)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500) 