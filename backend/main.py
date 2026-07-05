import os
import json
import torch
import joblib
import pydicom
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import UploadFile,FastAPI, File


# =========================
# APP
# =========================
app = FastAPI(title="Hybrid Lung Cancer Detection System")

# =========================
# LOAD ARTIFACTS
# =========================
ML_DIR = "backend/exported_best_model"
model = joblib.load(f"{ML_DIR}/model.joblib")
scaler = joblib.load(f"{ML_DIR}/scaler.joblib")

DL_PATH = "backend/exported_best_model/best_ResNet18_LSTM.pth"
dl_model = torch.load(DL_PATH, map_location="cpu")
dl_model.eval()

THRESHOLD = 0.5

# =========================
# PATIENT INPUT
# =========================
class Patient(BaseModel):
    AGE: float
    GENDER: int
    SMOKING: int
    FINGER_DISCOLORATION: int
    MENTAL_STRESS: int
    EXPOSURE_TO_POLLUTION: int
    LONG_TERM_ILLNESS: int
    ENERGY_LEVEL: float
    IMMUNE_WEAKNESS: int
    BREATHING_ISSUE: int
    ALCOHOL_CONSUMPTION: int
    THROAT_DISCOMFORT: int
    OXYGEN_SATURATION: float
    CHEST_TIGHTNESS: int
    FAMILY_HISTORY: int
    SMOKING_FAMILY_HISTORY: int

with open(f"{ML_DIR}/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

with open(f"{ML_DIR}/top5_features.json") as f:
    TOP5 = json.load(f)

with open(f"{ML_DIR}/metadata.json") as f:
    metadata = json.load(f)

# SOLO estas se escalan (igual que training)
CONTINUOUS_COLS = ["ENERGY_LEVEL", "OXYGEN_SATURATION"]

print("Model loaded OK")
print("Threshold:", THRESHOLD)


# =========================
# FEATURE ENGINEERING
# =========================
def build_features(df: pd.DataFrame):

    df = df.copy()

    # Variable derivada utilizada durante el entrenamiento
    df["STRESS_IMMUNE"] = (
        (df["MENTAL_STRESS"] == 1) &
        (df["IMMUNE_WEAKNESS"] == 1)
    ).astype(int)

    # Interacciones del Top 5
    for i in range(len(TOP5)):
        for j in range(i + 1, len(TOP5)):
            a = TOP5[i]
            b = TOP5[j]
            df[f"{a}_x_{b}"] = df[a] * df[b]

    return df

# =========================
# PREPROCESS
# =========================
def preprocess(patient: dict):

    df = pd.DataFrame([patient])

    # mismas interacciones del entrenamiento
    df = build_features(df)

    # agregar columnas faltantes
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    # mismo orden que entrenamiento
    df = df[FEATURE_NAMES]

    # Escalar únicamente las variables continuas
    df[CONTINUOUS_COLS] = scaler.transform(
        df[CONTINUOUS_COLS]
    )

    return df

# =========================
# PREDICT
# =========================
def predict_ml(patient: dict):

    X = preprocess(patient)

    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= THRESHOLD)

    # DEBUG (opcional)
    print("\n====================")
    print("INPUT:", patient)
    print("PROBA:", proba)
    print("PRED:", pred)

    return float(proba), pred





# =========================
# MIL + SLIDING WINDOW
# =========================
def sliding_window(volume, size=64, stride=48):

    D, H, W = volume.shape
    patches = []

    for z in range(0, D - size + 1, stride):
        for y in range(0, H - size + 1, stride):
            for x in range(0, W - size + 1, stride):

                patch = volume[z:z+size, y:y+size, x:x+size]

                if patch.shape == (size, size, size):
                    patches.append(patch)

    return patches

def predict_dl_volume(volume, model):

    model.eval()

    probs = []
    patches = sliding_window(volume)

    with torch.no_grad():

        for p in patches:

            x = torch.tensor(p, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,64,64,64]

            out = model(x)
            prob = torch.sigmoid(out).item()

            probs.append(prob)

    return probs

def mil_pooling(probs, k=5):

    if len(probs) == 0:
        return 0.0

    probs_sorted = sorted(probs, reverse=True)

    topk = probs_sorted[:k]

    return sum(topk) / len(topk)

# =========================
# LATE FUSION
# =========================
def late_fusion(ml_prob, dl_prob, alpha=0.5):

    return alpha * ml_prob + (1 - alpha) * dl_prob

# =========================
# LOAD DICOM
# =========================
def load_dicom_series(folder_path):

    slices = []

    for f in os.listdir(folder_path):
        if f.endswith(".dcm"):
            slices.append(pydicom.dcmread(os.path.join(folder_path, f)))

    slices.sort(key=lambda x: getattr(x, "InstanceNumber", 0))

    volume = np.stack([s.pixel_array for s in slices])

    volume = volume.astype(np.float32)

    return volume

# =========================
# ENDPOINTS
# =========================
@app.post("/predict")
def predict_route(patient: Patient):

    prob, pred = predict_ml(patient.model_dump())

    return {
        "probability": round(prob, 4),
        "prediction": pred,
        "risk": "High" if pred == 1 else "Low",
        "threshold": THRESHOLD
    }

@app.post("/predict-dl")
def predict_dl_ct_endpoint(folder_path: str):

    volume = load_dicom_series(folder_path)

    patch_probs = predict_dl_volume(volume, dl_model)

    patient_prob = mil_pooling(patch_probs)

    return {
        "dl_probability": patient_prob,
        "num_patches": len(patch_probs),
        "risk": "High" if patient_prob > THRESHOLD else "Low"
    }

@app.post("/predict-hybrid")
def predict_hybrid(patient: Patient, folder_path: str):

    ml_prob, _ = predict_ml(patient.model_dump())

    volume = load_dicom_series(folder_path)
    patch_probs = predict_dl_volume(volume, dl_model)
    dl_prob = mil_pooling(patch_probs)

    final = late_fusion(ml_prob, dl_prob)

    return {
        "ml_probability": ml_prob,
        "dl_probability": dl_prob,
        "final_probability": final,
        "risk": "High" if final > THRESHOLD else "Low"
    }


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)