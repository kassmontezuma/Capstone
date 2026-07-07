import os
import cv2
import json
import joblib
import pydicom
import numpy as np
import pandas as pd
from fastapi import FastAPI,UploadFile, File, Form

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms

import tempfile
import zipfile
import shutil
import shap
import base64

# APP
app = FastAPI(title="Hybrid Lung Cancer Detection System")

# LOAD ARTIFACTS
ML_DIR = "exported_best_model"
model = joblib.load(f"{ML_DIR}/model.joblib")
scaler = joblib.load(f"{ML_DIR}/scaler.joblib")
# Extraer el CatBoost del pipeline
cat_model = model.named_steps["model"]
# SHAP trabaja sobre el modelo, no sobre el Pipeline
explainer = shap.TreeExplainer(cat_model)

THRESHOLD = 0.5

# LOAD DL MODEL
DL_PATH = "exported_best_model/best_ResNet18_LSTM.pth"
DEVICE = torch.device("cpu")

#ARQUITECTURA RESNET18+LSTM
class ResNet18LSTM(nn.Module):

    def __init__(self,
                 num_classes=1,
                 lstm_hidden=256,
                 num_layers=2,
                 dropout=0.2):

        super().__init__()

        self.resnet = models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden,64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):

        batch, seq, c, h, w = x.shape

        x = x.view(batch*seq,c,h,w)

        features = self.resnet(x)

        features = features.view(batch,seq,-1)

        lstm_out,_ = self.lstm(features)

        last = lstm_out[:,-1,:]

        out = self.fc(last)

        return out.squeeze(1)

#GRAD CAM
class GradCAMResNetLSTM:

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = self.model.resnet.layer4[-1]
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, output, seq_len=16):
        self.model.zero_grad()
        output.backward(retain_graph=True)
        activations = self.activations.detach()
        gradients = self.gradients.detach()
        cams = []

        for i in range(activations.shape[0]):
            weights = gradients[i].mean(dim=(1,2))

            cam = torch.sum(
                weights[:,None,None] * activations[i],
                dim=0
            )

            cam = torch.relu(cam)
            cam -= cam.min()
            cam /= cam.max() + 1e-8

            cam = cv2.resize(
                cam.cpu().numpy(),
                (224,224)
            )

            cams.append(cam)

        # reorganizar como secuencia
        cams = np.array(cams).reshape(-1, seq_len, 224, 224)
        cams = cams[0]
        importance = [cam.max() for cam in cams]
        best_slice = int(np.argmax(importance))

        return cams, best_slice
    
#CARGAR MODELO DL
dl_model = ResNet18LSTM()

dl_model.load_state_dict(
    torch.load(DL_PATH, map_location=DEVICE)
)

dl_model.eval()
gradcam = GradCAMResNetLSTM(dl_model)

# LOAD DICOM SERIES
def load_dicom_series(folder_path: str):

    slices = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".dcm"):
            filepath = os.path.join(folder_path, filename)
            ds = pydicom.dcmread(filepath)
            slices.append(ds)

    if len(slices) == 0:
        raise ValueError("No se encontraron archivos DICOM.")

    slices.sort(
        key=lambda s: int(getattr(s, "InstanceNumber", 0))
    )

    volume = []

    for ds in slices:
        image = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept
        volume.append(image)
    volume = np.stack(volume, axis=0)

    return volume

#PREPROCESSING(IGUAL QUE ENTRENAMIENTO)
def preprocess_volume(volume):
    volume = np.clip(volume,-1000,400)
    volume = (volume + 1000) / 1400
    return volume.astype(np.float32)


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229,0.224,0.225]
)

#EXTRACCIÓN DE PATCHE
def extract_patch(volume, center, patch_size=64):
    z, y, x = center
    half = patch_size // 2
    
    z_min = max(0, z - half)
    z_max = min(volume.shape[0], z + half)
    y_min = max(0, y - half)
    y_max = min(volume.shape[1], y + half)
    x_min = max(0, x - half)
    x_max = min(volume.shape[2], x + half)
    
    patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    if patch.shape != (patch_size, patch_size, patch_size):
        temp = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
        temp[
            (half - (z - z_min)):(half - (z - z_min) + patch.shape[0]),
            (half - (y - y_min)):(half - (y - y_min) + patch.shape[1]),
            (half - (x - x_min)):(half - (x - x_min) + patch.shape[2])
        ] = patch
        patch = temp
        
    return patch
#SECUENCIA DE PATCHES
def patch_to_sequence(patch, seq_len=16):
    total_slices = patch.shape[0]
    z_center = total_slices // 2
    half_seq = seq_len // 2
    start_idx = max(0, z_center - half_seq)
    end_idx = min(total_slices, z_center + half_seq)

    indices = np.linspace(
        start_idx,
        end_idx - 1,
        seq_len
    ).astype(int)

    sequence = []
    original_images = []

    for z in indices:
        axial = patch[z]
        axial = (axial * 255).astype(np.uint8)

        axial = cv2.resize(
            axial,
            (224,224),
            interpolation=cv2.INTER_LINEAR
        )

        axial = axial.astype(np.float32) / 255.0
        original_images.append(axial.copy())
        tensor = torch.from_numpy(axial).float().unsqueeze(0)
        tensor = tensor.repeat(3, 1, 1)
        tensor = normalize(tensor)
        sequence.append(tensor)

    sequence = torch.stack(sequence)
    return sequence.unsqueeze(0), original_images

#SUPERPONER GRAD-CAM
def overlay_gradcam(image, cam):

    image = (image * 255).astype(np.uint8)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay

with open(f"{ML_DIR}/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

with open(f"{ML_DIR}/top5_features.json") as f:
    TOP5 = json.load(f)

with open(f"{ML_DIR}/metadata.json") as f:
    metadata = json.load(f)

# SOLO estas se escalan (igual que training)
CONTINUOUS_COLS = ["ENERGY_LEVEL", "OXYGEN_SATURATION"]

# FEATURE ENGINEERING
def build_features(df: pd.DataFrame):

    df = df.copy()

    df["STRESS_IMMUNE"] = (
        (df["MENTAL_STRESS"] == 1) &
        (df["IMMUNE_WEAKNESS"] == 1)
    ).astype(int)

    for i in range(len(TOP5)):
        for j in range(i + 1, len(TOP5)):
            a = TOP5[i]
            b = TOP5[j]
            df[f"{a}_x_{b}"] = df[a] * df[b]

    return df

# PREPROCESS ML
def preprocess_ml(patient: dict):

    df = pd.DataFrame([patient])
    df = build_features(df)

    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_NAMES]

    df[CONTINUOUS_COLS] = scaler.transform(
        df[CONTINUOUS_COLS]
    )

    return df

# PREDICT ML
def predict_ml(patient: dict):

    X = preprocess_ml(patient)

    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= THRESHOLD)

    return float(proba), pred
#SHAP
def explain_ml(patient):
    X = preprocess_ml(patient)
    shap_values = explainer(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = shap_values.values[0]
    explanation = []

    for feature, value in zip(X.columns, values):
        explanation.append({
            "feature": feature,
            "shap": float(value),
            "value": float(X.iloc[0][feature])
        })

    explanation = sorted(
        explanation,
        key=lambda x: abs(x["shap"]),
        reverse=True
    )

    return explanation[:10]

#CONVERTIR A BASE 64
def image_to_base64(image):

    _, buffer = cv2.imencode(".png", image)

    return base64.b64encode(
        buffer
    ).decode("utf-8")

#PREDICT DL
def predict_dl(folder_path: str):
    volume = load_dicom_series(folder_path)
    volume = preprocess_volume(volume)

    detections = []
    max_prob = 0.0
    contador = 0
    #aca iba -32, 16/ -32 32 / -32 32
    for z in range(32, volume.shape[0] - 32, 32):
        for y in range(32, volume.shape[1] - 64, 64):
            for x in range(32, volume.shape[2] - 64, 64):
                contador += 1
                if contador % 50 == 0:
                    print(f"Procesando patch {contador}")
                patch = extract_patch(
                    volume,
                    (z, y, x),
                    64
                )

                sequence, original_images = patch_to_sequence(patch)
                sequence = sequence.to(DEVICE)

                with torch.no_grad():
                    output = dl_model(sequence)

                prob = torch.sigmoid(output).item()

                if prob >= 0.30:
                    sequence.requires_grad_(True)
                    output = dl_model(sequence)
                    cams, best_slice = gradcam.generate(output, seq_len=sequence.shape[1])

                    overlay = overlay_gradcam(
                        original_images[best_slice],
                        cams[best_slice]
                    )

                    overlay_b64 = image_to_base64(overlay)

                    detections.append({
                        "z": z,
                        "y": y,
                        "x": x,
                        "prob": float(prob),
                        "gradcam": overlay_b64
                    })

                if prob > max_prob:
                    max_prob = prob

    return max_prob, detections

# LATE FUSION
def late_fusion(ml_prob, dl_prob, alpha=0.5):
    return alpha * ml_prob + (1 - alpha) * dl_prob


# ENDPOINTS
@app.post("/predict")
async def predict_route(
    AGE: float = Form(...),
    GENDER: int = Form(...),
    SMOKING: int = Form(...),
    FINGER_DISCOLORATION: int = Form(...),
    MENTAL_STRESS: int = Form(...),
    EXPOSURE_TO_POLLUTION: int = Form(...),
    LONG_TERM_ILLNESS: int = Form(...),
    ENERGY_LEVEL: float = Form(...),
    IMMUNE_WEAKNESS: int = Form(...),
    BREATHING_ISSUE: int = Form(...),
    ALCOHOL_CONSUMPTION: int = Form(...),
    THROAT_DISCOMFORT: int = Form(...),
    OXYGEN_SATURATION: float = Form(...),
    CHEST_TIGHTNESS: int = Form(...),
    FAMILY_HISTORY: int = Form(...),
    SMOKING_FAMILY_HISTORY: int = Form(...),

    dicom_zip: UploadFile = File(...)
):

    patient = {
        "AGE": AGE,
        "GENDER": GENDER,
        "SMOKING": SMOKING,
        "FINGER_DISCOLORATION": FINGER_DISCOLORATION,
        "MENTAL_STRESS": MENTAL_STRESS,
        "EXPOSURE_TO_POLLUTION": EXPOSURE_TO_POLLUTION,
        "LONG_TERM_ILLNESS": LONG_TERM_ILLNESS,
        "ENERGY_LEVEL": ENERGY_LEVEL,
        "IMMUNE_WEAKNESS": IMMUNE_WEAKNESS,
        "BREATHING_ISSUE": BREATHING_ISSUE,
        "ALCOHOL_CONSUMPTION": ALCOHOL_CONSUMPTION,
        "THROAT_DISCOMFORT": THROAT_DISCOMFORT,
        "OXYGEN_SATURATION": OXYGEN_SATURATION,
        "CHEST_TIGHTNESS": CHEST_TIGHTNESS,
        "FAMILY_HISTORY": FAMILY_HISTORY,
        "SMOKING_FAMILY_HISTORY": SMOKING_FAMILY_HISTORY
    }

    ml_prob, _ = predict_ml(patient)
    shap_explanation = explain_ml(patient)
    temp_dir = tempfile.mkdtemp()

    try:
        if not dicom_zip.filename.lower().endswith(".zip"):
                return {"error": "Debe subir un archivo ZIP"}

        zip_path = os.path.join(temp_dir, dicom_zip.filename)

        with open(zip_path, "wb") as f:
            shutil.copyfileobj(dicom_zip.file, f)

        extract_dir = os.path.join(temp_dir, "dicoms")
        os.makedirs(extract_dir)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        dicom_folder = None

        for root, dirs, files in os.walk(extract_dir):
            if any(file.lower().endswith(".dcm") for file in files):
                dicom_folder = root
                break

        if dicom_folder is None:
            return {"error": "No se encontraron archivos DICOM"}
        
        dl_prob, detections = predict_dl(dicom_folder)

        final_prob = late_fusion(ml_prob, dl_prob)
        prediction = int(final_prob >= THRESHOLD)

        return {
            "ml_probability": round(ml_prob,4),
            "dl_probability": round(dl_prob,4),
            "final_probability": round(final_prob,4),
            "prediction": prediction,
            "detections": detections,
            "shap": shap_explanation
        }

    except zipfile.BadZipFile:
        return {"error": "ZIP inválido"}

    except Exception as e:
        return {"error": str(e)}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)