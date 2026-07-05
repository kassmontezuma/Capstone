import io
import os
import base64
import json
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import shap
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="Lung Cancer Hybrid Diagnostic System")

# ------------------------------------------------------------
# 1. CARGA DEL MODELO ML (joblib)
# ------------------------------------------------------------
ML_DIR = "backend/exported_best_model"
ml_model = joblib.load(os.path.join(ML_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(ML_DIR, "scaler.joblib"))
with open(os.path.join(ML_DIR, "feature_names.json"), "r") as f:
    feature_names = json.load(f)

# ------------------------------------------------------------
# 2. ARQUITECTURA CORRECTA: ResNet18 + LSTM
# ------------------------------------------------------------
class ResNet18LSTM(nn.Module):
    def __init__(self, lstm_hidden=128, num_layers=1, num_classes=1):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # Quitamos AvgPool y FC, dejamos los mapas de características (14x14)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(lstm_hidden, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B,3,H,W) -> features: (B,512, H_f, W_f)
        B, C, H, W = self.features(x).shape
        x = self.features(x).view(B, C, H * W)   # (B, 512, L)
        x = x.permute(0, 2, 1)                   # (B, L, 512)
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                    # (B, hidden)
        out = self.classifier(last_hidden)
        return self.sigmoid(out).squeeze(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dl_model = ResNet18LSTM().to(DEVICE)
DL_PATH = "backend/best_ResNet18+LSTM.pth"
if os.path.exists(DL_PATH):
    dl_model.load_state_dict(torch.load(DL_PATH, map_location=DEVICE))
dl_model.eval()

# ------------------------------------------------------------
# 3. GRAD‑CAM
# ------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if output.dim() == 1:
            loss = output[0]
        else:
            loss = output[0, class_idx] if class_idx is not None else output[0, output.argmax().item()]
        loss.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

target_layer = dl_model.features[-1]   # Última capa de layer4
grad_cam = GradCAM(dl_model, target_layer)

# ------------------------------------------------------------
# 4. UTILIDADES
# ------------------------------------------------------------
def preprocess_ct(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig = np.array(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return orig, tensor

def compute_shap(ml_model, scaler, feature_vector):
    """feature_vector: dict con las 17 variables, en el orden correcto"""
    ordered = ["AGE", "GENDER", "SMOKING", "FINGER_DISCOLORATION", "MENTAL_STRESS",
               "EXPOSURE_TO_POLLUTION", "LONG_TERM_ILLNESS", "ENERGY_LEVEL",
               "IMMUNE_WEAKNESS", "BREATHING_ISSUE", "ALCOHOL_CONSUMPTION",
               "THROAT_DISCOMFORT", "OXYGEN_SATURATION", "CHEST_TIGHTNESS",
               "FAMILY_HISTORY", "SMOKING_FAMILY_HISTORY", "STRESS_IMMUNE"]
    X = np.array([[feature_vector[k] for k in ordered]])
    X_scaled = scaler.transform(X)

    # Intentar TreeExplainer (modelos de árboles)
    try:
        explainer = shap.TreeExplainer(ml_model)
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # clase positiva
    except:
        # Fallback a KernelExplainer
        def predict_fn(x):
            return ml_model.predict_proba(x)[:, 1]
        background = np.random.randn(100, X_scaled.shape[1])
        explainer = shap.KernelExplainer(predict_fn, background, link="logit")
        shap_vals = explainer.shap_values(X_scaled, nsamples=200)

    expected = explainer.expected_value
    if isinstance(expected, list):
        expected = expected[1]

    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(values=shap_vals[0],
                         base_values=expected,
                         data=X_scaled[0],
                         feature_names=ordered),
        max_display=10,
        show=False,
    )
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def compute_gradcam_overlay(original_np, cam):
    cam_resized = cv2.resize(cam, (original_np.shape[1], original_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
    pil_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ------------------------------------------------------------
# 5. ENDPOINT
# ------------------------------------------------------------
@app.post("/predict")
async def predict(
    age: float = Form(...),
    gender: int = Form(...),
    smoking: int = Form(...),
    finger_discoloration: int = Form(...),
    mental_stress: int = Form(...),
    exposure_to_pollution: int = Form(...),
    long_term_illness: int = Form(...),
    energy_level: float = Form(...),
    immune_weakness: int = Form(...),
    breathing_issue: int = Form(...),
    alcohol_consumption: int = Form(...),
    throat_discomfort: int = Form(...),
    oxygen_saturation: float = Form(...),
    chest_tightness: int = Form(...),
    family_history: int = Form(...),
    smoking_family_history: int = Form(...),
    file: UploadFile = File(...),
):
    # Cálculo interno de STRESS_IMMUNE
    stress_immune = 1 if (mental_stress == 1 and immune_weakness == 1) else 0

    features = {
        "AGE": age,
        "GENDER": gender,
        "SMOKING": smoking,
        "FINGER_DISCOLORATION": finger_discoloration,
        "MENTAL_STRESS": mental_stress,
        "EXPOSURE_TO_POLLUTION": exposure_to_pollution,
        "LONG_TERM_ILLNESS": long_term_illness,
        "ENERGY_LEVEL": energy_level,
        "IMMUNE_WEAKNESS": immune_weakness,
        "BREATHING_ISSUE": breathing_issue,
        "ALCOHOL_CONSUMPTION": alcohol_consumption,
        "THROAT_DISCOMFORT": throat_discomfort,
        "OXYGEN_SATURATION": oxygen_saturation,
        "CHEST_TIGHTNESS": chest_tightness,
        "FAMILY_HISTORY": family_history,
        "SMOKING_FAMILY_HISTORY": smoking_family_history,
        "STRESS_IMMUNE": stress_immune,
    }

    # ML
    ordered_keys = list(features.keys())
    X_ml = np.array([[features[k] for k in ordered_keys]])
    X_scaled = scaler.transform(X_ml)
    prob_ml = ml_model.predict_proba(X_scaled)[0, 1]

    # DL
    image_bytes = await file.read()
    original_np, img_tensor = preprocess_ct(image_bytes)
    with torch.no_grad():
        prob_dl = dl_model(img_tensor).item()

    # Fusión
    prob_final = 0.5 * prob_ml + 0.5 * prob_dl
    if prob_final < 0.5:
        risk_level, risk_color = "Bajo Riesgo de cáncer de pulmón. Se recomienda seguimiento rutinario y control preventivo.", "verde"
    elif 0.5 <= prob_final <= 0.7:
        risk_level, risk_color = "Riesgo Moderado de cáncer de pulmón. Se recomienda evaluación médica adicional, pruebas complementarias y seguimiento cercano.", "amarillo"
    else:
        risk_level, risk_color = "Alta probabilidad de cáncer de pulmón. Se recomienda atención atención médica urgente, estudios diagnósticos avanzados y valoración por especialista.", "rojo"

    # Explicabilidad
    shap_b64 = compute_shap(ml_model, scaler, features)
    cam = grad_cam(img_tensor)
    gradcam_b64 = compute_gradcam_overlay(original_np, cam)

    return JSONResponse({
        "prob_ml": float(prob_ml),
        "prob_dl": float(prob_dl),
        "prob_final": float(prob_final),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "shap_b64": shap_b64,
        "gradcam_b64": gradcam_b64,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)