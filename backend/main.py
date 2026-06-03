import io
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# Configuración inicial
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "exported_best_model"

# Variables globales para artefactos
model = None
scaler = None
feature_names = None
feature_means = None

DL_AVAILABLE = False                # Cambiar a true cuando lo tenga
ML_WEIGHT = 1.0                     # Peso de ML mientras no tengo DL
DL_WEIGHT = 0.0                     # Peso de DL hasta que lo tenga

# Orden de variables ML
FEATURE_ORDER = [
  "AGE", "GENDER", "SMOKING", "FINGER_DISCOLORATION", "MENTAL_STRESS",
  "EXPOSURE_TO_POLLUTION", "LONG_TERM_ILLNESS", "ENERGY_LEVEL",
  "IMMUNE_WEAKNESS", "BREATHING_ISSUE", "ALCOHOL_CONSUMPTION",
  "THROAT_DISCOMFORT", "OXYGEN_SATURATION", "CHEST_TIGHTNESS",
  "FAMILY_HISTORY", "SMOKING_FAMILY_HISTORY", "STRESS_IMMUNE"
]

TOP5 = ["SMOKING", "ENERGY_LEVEL", "THROAT_DISCOMFORT", "BREATHING_ISSUE", "OXYGEN_SATURATION"]

INTERACTION_PAIRS = [
  ("SMOKING", "ENERGY_LEVEL"),
  ("SMOKING", "THROAT_DISCOMFORT"),
  ("SMOKING", "BREATHING_ISSUE"),
  ("SMOKING", "OXYGEN_SATURATION"),
  ("ENERGY_LEVEL", "THROAT_DISCOMFORT"),
  ("ENERGY_LEVEL", "BREATHING_ISSUE"),
  ("ENERGY_LEVEL", "OXYGEN_SATURATION"),
  ("THROAT_DISCOMFORT", "BREATHING_ISSUE"),
  ("THROAT_DISCOMFORT", "OXYGEN_SATURATION"),
  ("BREATHING_ISSUE", "OXYGEN_SATURATION"),
]

# Cargar modelo ML, scaler y nombres de features
def load_artifacts():
  global model, scaler, feature_names, feature_means

  try:
    model = joblib.load(MODEL_DIR / "model.joblib")
    logger.info("Modelo ML cargado")
  except FileNotFoundError:
    logger.warning("model.joblib no encontrado. Se usará simulación.")

  try:
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    logger.info("Scaler cargado")
  except FileNotFoundError:
    logger.warning("scaler.joblib no encontrado.")

  try:
    with open(MODEL_DIR / "feature_names.json") as f:
      feature_names = json.load(f)
    logger.info(f"feature_names.json cargado ({len(feature_names)} features).")
  except FileNotFoundError:
    logger.warning("feature_names.json no encontrado. Se asumirán 28 features por defecto.")
    feature_names = TOP5 + [f"{a}_x_{b}" for a, b in INTERACTION_PAIRS] + \
                        [col for col in FEATURE_ORDER if col not in TOP5] + ["DUMMY_PADDING"]
    
  try:
    with open(MODEL_DIR / "feature_means.json") as f:
      feature_means = json.load(f)
    logger.info("feature_means.json cargado (SHAP analítico).")
  except FileNotFoundError:
    logger.warning("feature_means.json no encontrado. No se calcularán valores SHAP.")
    feature_means = None
    

# Construye Dataframe de 28 features a partir de 16 valores crudos
def preprocess_hybrid_input(raw:dict) -> pd.DataFrame:
  df = pd.DataFrame([raw], columns=FEATURE_ORDER)

  # Interacciones de las 5 principales
  for a, b in INTERACTION_PAIRS:
        df[f"{a}_x_{b}"] = df[a] * df[b]

  # Añadir columna dummy de padding
  df["DUMMY_PADDING"] = 0.0

  # Escalar columnas que involucran variables continuas
  if scaler is not None:
      cols_to_scale = [c for c in df.columns if "ENERGY_LEVEL" in c or "OXYGEN_SATURATION" in c]
      df[cols_to_scale] = scaler.transform(df[cols_to_scale])
  else:
      logger.warning("Scaler no disponible, no se aplica normalización.")

  # Reordenar para que coincida con el orden esperado por el modelo
  if feature_names is not None:
      for col in feature_names:
          if col not in df.columns:
              df[col] = 0.0
      df = df[feature_names]
  return df


# Cálculo de SHAP
def compute_shap_values_linear(X_df: pd.DataFrame) -> dict:
    
  if model is None or feature_means is None:
      return None

  coefs = model.coef_[0] 
  features = X_df.columns.tolist()
  x_vals = X_df.iloc[0].values

  mean_vals = np.array([feature_means.get(col, 0.0) for col in features])

  shap_vals = (x_vals - mean_vals) * coefs

  # Convertir a diccionario ordenado por magnitud
  contributions = []
  for i, col in enumerate(features):
      contributions.append({"feature": col, "contribucion": round(shap_vals[i], 4)})

  # Ordenar por contribución absoluta
  contributions.sort(key=lambda x: abs(x["contribucion"]), reverse=True)

  # Separar top positivas y negativas
  top_pos = [c for c in contributions if c["contribucion"] > 0][:5]
  top_neg = [c for c in contributions if c["contribucion"] < 0][:5]

  return {
      "metodo": "SHAP (lineal)",
      "valor_base": 0.0,
      "top_positivas": top_pos,
      "top_negativas": top_neg,
  }



# Predecir DL (mejorar a futuro)
def predict_dl_placeholder(image_bytes: bytes) -> float:
  img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  img = img.resize((224, 224))
  _ = np.array(img) / 255.0
  logger.info("DL placeholder: imagen válida, se devuelve 0.5.")
  gradcam_info = {
     "disponible": False,
     "mensaje": "Mapa de atención no disponible"
  }
  return 0.5, gradcam_info



# App de FastAPI
app = FastAPI(
   title="Diagnóstico Híbrido de Cancer de Pulmón",
   description="API que combina modelo ML y DL",
   version="2.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
   load_artifacts()

# Devuelve estado de modelos y métricas
@app.get("/")
def health_check():
  return {
    "status": "ok",
    "modelo_ml": "cargado" if model is not None else "simulado",
    "modelo_dl": "disponible" if DL_AVAILABLE else "placeholder (fase 2)",
    "shap_disponible": feature_means is not None,
    "metricas": {
       "recall": 0.9042,
       "auc": 0.9217,
       "nota": "Entrenado con 5000 registros"
    }
  }

# Predice diagnóstico
@app.post("/predict")
async def predict(
  clinical_data: str = Form(..., description="JSON con 17 variables clínicas"),
  tomografia: UploadFile = File(..., description="Imagen de tomografía (PNG, JPG, JPEG)")
):
  # Parsear datos clínicos
  try:
      raw = json.loads(clinical_data)
      if set(raw.keys()) != set(FEATURE_ORDER):
          raise ValueError("El JSON no contiene exactamente las 17 variables clínicas esperadas.")
  except Exception as e:
      raise HTTPException(status_code=400, detail=f"Datos clínicos inválidos: {str(e)}")

  # Preprocesar datos clínicos
  try:
      X_processed = preprocess_hybrid_input(raw)
  except Exception as e:
      logger.error(f"Error en preprocesamiento ML: {e}")
      raise HTTPException(status_code=500, detail="Error en el preprocesamiento de los datos clínicos.")

  # Predicción ML
  if model is not None:
      prob_ml = float(model.predict_proba(X_processed)[0, 1])
  else:
      logger.warning("Modelo ML no disponible, usando probabilidad aleatoria simulada.")
      prob_ml = round(random.uniform(0.3, 0.8), 4)
  
  # SHAP
  shap_info = None
  if model is not None and feature_means is not None:
    try:
      shap_info = compute_shap_values_linear(X_processed)
    except Exception as e:
       logger.error(f"Error calculando SHAP: {e}")
  
  # Predicción DL (GRAD-CAM FUTURO)
  try:
      img_bytes = await tomografia.read()
      prob_dl, gradcam_info = predict_dl_placeholder(img_bytes)
  except Exception as e:
      raise HTTPException(status_code=400, detail=f"Error en la imagen: {str(e)}")

  # Combinación
  if DL_AVAILABLE:
      riesgo = ML_WEIGHT * prob_ml + DL_WEIGHT * prob_dl
  else:
      riesgo = prob_ml

  clasificacion = "Alto riesgo" if riesgo >= 0.5 else "Bajo riesgo"

  return {
      "probabilidad_ml": round(prob_ml, 4),
      "probabilidad_dl": round(prob_dl, 4),
      "riesgo_final": round(riesgo, 4),
      "clasificacion": clasificacion,
      "dl_disponible": DL_AVAILABLE,
      "pesos_usados": {"ml": ML_WEIGHT if DL_AVAILABLE else 1.0, "dl": DL_WEIGHT if DL_AVAILABLE else 0.0},
      "shap": shap_info,
      "gradcam": gradcam_info,
  }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)