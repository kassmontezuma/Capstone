# Sistema Híbrido de Diagnóstico de Cáncer de Pulmón

## Requisitos

- Python 3.9+
- Carpeta `backend/exported_best_model/` y archivo `best_ResNet18+LSTM.pth` con los artefactos del modelo ML.

## Instalación

```bash
pip install -r requirements.txt
frontend:
  cd frontend
  streamlit run app.py
backend:
  cd backend
  uvicorn main:app --reload
```

Crear entornos virtuales

# Backend

cd backend
python -m venv venv
source venv/bin/activate # En Windows: venv\Scripts\activate

# Frontend (otra terminal)

cd ../frontend
python -m venv venv
source venv/bin/activate
