import streamlit as st
import requests

st.set_page_config(page_title="Diagnóstico Híbrido de Cáncer de Pulmón", layout="wide")

# ---------------------------------------------
# CSS GLOBAL – Tema azul/celeste + blanco
# ---------------------------------------------
st.markdown("""
<style>

/* Fondo */
.stApp {
    background: linear-gradient(135deg, #0f2a44, #1e5f9e);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Header */
.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cfe8ff;
    margin-bottom: 10px;
}

/* Línea */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #4da6ff, transparent);
    margin: 10px 0;
}

/* Inputs */
.stNumberInput input {
    background-color: white;
    color: black;
    border-radius: 8px;
}

/* Radios */
div[role="radiogroup"] {
    background: rgba(255,255,255,0.05);
    padding: 8px;
    border-radius: 10px;
}

/* File uploader FIX */
[data-testid="stFileUploader"] label {
    display: block !important;
    color: white !important;
    font-weight: 600;
    margin-bottom: 10px;
}

[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #4da6ff;
    border-radius: 12px;
    padding: 25px;
    background: rgba(255,255,255,0.05);
}

[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(77,166,255,0.1);
}

/* Botón */
.stButton button {
    background: linear-gradient(90deg, #4da6ff, #1e90ff);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

.stButton button:hover {
    background: linear-gradient(90deg, #1e90ff, #4da6ff);
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    backdrop-filter: blur(6px);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
}

.card-green { border: 2px solid #00c853; }
.card-yellow { border: 2px solid #ffd600; }
.card-red { border: 2px solid #ff5252; }

/* Imágenes */
.explain-img {
    background: white;
    padding: 10px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.markdown('<div class="main-title">Diagnóstico Híbrido de Cáncer de Pulmón</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sistema clínico para la detección temprana de cáncer de pulmón</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------
# FORMULARIO
# ---------------------------------------------
st.header("Datos Clínicos del Paciente")
st.markdown("Ingrese los datos y síntomas del paciente")


c1, c2, c3, c4 = st.columns(4)

with c1:
    age = st.number_input("Edad", 18, 120, 50)
    gender = st.radio("Género", ["Femenino", "Masculino"], horizontal=True)
    gender_val = 0 if gender == "Femenino" else 1
    smoking = st.radio("Fuma", ["No", "Sí"], horizontal=True)
    finger = st.radio("Decoloración de dedos", ["No", "Sí"], horizontal=True)

with c2:
    stress = st.radio("Estrés mental", ["No", "Sí"], horizontal=True)
    pollution = st.radio("Exposición a contaminación", ["No", "Sí"], horizontal=True)
    illness = st.radio("Enfermedad crónica", ["No", "Sí"], horizontal=True)
    energy = st.radio("Bajo nivel de energía", ["No", "Sí"], horizontal=True)#######################################################

with c3:
    immune = st.radio("Debilidad inmunológica", ["No", "Sí"], horizontal=True)
    breathing = st.radio("Dificultad respiratoria", ["No", "Sí"], horizontal=True)
    alcohol = st.radio("Consumo de alcohol", ["No", "Sí"], horizontal=True)
    throat = st.radio("Malestar de garganta", ["No", "Sí"], horizontal=True)

with c4:
    oxygen = st.number_input("Saturación O₂ (%)", 50.0, 100.0, 98.0)
    chest = st.radio("Opresión en el pecho", ["No", "Sí"], horizontal=True)
    family = st.radio("Historial familiar", ["No", "Sí"], horizontal=True)
    fam_smoke = st.radio("Fumador pasivo familiar", ["No", "Sí"], horizontal=True)

# ---------------------------------------------
# 🫁 TOMOGRAFÍA
# ---------------------------------------------
st.markdown("---")
st.header("Tomografía Computarizada")

ct_file = st.file_uploader(
    "Seleccione la imagen de la tomografía",
    type=["png", "jpg", "jpeg"]
)

# ---------------------------------------------
# 🔢 TRANSFORMACIÓN
# ---------------------------------------------
to_int = lambda x: 1 if x == "Sí" else 0

# ---------------------------------------------
# 🚀 BOTÓN
# ---------------------------------------------
st.markdown("---")

if st.button("Analizar Riesgo Híbrido", use_container_width=True):

    if not ct_file:
        st.error("Sube una tomografía primero.")
    else:
        with st.spinner("Analizando con IA..."):

            payload = {
                "age": age,
                "gender": gender_val,
                "smoking": to_int(smoking),
                "finger_discoloration": to_int(finger),
                "mental_stress": to_int(stress),
                "exposure_to_pollution": to_int(pollution),
                "long_term_illness": to_int(illness),
                "energy_level": to_int(energy),
                "immune_weakness": to_int(immune),
                "breathing_issue": to_int(breathing),
                "alcohol_consumption": to_int(alcohol),
                "throat_discomfort": to_int(throat),
                "oxygen_saturation": oxygen,
                "chest_tightness": to_int(chest),
                "family_history": to_int(family),
                "smoking_family_history": to_int(fam_smoke),
            }

            files = {"file": (ct_file.name, ct_file, ct_file.type)}

            try:
                resp = requests.post("http://127.0.0.1:8000/predict", data=payload, files=files)

                if resp.status_code == 200:
                    res = resp.json()

                    st.markdown("---")
                    st.header("Resultados")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>ML Clínico</h3>
                            <div class="metric-value">{res['prob_ml']:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>DL Tomografía</h3>
                            <div class="metric-value">{res['prob_dl']:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    color = f"card-{res['risk_color']}"

                    with col3:
                        st.markdown(f"""
                        <div class="metric-card {color}">
                            <h3>Riesgo Final</h3>
                            <div class="metric-value">{res['prob_final']:.1%}</div>
                            <p><b>{res['risk_level']}</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Explicabilidad
                    st.markdown("---")
                    st.header("🧠 Explicabilidad")

                    e1, e2 = st.columns(2)

                    with e1:
                        st.subheader("SHAP")
                        if res["shap_b64"]:
                            st.markdown(f'<div class="explain-img"><img src="data:image/png;base64,{res["shap_b64"]}" width="100%"></div>', unsafe_allow_html=True)

                    with e2:
                        st.subheader("Grad-CAM")
                        if res["gradcam_b64"]:
                            st.markdown(f'<div class="explain-img"><img src="data:image/png;base64,{res["gradcam_b64"]}" width="100%"></div>', unsafe_allow_html=True)

                else:
                    st.error("Error del backend")

            except:
                st.error("No conecta con FastAPI")