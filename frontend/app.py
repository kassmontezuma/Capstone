import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
from PIL import Image 
from io import BytesIO

st.set_page_config(
    page_title="Lung Cancer Diagnosis",
    layout="wide"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">', unsafe_allow_html=True)
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg, #f0f4f8, #E5F6FF);
    color:#00234B;
    font-size: 18px;
}

.main-title{
    font-size:46px;
    font-weight:800;
    text-align:center;
    margin-bottom: 5px;
}

.subtitle{
    font-size: 18px;
    text-align:center;
    color:#48658;
    font-weight: 500;
}
            
.stWidget label, [data-testid="stWidgetLabel"] p {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #00234B !important;
    line-height: 1.4 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] [data-testid="stMarkdownContainer"] p {
    font-size: 20px !important;
    font-weight: 500 !important;
    color: #00234B !important;
}
            
.metric-card{
    background:rgba(255,255,255,0.08);
    padding:25px;
    border-radius:15px;
    text-align:center;
}

.metric-value{
    font-size:34px;
    font-weight:bold;
}
            
.stNumberInput input {
    font-size: 20px !important;
    background-color: white !important;
    border: 2px solid #bcccdc !important;
    border-radius: 8px !important;
}
            
.stNumberInput button {
    color: white !important;
    background-color: #00234B !important;
}
            
.stButton > button {
    background:#00234B !important;
    color:white !important;
    height:65px !important;
    border-radius:10px !important;
}

.stButton > button div[data-testid="stMarkdownContainer"] {
    font-size:28px !important; 
    font-weight:800 !important;
    margin:0 !important;
}
            
div[data-testid="stFileUploader"] button {
    background:#00234B !important;
    color:white !important;
    font-weight:bold !important;
    border-radius:10px !important;
    height:50px !important;
    align-items:center !important;
    justify-content:center !important;
}
            
div[data-testid="stNumberInput"] {
    max-width: 150px;  
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">Diagnóstico de Cáncer de Pulmón</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Hybrid system based on predictive models for the early detection of lung cancer</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# FORMULARIO
# ---------------------------------------------------

col_formulario, col_tomografia = st.columns([2.2, 1.2], gap="large")

with col_formulario:
    st.subheader("Patient details")
    
    c1, c2, c3 = st.columns(3)

    # Columna 1
    with c1:
        age = st.number_input("Edad", 18, 120, 50)
        gender = st.radio("Género", ["Femenino", "Masculino"], horizontal=True)
        smoking = st.radio("Fuma", ["No", "Sí"], horizontal=True)
        finger = st.radio("Decoloración de dedos", ["No", "Sí"], horizontal=True)
        oxygen = st.number_input("Saturación O₂", 50.0, 100.0, 98.0)

    # Columna 2
    with c2:
        stress = st.radio("Estrés mental", ["No", "Sí"], horizontal=True)
        pollution = st.radio("Exposición a contaminación", ["No", "Sí"], horizontal=True)
        illness = st.radio("Enfermedad crónica", ["No", "Sí"], horizontal=True)
        energy = st.number_input("Nivel de energía", 0.0, 100.0, 75.0)
        chest = st.radio("Opresión en el pecho", ["No", "Sí"], horizontal=True)

    # Columna 3
    with c3:
        immune = st.radio("Debilidad inmunológica", ["No", "Sí"], horizontal=True)
        breathing = st.radio("Dificultad respiratoria", ["No", "Sí"], horizontal=True)
        alcohol = st.radio("Consumo de alcohol", ["No", "Sí"], horizontal=True)
        throat = st.radio("Malestar de garganta", ["No", "Sí"], horizontal=True)
        family = st.radio("Historial familiar", ["No", "Sí"], horizontal=True)
        fam_smoke = st.radio("Fumador pasivo familiar", ["No", "Sí"], horizontal=True)

# ---------------------------------------------------
# Conversión Sí/No
# ---------------------------------------------------

to_int = lambda x: 1 if x == "Sí" else 0


with col_tomografia:
    st.subheader("Load CT Scan")
    uploaded_file = st.file_uploader(
        "Label oculto", 
        type=["zip"], 
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        st.success(f"¡Cargado: {uploaded_file.name}!")
    
    if st.button("Analizar Riesgo", width="stretch"):
        if uploaded_file is None:
            st.warning("Debe subir un archivo ZIP con la serie DICOM.")
            st.stop()

        payload = {

            "AGE": age,
            "GENDER": 1 if gender == "Masculino" else 0,
            "SMOKING": to_int(smoking),
            "FINGER_DISCOLORATION": to_int(finger),
            "MENTAL_STRESS": to_int(stress),
            "EXPOSURE_TO_POLLUTION": to_int(pollution),
            "LONG_TERM_ILLNESS": to_int(illness),
            "ENERGY_LEVEL": energy,
            "IMMUNE_WEAKNESS": to_int(immune),
            "BREATHING_ISSUE": to_int(breathing),
            "ALCOHOL_CONSUMPTION": to_int(alcohol),
            "THROAT_DISCOMFORT": to_int(throat),
            "OXYGEN_SATURATION": oxygen,
            "CHEST_TIGHTNESS": to_int(chest),
            "FAMILY_HISTORY": to_int(family),
            "SMOKING_FAMILY_HISTORY": to_int(fam_smoke)

        }

        with st.spinner("Analizando..."):

            try:

                files = {
                    "dicom_zip": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/zip"
                    )
                }

                response = requests.post(
                    "https://mi-api-cancer.onrender.com/predict",
                    data=payload,
                    files=files
                )
                print(response.status_code)
                print(response.text)

                if response.status_code == 200:

                    res = response.json()
                    if "error" in res:
                        st.error(res["error"])
                        st.stop()

                    st.session_state["resultado"] = res

                else:
                    st.error(f"Error {response.status_code}")

                    try:
                        st.json(response.json())
                    except:
                        st.write(response.text)

            except Exception as e:

                st.error("No se pudo conectar con FastAPI.")
                st.write(e)

if "resultado" in st.session_state:
        st.markdown("---")
        st.markdown("""
        <h1 style='
            text-align: center;
            font-size:42px;
            font-weight:800;
            color:#00234B;
        '>
            Analysis results
        </h1>
        """, unsafe_allow_html=True)

        res = st.session_state["resultado"]

        if "error" in res:
            st.error("Error en FastAPI:")
            st.json(res)
            st.stop()

        st.markdown("### Probabilidades")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "ML",
            f"{res['ml_probability']:.2%}"
        )

        c2.metric(
            "DL",
            f"{res['dl_probability']:.2%}"
        )

        c3.metric(
            "Hybrid",
            f"{res['final_probability']:.2%}"
        )

        # ==================================================
        # Estado del paciente
        # ==================================================

        st.subheader("Diagnóstico")

        if "risk_level" not in res:
            st.error("Error en el análisis:")
            st.json(res)
            st.stop()

        risk = res["risk_level"]


        if risk == "Alto riesgo":

            st.error(
                "Alto riesgo de cáncer de pulmón"
            )

        elif risk == "Riesgo moderado":

            st.warning(
                "Riesgo moderado de cáncer de pulmón"
            )

        else:

            st.success(
                "Bajo riesgo de cáncer de pulmón"
            )

        # ==================================================
        # SHAP
        # ==================================================

        st.subheader("Model Explanation (SHAP)")

        df_shap = pd.DataFrame(res["shap"])

        df_shap["abs_shap"] = df_shap["shap"].abs()

        df_shap = (
            df_shap
            .sort_values("abs_shap", ascending=False)
        )

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        colors = [
            "#D62728" if x > 0 else "#1F77B4"
            for x in df_shap["shap"]
        ]

        ax.barh(
            df_shap["feature"],
            df_shap["shap"],
            color=colors
        )

        ax.invert_yaxis()

        ax.axvline(
            0,
            color="black",
            linewidth=1
        )

        ax.set_xlabel("SHAP Value")

        ax.set_ylabel("Variables")

        ax.set_title(
            "Contribución de Variables al Diagnóstico"
        )

        plt.tight_layout()

        st.pyplot(fig)

        # ==================================================
        # Tabla SHAP
        # ==================================================

        st.subheader(
            "Variable details"
        )

        df_table = df_shap.drop(columns="abs_shap").copy()

        df_table["value"] = df_table["value"].round(2)

        df_table["shap"] = df_table["shap"].round(3)

        st.dataframe(
            df_table.rename(
                columns={
                    "feature":"Variable",
                    "value":"Valor",
                    "shap":"Impacto SHAP"
                }
            ),
            width="stretch"
        )

        # ==================================================
        # Deep Learning
        # ==================================================

        st.subheader(
            "CT Detections"
        )

        st.metric(
            "Regiones sospechosas",
            len(res["detections"])
        )

        if len(res["detections"]) > 0:

            detections_df = pd.DataFrame(
                res["detections"]
            )

            st.dataframe(
                detections_df,
                width="stretch"
            )
    
        res = st.session_state["resultado"]
        detections = res.get("detections", [])

        detections_df = pd.json_normalize(res["detections"])
        cols = st.columns(3)
        for i, det in enumerate(detections):
            img = Image.open(
                BytesIO(
                    base64.b64decode(det["gradcam"])
                )
            )

            with cols[i % 2]:
                st.image(
                    img,
                    caption=(
                        f"P={det['probability']:.2%}\n"
                        f"Slice={det['tomography_coordinates']['slice']}\n"
                        f"X={det['tomography_coordinates']['x']} "
                        f"Y={det['tomography_coordinates']['y']}"
                    )
                )