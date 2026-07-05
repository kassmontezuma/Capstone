import streamlit as st
import requests

st.set_page_config(
    page_title="Diagnóstico de Cáncer de Pulmón",
    layout="wide"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------

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
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #00234B !important;
    line-height: 1.4 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] [data-testid="stMarkdownContainer"] p {
    font-size: 14px !important;
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
    font-size: 18px !important;
    background-color: white !important;
    border: 2px solid #bcccdc !important;
    border-radius: 8px !important;
}
            
.stNumberInput button {
    color: white !important;
    background-color: #00234B !important;
}

.upload-container {
    background-color: white;
    padding: 30px 20px;
    border-radius: 12px;
    border: 2px dashed #00234B;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}
                          
.stButton button{
    background:#00234B;
    color:white;
    height:50px;
    border-radius:10px;
    font-size:20px;
    font-weight:800px;
}

/* Estilo para el ícono de la nube */
.upload-icon {
    font-size: 55px;
    color: #a0aec0;
    margin-bottom: 10px;
    display: block;
    text-align: center;
}

.upload-text-main {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #486581 !important;
}

[data-testid="stFileUploader"] {
    background-color: white !important;
    padding: 0px 20px 20px 20px !important;
    border-bottom-left-radius: 12px !important;
    border-bottom-right-radius: 12px !important;
    border: 2px dashed #00234B !important;
    border-top: none !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background-color: #00234B !important;
    color: white !important;
    border-radius: 8px !important;
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
    '<div class="subtitle">Modelo de Machine Learning basado en datos clínicos</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# FORMULARIO
# ---------------------------------------------------

col_formulario, col_tomografia = st.columns([2.2, 1.2], gap="large")

with col_formulario:
    st.subheader("Datos del paciente")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        age = st.number_input("Edad", 18, 120, 50)
        gender = st.radio("Género", ["Femenino", "Masculino"], horizontal=True)
        smoking = st.radio("Fuma", ["No", "Sí"], horizontal=True)
        finger = st.radio("Decoloración de dedos", ["No", "Sí"], horizontal=True)

    with c2:
        stress = st.radio("Estrés mental", ["No", "Sí"], horizontal=True)
        pollution = st.radio("Exposición a contaminación", ["No", "Sí"], horizontal=True)
        illness = st.radio("Enfermedad crónica", ["No", "Sí"], horizontal=True)
        energy = st.number_input("Nivel de energía", 0.0, 100.0, 75.0)

    with c3:
        immune = st.radio("Debilidad inmunológica", ["No", "Sí"], horizontal=True)
        breathing = st.radio("Dificultad respiratoria", ["No", "Sí"], horizontal=True)
        alcohol = st.radio("Consumo de alcohol", ["No", "Sí"], horizontal=True)
        throat = st.radio("Malestar de garganta", ["No", "Sí"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_bottom1, c_bottom2, c_bottom3 = st.columns(3)
    with c_bottom1:
        oxygen = st.number_input("Saturación O₂", 50.0, 100.0, 98.0)
    with c_bottom2:
        chest = st.radio("Opresión en el pecho", ["No", "Sí"], horizontal=True)
    with c_bottom3:
        family = st.radio("Historial familiar", ["No", "Sí"], horizontal=True)
        fam_smoke = st.radio("Fumador pasivo familiar", ["No", "Sí"], horizontal=True)

with col_tomografia:
    st.subheader("Subir Archivos")
    
    st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">', unsafe_allow_html=True)
    
    # Cabecera visual del contenedor
    st.markdown("""
        <div style="background-color: white; padding: 30px 20px 0px 20px; border-top-left-radius: 12px; border-top-right-radius: 12px; border: 2px dashed #00234B; border-bottom: none; text-align: center;">
            <i class="bi bi-cloud-upload upload-icon"></i>
            <div class="upload-text-main">Drag and drop files here</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Cargador nativo integrado
    uploaded_file = st.file_uploader(
        "Label oculto", 
        type=["dcm", "zip"], 
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.success(f"¡Cargado: {uploaded_file.name}!")

# ---------------------------------------------------
# Conversión Sí/No
# ---------------------------------------------------

to_int = lambda x: 1 if x == "Sí" else 0

gender_val = 1 if gender == "Masculino" else 0

# ---------------------------------------------------
# BOTÓN
# ---------------------------------------------------

st.markdown("---")

if st.button("Analizar Riesgo", use_container_width=True):

    payload = {

        "AGE": age,
        "GENDER": gender_val,
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

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )

            if response.status_code == 200:

                res = response.json()

                st.markdown("---")

                st.header("Resultado")

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Probabilidad",
                        f"{res['probability']:.2%}"
                    )

                with c2:

                    if res["prediction"] == 1:

                        st.error("Alto riesgo de cáncer de pulmón")

                    else:

                        st.success("Bajo riesgo de cáncer de pulmón")

            else:

                st.error("Error del servidor.")

        except Exception as e:

            st.error("No se pudo conectar con FastAPI.")
            st.write(e)