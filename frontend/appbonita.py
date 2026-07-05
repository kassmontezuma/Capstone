import streamlit as st
import requests

st.set_page_config(
    page_title="Diagnóstico de Cáncer de Pulmón",
    layout="wide"
)

# ---------------------------------------------------
# CSS PARA LOGRAR EL DISEÑO LIMPIO Y PROFESIONAL
# ---------------------------------------------------
st.markdown("""
<style>
/* Fondo general de la app (Gris muy claro/azulino médico) */
.stApp {
    background-color: #f4f7f9;
    color: #1e293b;
}

/* Títulos y subtítulos principales */
.main-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    color: #0f172a;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #475569;
    margin-bottom: 25px;
}

/* Estilo para las secciones/tarjetas */
div[data-testid="stVerticalBlock"] > div {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 5px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}

/* Ajustar los bordes de los contenedores internos */
.custom-card {
    background: #ffffff;
    padding: 20px 25px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Modificar los Radio Buttons nativos de Streamlit */
div[data-testid="stRadio"] label {
    font-weight: 500;
    color: #334155 !important;
}

/* Botón Analizar Riesgo */
div.stButton > button {
    background-color: #2b7a78 !important;
    color: white !important;
    border: none;
    padding: 12px 24px;
    font-size: 18px;
    font-weight: 600;
    border-radius: 30px;
    box-shadow: 0 4px 10px rgba(43, 122, 120, 0.3);
    transition: all 0.3s ease;
    display: block;
    margin: 0 auto;
}

div.stButton > button:hover {
    background-color: #17252a !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(43, 122, 120, 0.4);
}

/* Inputs numéricos y sliders */
div[data-testid="stNumberInput"] input {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    color: #0f172a;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="main-title">Diagnóstico de Cáncer de Pulmón</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modelo de Machine Learning de Predicción Basado en Datos Clínicos</div>', unsafe_allow_html=True)

st.markdown('<h3 style="color:#0f172a; margin-bottom:20px;">Datos del paciente</h3>', unsafe_allow_html=True)

# ---------------------------------------------------
# FORMULARIO ORGANIZADO EN COLUMNAS (Estilo Bloques)
# ---------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    # Bloque: Información Básica
    st.markdown('<div class="custom-card"><div class="card-title">👤 Información Básica</div>', unsafe_allow_html=True)
    age = st.number_input("Edad", 18, 120, 50)
    gender = st.radio("Género", ["Femenino", "Masculino"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bloque: Hábitos de Vida
    st.markdown('<div class="custom-card"><div class="card-title">🚬 Hábitos de Vida</div>', unsafe_allow_html=True)
    smoking = st.radio("Fuma", ["No", "Sí"], horizontal=True)
    alcohol = st.radio("Consumo de alcohol", ["No", "Sí"], horizontal=True)
    fam_smoke = st.radio("Fumador pasivo familiar", ["No", "Sí"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    # Bloque: Síntomas y Signos
    st.markdown('<div class="custom-card"><div class="card-title">🫁 Síntomas y Signos</div>', unsafe_allow_html=True)
    breathing = st.radio("Dificultad respiratoria", ["No", "Sí"], horizontal=True)
    chest = st.radio("Opresión en el pecho", ["No", "Sí"], horizontal=True)
    throat = st.radio("Malestar de garganta", ["No", "Sí"], horizontal=True)
    finger = st.radio("Decoloración de dedos", ["No", "Sí"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bloque: Exposición
    st.markdown('<div class="custom-card"><div class="card-title">🌍 Exposición</div>', unsafe_allow_html=True)
    pollution = st.radio("Exposición a contaminación", ["No", "Sí"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    # Bloque: Mediciones
    st.markdown('<div class="custom-card"><div class="card-title">📊 Mediciones</div>', unsafe_allow_html=True)
    energy = st.slider("Nivel de energía", 0.0, 100.0, 75.0, format="%.2f%%")
    oxygen = st.slider("Saturación O₂", 50.0, 100.0, 98.0, format="%.2f%%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bloque: Antecedentes y Salud
    st.markdown('<div class="custom-card"><div class="card-title">🏥 Antecedentes y Salud</div>', unsafe_allow_html=True)
    illness = st.radio("Enfermedad crónica", ["No", "Sí"], horizontal=True)
    family = st.radio("Historial familiar", ["No", "Sí"], horizontal=True)
    stress = st.radio("Estrés mental", ["No", "Sí"], horizontal=True)
    immune = st.radio("Debilidad inmunológica", ["No", "Sí"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# CONVERSIÓN SÍ/NO A BINARIO
# ---------------------------------------------------
to_int = lambda x: 1 if x == "Sí" else 0
gender_val = 1 if gender == "Masculino" else 0

# ---------------------------------------------------
# BOTÓN DE ACCIÓN CENTRADO
# ---------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("📊 Analizar Riesgo", use_container_width=False):

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

    with st.spinner("Analizando datos clínicos..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )

            if response.status_code == 200:
                res = response.json()
                
                st.markdown("---")
                st.markdown('<h3 style="color:#0f172a;">Resultados del Análisis</h3>', unsafe_allow_html=True)
                
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.metric(
                        label="Probabilidad calculada",
                        value=f"{res['probability']:.2%}"
                    )

                with rc2:
                    if res["prediction"] == 1:
                        st.error("⚠️ Alto riesgo detectado. Se sugiere revisión médica avanzada.")
                    else:
                        st.success("✅ Bajo riesgo detectado. Continúe con sus chequeos regulares.")
            else:
                st.error("❌ Error en el servidor de predicción.")
        except Exception as e:
            st.error("🔌 No se pudo establecer conexión con el backend de FastAPI.")
            st.caption(f"Detalle del error: {e}")