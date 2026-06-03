import json
import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "http://localhost:8000/predict"
BACKEND_STATUS_URL = "http://localhost:8000/"

# Configuración de la página
st.set_page_config(page_title="Diagnóstico Híbrido - Cáncer de Pulmón", page_icon="🫁", layout="wide")


# Estilo CSS
st.markdown("""
<style>
.section-header {
    background: linear-gradient(90deg, #1a5276, #2980b9);
    color: white; padding: 0.5rem 1rem; border-radius: 6px;
    margin-bottom: 1rem; font-weight: 600;
}
.result-card {
    background: white; border-radius: 12px; padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); text-align: center;
}
.badge {
    background: #f0f0f0; border-radius: 4px; padding: 0.1rem 0.4rem;
    font-size: 0.85rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=80)
    st.title("Sistema Híbrido")

    st.markdown("---")
    st.markdown("""
    ** Cómo usar **
    1. Complete los datos clínicos.
    2. Cargue la tomografía.
    3. Presione **Predicir Riesgo**.
    """)
    st.markdown("---")
    st.caption("Herramienta de apoyo clínico. No sustituye el diagnóstico médico.")


# Título principal
st.title("Diagnóstico Híbrido de Cáncer de Pulmón")
st.markdown("Complete los datos del paciente y cargue la tomografía para estimar el riesgo.")


# Formulario de datos clínicos
st.markdown('<div class="section-header">Variables Clínicas</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Datos generales**")
    age = st.number_input("Edad", 0, 120, 55, help="Edad en años")
    gender = st.radio("Género", ["M", "F"], horizontal=True, index=0)
    smoking = st.radio("Tabaquismo", options=[0, 1], format_func=lambda x: "Sí" if x else "No",
                       horizontal=True, help="¿El paciente fuma actualmente?")
    smoking_family = st.radio("Familiares fumadores", [0, 1], format_func=lambda x: "Sí" if x else "No",
                              horizontal=True)
    family_history = st.radio("Antecedentes familiares de cáncer", [0, 1], format_func=lambda x: "Sí" if x else "No",
                              horizontal=True)
    alcohol = st.radio("Consumo de alcohol", [0, 1], format_func=lambda x: "Sí" if x else "No",
                       horizontal=True)

with col2:
    st.markdown("**Síntomas respiratorios**")
    breathing = st.radio("Problemas respiratorios", [0, 1], format_func=lambda x: "Sí" if x else "No",
                         horizontal=True, help="Dificultad para respirar en reposo o esfuerzo")
    throat = st.radio("Molestia en la garganta", [0, 1], format_func=lambda x: "Sí" if x else "No",
                      horizontal=True)
    chest = st.radio("Opresión en el pecho", [0, 1], format_func=lambda x: "Sí" if x else "No",
                     horizontal=True)
    finger = st.radio("Decoloración de dedos", [0, 1], format_func=lambda x: "Sí" if x else "No",
                      horizontal=True, help="¿Presenta cianosis o cambios de color?")
    oxygen = st.number_input("Saturación O₂ (%)", 50.0, 100.0, 98.0, 0.1,
                             help="Saturación de oxígeno medida con oxímetro")
    energy = st.number_input("Nivel de energía (0-10)", 0.0, 10.0, 6.0, 0.1,
                             help="Escala subjetiva: 0 = muy bajo, 10 = muy alto")

with col3:
    st.markdown("**Estado general y riesgos**")
    stress = st.radio("Estrés mental", [0, 1], format_func=lambda x: "Sí" if x else "No",
                      horizontal=True)
    immune = st.radio("Debilidad del sistema inmune", [0, 1], format_func=lambda x: "Sí" if x else "No",
                      horizontal=True)
    chronic = st.radio("Enfermedad de larga duración", [0, 1], format_func=lambda x: "Sí" if x else "No",
                       horizontal=True)
    pollution = st.radio("Exposición a contaminación", [0, 1], format_func=lambda x: "Sí" if x else "No",
                         horizontal=True)


# Carga de imagen
st.markdown('<div class="section-header">Tomografía</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Cargar tomografía (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"],
                                 help="Asegúrese de cargar una tomografía de tórax.")
if uploaded_file:
    st.image(uploaded_file, caption="Vista previa", width=300)


# Botón de predicción
predict_btn = st.button("Predecir Riesgo", type="primary", disabled=(uploaded_file is None))
if not uploaded_file:
    st.caption("Cargue una tomografía para habilitar la predicción.")


# Llamar backend y visualizar resultados
if predict_btn and uploaded_file:
    
  # Calcular STRESS_INMUNE automáticamente
  stress_immune_calc = 1 if (stress == 1 and immune == 1) else 0

  # Construir JSON clínico
  clinical_dict = {
    "AGE": age, "GENDER": gender, "SMOKING": smoking,
    "FINGER_DISCOLORATION": finger, "MENTAL_STRESS": stress,
    "EXPOSURE_TO_POLLUTION": pollution, "LONG_TERM_ILLNESS": chronic,
    "ENERGY_LEVEL": energy, "IMMUNE_WEAKNESS": immune,
    "BREATHING_ISSUE": breathing, "ALCOHOL_CONSUMPTION": alcohol,
    "THROAT_DISCOMFORT": throat, "OXYGEN_SATURATION": oxygen,
    "CHEST_TIGHTNESS": chest, "FAMILY_HISTORY": family_history,
    "SMOKING_FAMILY_HISTORY": smoking_family, "STRESS_IMMUNE": stress_immune_calc
  }

  with st.spinner("Calculando riesgo y generando explicación..."):
    try:
        files = {"tomografia": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"clinical_data": json.dumps(clinical_dict)}
        resp = requests.post(BACKEND_URL, files=files, data=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        st.stop()

  # Mostrar resultado
  riesgo = result["riesgo_final"]
  clasif = result["clasificacion"]

  # Tarjeta de resultado
  st.markdown("##Resultado del Diagnóstico")
  col_center = st.columns([1,2,1])
  with col_center[1]:
    # Semáforo
    if riesgo >= 0.7:
        color, icono = "#e74c3c", "🔴"
    elif riesgo >= 0.5:
        color, icono = "#e67e22", "🟠"
    elif riesgo >= 0.3:
        color, icono = "#f1c40f", "🟡"
    else:
        color, icono = "#27ae60", "🟢"

    st.markdown(f"<div class='result-card'>"
                f"<div style='font-size:4rem'>{icono}</div>"
                f"<div style='font-size:2.5rem; font-weight:700; color:{color}'>{riesgo*100:.1f}%</div>"
                f"<div style='font-size:1.3rem; font-weight:600; color:{color}'>{clasif}</div>"
                f"</div>", unsafe_allow_html=True)

  st.progress(float(riesgo))

  # Desglose por modelo
  st.subheader("Desglose por modelo")
  col1, col2 = st.columns(2)
  with col1:
      st.metric("Modelo ML", f"{result['probabilidad_ml']*100:.1f}%")
      st.progress(float(result["probabilidad_ml"]))
  with col2:
      dl_badge = " (placeholder)" if not result["dl_disponible"] else ""
      st.metric(f"Modelo DL{dl_badge}", f"{result['probabilidad_dl']*100:.1f}%")
      st.progress(float(result["probabilidad_dl"]))
      if not result["dl_disponible"]:
          st.caption("El valor del DL es simulado; será reemplazado por la CNN real.")

  # Explicación SHAP
  shap_data = result.get("shap")
  if shap_data:
    st.subheader("Explicación del modelo clínico (SHAP)")
    st.markdown("**Factores que más influyen en el riesgo** (contribuciones en log-odds)")

    tab1, tab2 = st.tabs(["Aumentan riesgo", "Disminuyen riesgo"])

    with tab1:
      if shap_data["top_positivas"]:
        df_pos = pd.DataFrame(shap_data["top_positivas"])
        st.bar_chart(df_pos.set_index("feature")["contribucion"], color="#e74c3c")
      else:
        st.info("No hay factores con contribución positiva significativa.")

    with tab2:
      if shap_data["top_negativas"]:
        df_neg = pd.DataFrame(shap_data["top_negativas"])
        # Invertir signo para que se vea como barras hacia abajo (positivo en gráfica = reduce riesgo)
        df_neg["reduccion"] = -df_neg["contribucion"]
        st.bar_chart(df_neg.set_index("feature")["reduccion"], color="#27ae60")
      else:
        st.info("No hay factores con contribución negativa significativa.")

    with st.expander("Interpretación"):
      st.markdown("""
      - Las barras **rojas** indican variables que aumentan la probabilidad de cáncer.
      - Las barras **verdes** indican variables que la reducen.
      - La magnitud refleja la importancia relativa en esta predicción concreta.
      """)
  else:
      st.info("La explicación SHAP no está disponible (falta el archivo de medias de entrenamiento).")

  # Explicación GRAD-CAM
  gradcam = result.get("gradcam", {})
  if gradcam:
     st.subheader("Mapa de atención sobre la tomografía")
     if gradcam.get("disponible"):
        # Aquí se mostraría la imagen con overlay en la fase 2
        st.image(gradcam.get("imagen_base64"), caption="Zonas de interés de la CNN")
     else:
        st.info(gradcam.get("mensaje", "Mapa de atención no disponible"))

  # Recomendación clínica
  if riesgo >= 0.7:
      st.error("Riesgo alto: Se recomienda derivación urgente a oncología torácica y biopsia confirmatoria.")
  elif riesgo >= 0.5:
      st.warning("Riesgo moderado: Seguimiento estrecho, repetir tomografía en 3 meses y evaluación por especialistas.")
  elif riesgo >= 0.3:
      st.info("Riesgo bajo-moderado: Control anual y manejo de factores de riesgo modificables.")
  else:
      st.success("Riesgo bajo: Continuar con seguimiento de rutina según protocolo estándar.")
  with st.expander("Respuesta completa del servidor"):
      st.json(result)