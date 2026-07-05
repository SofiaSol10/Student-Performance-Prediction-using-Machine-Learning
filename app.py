import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de GPA | Student Performance",
    page_icon="🎓",
    layout="centered",
)

# ------------------------------------------------------------------
# Carga del modelo (cacheada para no recargar en cada interacción)
# ------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load("notebook/modelo_regresion_lineal.pkl")

modelo = cargar_modelo()

# Orden EXACTO de columnas usado en el entrenamiento (X = df.drop(["StudentID","GradeClass","GPA"]))
FEATURE_ORDER = [
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("🎓 Predicción de GPA con Regresión Lineal")
st.markdown(
    """
Esta aplicación estima el **GPA (Grade Point Average)** de un estudiante a partir de
características académicas, familiares y de estilo de vida, usando un modelo de
**Regresión Lineal** entrenado sobre el dataset
[Student Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset).

**Métricas del modelo (conjunto de prueba):**
"""
)

col1, col2, col3 = st.columns(3)
col1.metric("R²", "0.953")
col2.metric("MAE", "0.155")
col3.metric("RMSE", "0.197")

st.info(
    "ℹ️ Se eliminó la variable `GradeClass` del entrenamiento porque generaba "
    "*data leakage* (se calcula directamente a partir del GPA).",
    icon="ℹ️",
)

st.divider()

# ------------------------------------------------------------------
# Formulario de entrada
# ------------------------------------------------------------------
st.subheader("📋 Datos del estudiante")

with st.form("formulario_prediccion"):

    st.markdown("**Datos demográficos**")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Edad", min_value=15, max_value=18, value=16, step=1)
        gender = st.selectbox("Género", options=["Masculino", "Femenino"])
    with c2:
        ethnicity = st.selectbox(
            "Etnia",
            options=["Caucásico", "Afroamericano", "Asiático", "Otro"],
        )
        parental_education = st.selectbox(
            "Nivel educativo de los padres",
            options=[
                "Ninguno",
                "Secundaria",
                "Educación superior incompleta",
                "Licenciatura/Pregrado",
                "Posgrado (Maestría/Doctorado)",
            ],
        )

    st.markdown("**Hábitos de estudio y asistencia**")
    c3, c4 = st.columns(2)
    with c3:
        study_time = st.slider(
            "Horas de estudio semanales", min_value=0.0, max_value=20.0, value=10.0, step=0.5
        )
    with c4:
        absences = st.slider("Ausencias (en el año)", min_value=0, max_value=30, value=5, step=1)

    tutoring = st.radio("¿Recibe clases de apoyo (tutoring)?", options=["No", "Sí"], horizontal=True)
    parental_support = st.selectbox(
        "Nivel de apoyo parental",
        options=["Ninguno", "Bajo", "Moderado", "Alto", "Muy alto"],
        index=2,
    )

    st.markdown("**Actividades extracurriculares**")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        extracurricular = st.checkbox("Extracurriculares")
    with c6:
        sports = st.checkbox("Deportes")
    with c7:
        music = st.checkbox("Música")
    with c8:
        volunteering = st.checkbox("Voluntariado")

    submitted = st.form_submit_button("🔮 Predecir GPA", use_container_width=True)

# ------------------------------------------------------------------
# Mapeos de texto -> código numérico (igual que el dataset original)
# ------------------------------------------------------------------
map_gender = {"Masculino": 0, "Femenino": 1}
map_ethnicity = {"Caucásico": 0, "Afroamericano": 1, "Asiático": 2, "Otro": 3}
map_parental_education = {
    "Ninguno": 0,
    "Secundaria": 1,
    "Educación superior incompleta": 2,
    "Licenciatura/Pregrado": 3,
    "Posgrado (Maestría/Doctorado)": 4,
}
map_yes_no = {"No": 0, "Sí": 1}
map_parental_support = {"Ninguno": 0, "Bajo": 1, "Moderado": 2, "Alto": 3, "Muy alto": 4}

# ------------------------------------------------------------------
# Predicción
# ------------------------------------------------------------------
if submitted:
    entrada = pd.DataFrame(
        [[
            age,
            map_gender[gender],
            map_ethnicity[ethnicity],
            map_parental_education[parental_education],
            study_time,
            absences,
            map_yes_no[tutoring],
            map_parental_support[parental_support],
            int(extracurricular),
            int(sports),
            int(music),
            int(volunteering),
        ]],
        columns=FEATURE_ORDER,
    )

    pred = modelo.predict(entrada)[0]
    pred_clip = float(np.clip(pred, 0, 4))

    st.divider()
    st.subheader("📊 Resultado")

    r1, r2_ = st.columns([1, 2])
    with r1:
        st.metric("GPA estimado", f"{pred_clip:.2f} / 4.0")
    with r2_:
        st.progress(pred_clip / 4.0)

    if pred_clip >= 3.5:
        nivel, color = "Excelente (A)", "🟢"
    elif pred_clip >= 3.0:
        nivel, color = "Bueno (B)", "🟢"
    elif pred_clip >= 2.0:
        nivel, color = "Regular (C)", "🟡"
    elif pred_clip >= 1.0:
        nivel, color = "Bajo (D)", "🟠"
    else:
        nivel, color = "Muy bajo (F)", "🔴"

    st.write(f"{color} Nivel de desempeño estimado: **{nivel}**")

    with st.expander("Ver datos usados para la predicción"):
        st.dataframe(entrada, use_container_width=True)

    st.caption(
        "⚠️ Esta predicción es una estimación estadística basada en un modelo de "
        "regresión lineal y no debe usarse como único criterio para decisiones académicas."
    )

st.divider()
st.caption(
    "Proyecto: Student Performance Prediction using Machine Learning · "
    "Modelo: Regresión Lineal (Scikit-learn) · "
    "Dataset: Kaggle - rabieelkharoua/students-performance-dataset"
)
