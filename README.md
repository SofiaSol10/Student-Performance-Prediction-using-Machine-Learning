Student Performance Prediction using Machine Learning

Objetivo
    Desarrollar un modelo de Machine Learning capaz de predecir el GPA de estudiantes utilizando variables académicas y socioeconómicas.

Dataset

    Student Performance Dataset (Kaggle)

Tecnologías
    Python
    Pandas
    NumPy
    Matplotlib
    Seaborn
    Scikit-Learn
    Streamlit

Flujo del proyecto
    Limpieza de datos
    EDA
    Eliminación de Data Leakage
    Comparación de modelos
    Validación Cruzada
    Interpretación de resultados


Hallazgos
    Las ausencias fueron la variable más importante.
    El apoyo parental y las horas de estudio tuvieron una influencia positiva.
    Se detectó y eliminó una variable con data leakage (GradeClass).
    La Regresión Lineal obtuvo el mejor desempeño.

Comparación entre Regresión Lineal, Árbol de Decisión y Random Forest.
El análisis de correlación mostró que el número de ausencias (Absences) es la variable más influyente sobre el GPA, con una correlación de -0.919, indicando que a medida que aumentan las ausencias, el rendimiento académico disminuye considerablemente. Por otro lado, variables como el apoyo parental, las horas de estudio semanales y las tutorías presentan correlaciones positivas, aunque de menor magnitud, lo que sugiere que también contribuyen al desempeño académico.