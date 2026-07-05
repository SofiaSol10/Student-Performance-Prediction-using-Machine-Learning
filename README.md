# Student-Performance-Prediction-using-Machine-Learning
Machine learning project for predicting student GPA using Python and Scikit-learn.

Dataset: https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset

## 🚀 Aplicación en línea

🔗 https://student-performance-prediction-using-machine-learning-bqg7bwzs.streamlit.app/

## Tecnologías
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Modelos evaluados
- Regresión Lineal
- Árbol de Decisión
- Random Forest

## Resultados
El mejor modelo fue la Regresión Lineal con:
- R²: 0.953
- MAE: 0.155
- RMSE: 0.197

Además, se detectó y eliminó una variable con *data leakage* (`GradeClass`), mejorando la validez del análisis.
