#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import pickle
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# --- Funciones Auxiliares ---
@st.cache_data  # Cacheo para mejorar el rendimiento (datos no cambian frecuentemente)
def cargar_datos(url):
    """Carga el dataset desde la URL."""
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    return pd.read_csv(url, names=column_names)

@st.cache_data
def preprocesar_datos(data):
    """Preprocesa el dataset: reemplaza ceros por NaN y los imputa con la mediana."""
    columns_a_procesar = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data_procesada = data.copy()
    for col in columns_a_procesar:
        data_procesada[col] = data_procesada[col].replace(0, np.nan)
        data_procesada[col] = data_procesada[col].fillna(data_procesada[col].median())
    return data_procesada

def dividir_y_escalar(data):
    """Divide los datos en train/test y escala las características."""
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_escalado = scaler.fit_transform(X_train)
    X_test_escalado = scaler.transform(X_test)  # Escalar también el test
    return X_train_escalado, X_test_escalado, y_train, y_test, scaler

def cargar_o_entrenar_modelo(X_train, y_train, ruta_modelo='diabetes_model.pkl'):
    """Carga el modelo si existe, si no, lo entrena y guarda."""
    try:
        modelo = pickle.load(open(ruta_modelo, 'rb'))
    except FileNotFoundError:
        modelo = LogisticRegression(random_state=42, max_iter=1000)
        modelo.fit(X_train, y_train)
        pickle.dump(modelo, open(ruta_modelo, 'wb'))
    return modelo

def predecir_riesgo(datos_paciente, escalador, modelo):
    """Realiza la predicción de riesgo de diabetes."""
    arreglo_paciente = np.array(datos_paciente).reshape(1, -1)
    paciente_escalado = escalador.transform(arreglo_paciente)
    probabilidad = modelo.predict_proba(paciente_escalado)[0, 1]
    return probabilidad

def graficar_metricas(y_true, y_pred, y_prob):
    """Genera y muestra las métricas y gráficos."""
    st.subheader("Métricas de Evaluación")
    st.write(classification_report(y_true, y_pred))

    matriz_confusion = confusion_matrix(y_true, y_pred)
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt.gcf())  # Obtener la figura actual para Streamlit
    plt.clf()  # Limpiar la figura para futuras gráficas

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# --- Main de Streamlit ---
def main():
    st.title('Predicción de Riesgo de Diabetes')

    # Cargar y preprocesar datos
    url_datos = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    datos = cargar_datos(url_datos)
    datos_procesados = preprocesar_datos(datos)

    # Dividir y escalar
    X_train_escalado, X_test_escalado, y_train, y_test, escalador = dividir_y_escalar(datos_procesados)

    # Entrenar o cargar el modelo
    modelo = cargar_o_entrenar_modelo(X_train_escalado, y_train)

    # Predicciones en el conjunto de prueba (para evaluación)
    y_pred_test = modelo.predict(X_test_escalado)
    y_prob_test = modelo.predict_proba(X_test_escalado)[:, 1]

    # Mostrar métricas de evaluación
    with st.expander("Mostrar Métricas del Modelo"): # Expander para ocultar/mostrar
        graficar_metricas(y_test, y_pred_test, y_prob_test)

    # Interfaz de entrada para el usuario
    st.sidebar.header('Ingrese los datos del paciente:')
    embarazos = st.sidebar.slider('Embarazos', 0, 17, 3)
    glucosa = st.sidebar.slider('Glucosa', 0, 200, 120)
    presion_arterial = st.sidebar.slider('Presión Arterial', 0, 150, 80)
    grosor_piel = st.sidebar.slider('Grosor de la Piel', 0, 100, 20)
    insulina = st.sidebar.slider('Insulina', 0, 900, 30)
    imc = st.sidebar.slider('Índice de Masa Corporal', 0.0, 67.0, 30.0)
    pedigri_diabetes = st.sidebar.slider('Función Pedigrí Diabetes', 0.0, 2.5, 0.5)
    edad = st.sidebar.slider('Edad', 21, 81, 35)

    datos_paciente = [embarazos, glucosa, presion_arterial, grosor_piel, insulina, imc, pedigri_diabetes, edad]

    # Botón de predicción
    if st.button('Predecir Riesgo'):
        riesgo = predecir_riesgo(datos_paciente, escalador, modelo)
        st.write(f'El riesgo de diabetes para el paciente es: {riesgo * 100:.2f}%')

        if riesgo > 50:
            st.warning('El paciente tiene un alto riesgo de diabetes.')
        else:
            st.success('El paciente tiene un bajo riesgo de diabetes.')

if __name__ == '__main__':
    main()

