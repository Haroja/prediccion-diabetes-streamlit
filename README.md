# Predicción de Riesgo de Diabetes con Regresión Logística

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://haroja-prediccion-diabetes-streamlit.streamlit.app/)

Este proyecto implementa un modelo de regresión logística para predecir el riesgo de diabetes en pacientes. Incluye tanto el análisis exploratorio y la construcción del modelo (en un Jupyter Notebook) como una aplicación interactiva de Streamlit para usar el modelo.

## Descripción

La regresión logística es un método estadístico que se utiliza para predecir la probabilidad de un resultado binario. En este caso, el objetivo es predecir si un paciente tiene o no diabetes, basándose en varias características médicas.

Este proyecto se divide en dos partes principales:

1.  **Análisis y Modelado (Jupyter Notebook):**
    * Exploración del dataset de diabetes de Pima Indians.
    * Preprocesamiento de datos para manejar valores faltantes.
    * Construcción y evaluación de un modelo de regresión logística.
    * Visualización de métricas de rendimiento del modelo.
2.  **Aplicación Streamlit:**
    * Interfaz web interactiva que permite a los usuarios ingresar datos de pacientes.
    * Utilización del modelo entrenado para predecir el riesgo de diabetes en tiempo real.
    * Visualización clara de los resultados de la predicción.

## Archivos

* `Predicción Diabetes.ipynb`: Jupyter Notebook que contiene el análisis exploratorio de datos, el preprocesamiento, el entrenamiento del modelo y la evaluación.
* `streamlit_diabetes.py`: Script de Python que implementa la aplicación Streamlit para la predicción interactiva.
* `diabetes_model.pkl`: Archivo que contiene el modelo de regresión logística entrenado (serializado con `pickle`).
* `requirements.txt`: Archivo que lista todas las dependencias de Python necesarias para ejecutar el proyecto.
* `README.md`: Este archivo, que proporciona información sobre el proyecto.

## Dataset

El dataset utilizado es el [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database), disponible en Kaggle y originalmente del UCI Machine Learning Repository. Contiene información sobre pacientes femeninas de ascendencia Pima Indian, incluyendo medidas como:

* Número de embarazos
* Nivel de glucosa en plasma
* Presión arterial diastólica
* Grosor del pliegue cutáneo del tríceps
* Nivel de insulina en suero
* Índice de masa corporal (BMI)
* Función pedigrí de diabetes
* Edad
* Variable objetivo: si tiene o no diabetes

## Instalación

1.  Clona el repositorio:

    ```bash
    git clone [https://github.com/Haroja/prediccion-diabetes-streamlit.git](https://github.com/Haroja/prediccion-diabetes-streamlit.git)
    ```

2.  Navega al directorio del proyecto:

    ```bash
    cd prediccion-diabetes-streamlit
    ```

3.  Crea un entorno virtual (recomendado):

    ```bash
    python -m venv venv
    ```

    * Activa el entorno virtual:
        * En Windows:

            ```bash
            venv\Scripts\activate
            ```

        * En macOS y Linux:

            ```bash
            source venv/bin/activate
            ```

4.  Instala las dependencias de Python:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

### Ejecutar la Aplicación Streamlit

1.  Asegúrate de tener el entorno virtual activado.
2.  Ejecuta la aplicación Streamlit:

    ```bash
    streamlit run streamlit_diabetes.py
    ```

3.  Streamlit abrirá automáticamente la aplicación en tu navegador web.

### Usar la Aplicación

1.  **Ingreso de Datos:**
    * En la barra lateral izquierda, encontrarás controles deslizantes para ingresar los datos del paciente.
    * Los parámetros incluyen:
        * **Embarazos:** Número de embarazos.
        * **Glucosa:** Nivel de glucosa en plasma.
        * **Presión Arterial:** Presión arterial diastólica.
        * **Grosor de la Piel:** Grosor del pliegue cutáneo del tríceps.
        * **Insulina:** Nivel de insulina en suero.
        * **BMI:** Índice de masa corporal.
        * **Función Pedigrí Diabetes:** Función pedigrí de diabetes.
        * **Edad:** Edad del paciente.
2.  **Predicción:**
    * Después de ingresar los datos, haz clic en el botón "Predecir Riesgo".
3.  **Resultados:**
    * La aplicación mostrará el riesgo de diabetes predicho como un porcentaje.
    * También proporcionará una interpretación del riesgo (alto o bajo) basada en un umbral del 50%.
4.  **Métricas (Opcional):**
    * Puedes expandir la sección "Mostrar Métricas del Modelo" para ver métricas detalladas de rendimiento del modelo (reporte de clasificación, matriz de confusión, curva ROC) calculadas en el conjunto de prueba.

## Modelo de Machine Learning

Se utiliza un modelo de regresión logística de scikit-learn para la predicción. El modelo se entrena en el dataset de Pima Indians y se guarda en el archivo `diabetes_model.pkl`.

## Ver la Aplicación en Streamlit Cloud

Puedes ver la aplicación en funcionamiento aquí:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://haroja-prediccion-diabetes-streamlit.streamlit.app/)

## Contribución

¡Las contribuciones son bienvenidas! Si encuentras errores o tienes sugerencias de mejora, siéntete libre de abrir un issue o enviar un pull request.

