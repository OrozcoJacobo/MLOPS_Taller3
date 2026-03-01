# MLOPS_Taller3

## Mapa Taller 3 Airflow
1. En alto nivel este proyecto tendra la siguiente estructura:
    * Airflow (webserver, scheduler) + BD de metadatos (Postgres)
    * Una BD MySQL exclusiva para datos de penguins
    * Un servicio API FastAPI para inferencia, que leera modelos. entrenados

2. Un paquete Python (con uv) donde vivira:
    * Codigo para
        * Conectase a MySQL
        * Cargar datos sin procesar
        * Preprocesar
        * Entrenar modelos y guardarlos 

3. Un DAG en Airflow con tasks que hagan:
    1. Borrar contenido de la BD
    2. Cargar datos de penguis sin preprocesar
    3. Preprocesar datos para entrenamiento
    4. Entrenar modelo eusando lo datos preprocesados de la BD

4. El API de ingerencia que use los modelos que entreno Airflows