# =========================================================
# Imagen base oficial de Airflow
# =========================================================
FROM apache/airflow:2.9.1-python3.11

# =========================================================
# Instalar dependencias de sistema necesarias para compilar
# paquetes de Python (por si alguna lib lo necesita)
# =========================================================
USER root

RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# =========================================================
# Volver al usuario airflow y definir el WORKDIR
# =========================================================
USER airflow
WORKDIR /opt/airflow

# 👇 **AÑADE ESTA LÍNEA**
ENV PYTHONPATH=/opt/airflow

# =========================================================
# Instalar dependencias de Python que usa tu proyecto
# (joblib, pandas, scikit-learn, palmerpenguins, pymysql)
# =========================================================
RUN pip install --no-cache-dir \
    joblib \
    pandas \
    scikit-learn \
    palmerpenguins \
    pymysql

# =========================================================
# Copiar el paquete de Python al root de Airflow
# /opt/airflow/airflow_pinguinos queda en el PYTHONPATH
# =========================================================
COPY src/airflow_pinguinos ./airflow_pinguinos

# =========================================================
# Copiar DAGs y plugins de Airflow
# =========================================================
COPY dags ./dags
COPY plugins ./plugins