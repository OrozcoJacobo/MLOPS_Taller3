# =========================================================
# Imagen base oficial de Airflow
# =========================================================
FROM apache/airflow:2.9.1-python3.11

# =========================================================
# Instalar dependencias del sistema necesarias
# =========================================================
USER root

RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# =========================================================
# Volver al usuario airflow
# =========================================================
USER airflow
WORKDIR /opt/airflow

# =========================================================
# Instalar dependencias del proyecto con pip
# (requirements.txt fue generado con uv en tu máquina)
# =========================================================
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# =========================================================
# Copiar DAGs, plugins y código fuente
# =========================================================
COPY dags ./dags
COPY plugins ./plugins
COPY src ./src

# Exponer el paquete Python
ENV PYTHONPATH=/opt/airflow/src