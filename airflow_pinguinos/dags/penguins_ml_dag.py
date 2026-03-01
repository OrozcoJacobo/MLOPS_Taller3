from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from airflow_pinguinos.db import get_engine, clear_penguins_tables
from airflow_pinguinos.etl import (
    load_raw_penguins_to_db,
    preprocess_penguins_in_db,
)
from airflow_pinguinos.train import train_and_save_models

# Callables para las tasks

def _clear_tables() -> None:
    """Borra el contenido/tablas de la BD de datos de penguins."""
    engine = get_engine()
    clear_penguins_tables(engine)


def _load_raw() -> None:
    """Carga los datos crudos de Palmer Penguins a la tabla penguins_raw."""
    engine = get_engine()
    load_raw_penguins_to_db(engine, table_name="penguins_raw")


def _preprocess() -> None:
    """
    Toma los datos crudos desde penguins_raw, aplica preprocesamiento ligero
    y guarda el resultado en penguins_preprocessed.
    """
    engine = get_engine()
    preprocess_penguins_in_db(
        engine,
        source_table="penguins_raw",
        target_table="penguins_preprocessed",
    )


def _train_models() -> None:
    """
    Entrena los modelos usando la tabla penguins_preprocessed y guarda los
    modelos + registry.json en el directorio configurado (MODELS_DIR).
    """
    engine = get_engine()
    train_and_save_models(
        engine,
        source_table="penguins_preprocessed",
    )


# Definición del DAG

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="penguins_ml_pipeline",
    default_args=default_args,
    description="Pipeline de entrenamiento para modelos de Palmer Penguins",
    schedule_interval=None,  # lo corremos manualmente desde la UI
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["penguins", "mlops", "workshop"],
) as dag:

    clear_tables = PythonOperator(
        task_id="clear_penguins_tables",
        python_callable=_clear_tables,
    )

    load_raw = PythonOperator(
        task_id="load_raw_penguins",
        python_callable=_load_raw,
    )

    preprocess = PythonOperator(
        task_id="preprocess_penguins",
        python_callable=_preprocess,
    )

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=_train_models,
    )

    # Orden de ejecución del pipeline
    clear_tables >> load_raw >> preprocess >> train_models