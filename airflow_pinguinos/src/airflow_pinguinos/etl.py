# src/airflow_pinguinos/etl.py

from sqlalchemy.engine import Engine
import pandas as pd
from palmerpenguins import load_penguins


def load_raw_penguins_to_db(engine: Engine, table_name: str = "penguins_raw") -> None:
    """
    Carga los datos crudos de Palmer Penguins a la base de datos,
    sin ningún preprocesamiento.

    - Usa palmerpenguins.load_penguins()
    - Escribe en una tabla 'penguins_raw'
    """
    df = load_penguins()
    # Guardamos tal cual vienen
    df.to_sql(table_name, engine, if_exists="replace", index=False)


def preprocess_penguins_in_db(
    engine: Engine,
    source_table: str = "penguins_raw",
    target_table: str = "penguins_preprocessed",
) -> None:
    """
    Realiza un preprocesamiento ligero y guarda el resultado en otra tabla.

    Aquí separamos explícitamente la etapa de 'preprocesamiento' para que
    cumpla con el requerimiento del taller:

        - Leer datos crudos de la tabla source_table.
        - Hacer limpiezas básicas (por ejemplo, eliminar filas sin 'species').
        - Guardar el resultado en target_table.

    El resto del preprocesamiento (imputación, one-hot, etc.) lo hará el
    Pipeline de scikit-learn dentro de la función de entrenamiento.
    """
    # Leer datos crudos
    df = pd.read_sql(f"SELECT * FROM {source_table}", engine)

    # Ejemplo de preprocesamiento: eliminar filas sin etiqueta
    df = df.dropna(subset=["species"]).reset_index(drop=True)

    # Podríamos añadir más transformaciones aquí si el profe lo pide

    # Guardar tabla preprocesada
    df.to_sql(target_table, engine, if_exists="replace", index=False)