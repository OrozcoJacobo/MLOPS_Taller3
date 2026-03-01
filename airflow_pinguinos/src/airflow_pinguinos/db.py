from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import get_mysql_url


def get_engine() -> Engine:
    """
    Crea un engine de SQLAlchemy para conectarse a la BD de datos.
    """
    url = get_mysql_url()
    return create_engine(url, future=True)


def clear_penguins_tables(engine: Engine) -> None:
    """
    Borra el contenido de las tablas de pingüinos si existen.

    No elimina el esquema completo, solo hace TRUNCATE/DELETE
    sobre tablas conocidas.
    """
    tables = [
        "penguins_raw",
        "penguins_preprocessed",
    ]

    with engine.begin() as conn:
        for table in tables:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))