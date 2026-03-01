
import os
from pathlib import Path

# Directorio donde se guardan modelos y registry.json
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

# Parámetros de conexión a la BD de datos (MySQL)
MYSQL_HOST = os.getenv("MYSQL_HOST", "penguins-db")  # nombre del servicio en docker-compose
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "penguins_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "penguins_pass")
MYSQL_DB = os.getenv("MYSQL_DB", "penguins_db")

def get_mysql_url() -> str:
    """
    Construye el connection string para SQLAlchemy.
    """
    return (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )