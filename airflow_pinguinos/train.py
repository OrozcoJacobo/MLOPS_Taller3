# src/airflow_pinguinos/train.py

from pathlib import Path
from typing import Dict

import os
import json
import joblib
import pandas as pd

from sqlalchemy.engine import Engine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .config import MODELS_DIR


def train_and_save_models(
    engine: Engine,
    source_table: str = "penguins_preprocessed",
    models_dir: Path | None = None,
) -> Dict[str, float]:
    """
    Entrena varios modelos usando la tabla preprocesada en la BD
    y guarda los pipelines entrenados + registry.json.

    Devuelve un diccionario {nombre_modelo: accuracy_val}.
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Leer datos desde la BD
    df = pd.read_sql(f"SELECT * FROM {source_table}", engine)

    X = df.drop("species", axis=1)
    y = df["species"]

    # 2. Definir preprocesamiento (igual al notebook)
    numeric_features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "year",
    ]

    categorical_features = [
        "island",
        "sex",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 3. Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # 4. Definir modelos y entrenar en loop
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    trained_models: Dict[str, Pipeline] = {}
    val_scores: Dict[str, float] = {}

    for name, clf in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clf),
            ]
        )

        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_val)
        acc = accuracy_score(y_val, preds)

        trained_models[name] = pipe
        val_scores[name] = acc

        print(f"{name} -> val accuracy: {acc:.4f}")

    # 5. Guardar modelos y registry.json
    for name, pipe in trained_models.items():
        joblib.dump(pipe, models_dir / f"{name}.joblib")

    registry = {
        "default_model": "rf",
        "available_models": list(trained_models.keys()),
    }

    with open(models_dir / "registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Modelos guardados en {models_dir}")

    return val_scores