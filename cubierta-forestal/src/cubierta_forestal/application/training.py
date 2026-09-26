"""Entrenamiento del clasificador de cobertura forestal (s4 Jupyter).

Este modulo contiene la logica *pura* de entrenamiento: no habla con Postgres
ni con MinIO, solo transforma un ``DataFrame`` en un ``Pipeline`` de scikit-learn
listo para publicarse. Asi puede probarse sin infraestructura y reutilizarse
tanto desde el notebook como desde ``train_pipeline.run``.

Contrato de features con la Inference API (s6)
----------------------------------------------
La Inference API recibe ``instances: list[list[float]]`` (solo numeros). Por eso
el vector de entrada del modelo esta compuesto, en este orden fijo
(``FEATURE_ORDER``), por:

    1..10  las 10 columnas numericas del dataset covertype
    11     el codigo entero de ``wilderness_area``
    12     el codigo entero de ``soil_type``

Los codigos de las dos columnas categoricas se calculan de forma estable
(orden alfabetico de las categorias vistas en entrenamiento) y se guardan en
los metadatos del modelo para poder reconstruir el vector en el cliente. Dentro
del ``Pipeline`` esas dos posiciones se expanden con ``OneHotEncoder`` antes de
llegar al clasificador.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Columnas tal como las entrega la Data API / DAG de ingesta.
NUMERIC_FEATURES = [
    "elevation",
    "aspect",
    "slope",
    "horizontal_distance_to_hydrology",
    "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "horizontal_distance_to_fire_points",
]
CATEGORICAL_FEATURES = ["wilderness_area", "soil_type"]
# Orden exacto del vector numerico que espera el modelo publicado.
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "cover_type"

# Indices (0-based) de las columnas categoricas dentro de FEATURE_ORDER.
_CATEGORICAL_INDICES = [FEATURE_ORDER.index(col) for col in CATEGORICAL_FEATURES]


@dataclass
class TrainedModel:
    """Resultado del entrenamiento listo para publicar en MinIO."""

    pipeline: Pipeline
    metrics: dict[str, float]
    categories: dict[str, list[str]]
    feature_order: list[str] = field(default_factory=lambda: list(FEATURE_ORDER))
    target: str = TARGET
    n_samples: int = 0

    def metadata(self) -> dict[str, Any]:
        """Diccionario JSON-serializable con la ficha del modelo."""
        return {
            "target": self.target,
            "feature_order": self.feature_order,
            "categorical_features": list(CATEGORICAL_FEATURES),
            "categories": self.categories,
            "metrics": self.metrics,
            "n_samples": self.n_samples,
            "model_type": type(self.pipeline.named_steps["clf"]).__name__,
        }


def category_maps(df: pd.DataFrame) -> dict[str, list[str]]:
    """Categorias ordenadas (estables) por cada columna categorica."""
    return {
        col: sorted(df[col].astype(str).unique().tolist())
        for col in CATEGORICAL_FEATURES
    }


def build_feature_matrix(
    df: pd.DataFrame, categories: dict[str, list[str]]
) -> np.ndarray:
    """Convierte el ``DataFrame`` en la matriz numerica que consume el modelo.

    Las columnas categoricas se reemplazan por su codigo entero segun
    ``categories``; una categoria no vista queda como ``-1`` (el ``OneHotEncoder``
    del pipeline la ignora gracias a ``handle_unknown='ignore'``).
    """
    frame = df.copy()
    for col in CATEGORICAL_FEATURES:
        lookup = {value: code for code, value in enumerate(categories[col])}
        frame[col] = frame[col].astype(str).map(lookup).fillna(-1).astype(int)
    return frame[FEATURE_ORDER].to_numpy(dtype=float)


def _build_pipeline(random_state: int) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                _CATEGORICAL_INDICES,
            )
        ],
        remainder="passthrough",
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
    return Pipeline([("preprocess", preprocess), ("clf", classifier)])


def train_model(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainedModel:
    """Entrena el clasificador de ``cover_type`` a partir del dataset crudo."""
    missing = [c for c in FEATURE_ORDER + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")

    frame = df.dropna(subset=FEATURE_ORDER + [TARGET]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("El dataset no tiene filas utilizables para entrenar")

    categories = category_maps(frame)
    X = build_feature_matrix(frame, categories)
    y = frame[TARGET].astype(int).to_numpy()

    # Estratificar solo si cada clase tiene al menos 2 ejemplos.
    _, counts = np.unique(y, return_counts=True)
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipeline = _build_pipeline(random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    return TrainedModel(
        pipeline=pipeline,
        metrics=metrics,
        categories=categories,
        n_samples=int(len(frame)),
    )
