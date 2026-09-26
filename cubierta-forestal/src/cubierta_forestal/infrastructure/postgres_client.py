"""Lectura de la base de datos de negocio (s3 PostgreSQL) para el entrenamiento.

El DAG de Airflow (s2) va acumulando filas del dataset covertype en la tabla
``covertype_raw``. Jupyter (s4) las lee desde aqui para entrenar el modelo.

La conexion se configura con las mismas variables de entorno que usa el DAG y
que ``docker-compose`` inyecta en el contenedor de Jupyter:

    DATA_DB_HOST, DATA_DB_PORT, DATA_DB_NAME, DATA_DB_USER, DATA_DB_PASSWORD
"""
from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

RAW_TABLE = "covertype_raw"


def database_url() -> str:
    """URL SQLAlchemy (psycopg3) construida desde las variables de entorno."""
    host = os.environ.get("DATA_DB_HOST", "postgres")
    port = os.environ.get("DATA_DB_PORT", "5432")
    name = os.environ.get("DATA_DB_NAME", "cubierta_forestal")
    user = os.environ.get("DATA_DB_USER", "app")
    password = os.environ.get("DATA_DB_PASSWORD", "app")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"


def get_engine() -> Engine:
    """Engine SQLAlchemy hacia la base de negocio."""
    return create_engine(database_url(), pool_pre_ping=True)


def read_covertype(
    limit: int | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Devuelve las filas de ``covertype_raw`` como ``DataFrame``.

    Args:
        limit: si se indica, trae solo las ``limit`` filas mas recientes.
        engine: engine ya creado (util en pruebas); si es ``None`` se crea uno.
    """
    own_engine = engine is None
    engine = engine or get_engine()
    query = f"SELECT * FROM {RAW_TABLE} ORDER BY id"
    if limit is not None:
        query = f"SELECT * FROM {RAW_TABLE} ORDER BY id DESC LIMIT {int(limit)}"
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
    finally:
        if own_engine:
            engine.dispose()
    return df
