"""Orquestacion del entrenamiento end-to-end (s3 -> s4 -> s5).

Une las tres piezas:

    postgres_client.read_covertype()   # lee s3 PostgreSQL
    training.train_model()             # entrena el modelo (s4 Jupyter)
    minio_client.publish_model()       # publica en s5 MinIO

Se usa desde el notebook ``train_covertype.ipynb`` y tambien puede ejecutarse
como script::

    python -m cubierta_forestal.application.train_pipeline
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from cubierta_forestal.application import training
from cubierta_forestal.infrastructure import minio_client, postgres_client


def _object_name() -> str:
    """Nombre del objeto en MinIO, con timestamp para conservar historico."""
    model_name = os.getenv("MODEL_NAME", "cubierta-forestal")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_name}/covertype-rf-{stamp}.joblib"


def run(
    limit: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Lee Postgres, entrena y publica el modelo. Devuelve un resumen."""
    df = postgres_client.read_covertype(limit=limit)
    if df.empty:
        raise RuntimeError(
            f"La tabla '{postgres_client.RAW_TABLE}' esta vacia: "
            "espera a que el DAG de Airflow ingiera datos antes de entrenar."
        )

    trained = training.train_model(
        df, test_size=test_size, random_state=random_state
    )

    object_name = _object_name()
    published = minio_client.publish_model(
        trained.pipeline, object_name, metadata=trained.metadata()
    )

    summary = {
        "rows_read": int(len(df)),
        "metrics": trained.metrics,
        "published": published,
    }
    return summary


def main() -> None:
    summary = run()
    print("Entrenamiento completado:")
    print(f"  filas leidas : {summary['rows_read']}")
    print(f"  accuracy     : {summary['metrics']['accuracy']:.4f}")
    print(f"  f1_macro     : {summary['metrics']['f1_macro']:.4f}")
    print(f"  publicado en : {summary['published']['bucket']}/{summary['published']['object']}")


if __name__ == "__main__":
    main()
