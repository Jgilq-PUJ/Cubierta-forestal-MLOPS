"""Consumo de MinIO para la Inference API (s6).

Carga el modelo publicado por el notebook de entrenamiento siguiendo la
convencion de rutas ``<model_name>/<dataset_version>/<trained_at>/``. Bajo ese
prefijo conviven dos objetos: el modelo serializado con ``pickle``
(``model.pickle``) y un ``metadata.json`` con la lista de ``features``, las
clases y las metricas de entrenamiento.

El modelo y su metadata se cachean en memoria; ``load_model(force=True)``
recarga desde el bucket sin reiniciar el proceso.
"""
from __future__ import annotations

import io
import json
import os
import pickle
from dataclasses import dataclass, field
from posixpath import dirname
from typing import Any

import joblib
from minio import Minio
from minio.error import S3Error


class ModelNotAvailable(RuntimeError):
    """Aun no hay un modelo publicado en el bucket."""


# Convencion de features del modelo (tambien viene en metadata.json). Se usa
# como respaldo cuando el metadata no esta disponible.
DEFAULT_NUMERIC = [
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
DEFAULT_CATEGORICAL = ["wilderness_area", "soil_type"]
DEFAULT_FEATURES = DEFAULT_NUMERIC + DEFAULT_CATEGORICAL

_SERIALISED = (".pickle", ".pkl", ".joblib")
_METADATA_OBJECT = "metadata.json"


@dataclass
class LoadedModel:
    """Modelo cargado junto a su metadata y su ruta en el bucket."""

    model: Any
    object_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def info(self) -> dict:
        """Identificacion del modelo para la respuesta del endpoint.

        Prefiere los campos del ``metadata.json`` y cae en el parseo de la
        ruta ``<name>/<dataset_version>/<trained_at>/<file>`` cuando faltan.
        """
        parts = self.object_name.split("/")
        name = self.metadata.get("model_name")
        dataset_version = self.metadata.get("dataset_version")
        trained_at = self.metadata.get("trained_at")
        if name is None and len(parts) >= 4:
            name = parts[0]
        if dataset_version is None and len(parts) >= 4:
            dataset_version = parts[1]
        if trained_at is None and len(parts) >= 4:
            trained_at = parts[2]
        return {
            "name": name,
            "dataset_version": None if dataset_version is None else str(dataset_version),
            "trained_at": trained_at,
        }

    @property
    def features(self) -> list[str]:
        feats = self.metadata.get("features")
        return list(feats) if feats else list(DEFAULT_FEATURES)

    @property
    def categorical(self) -> list[str]:
        cats = self.metadata.get("categorical")
        return list(cats) if cats else list(DEFAULT_CATEGORICAL)


def _split_endpoint(raw: str) -> tuple[str, bool]:
    secure = raw.startswith("https://")
    return raw.split("://", 1)[-1], secure


def get_client() -> Minio:
    host, secure = _split_endpoint(os.getenv("MINIO_ENDPOINT", "http://minio:9000"))
    return Minio(
        host,
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=secure,
    )


def bucket() -> str:
    return os.getenv("MODELS_BUCKET", "models")


def model_name() -> str | None:
    return os.getenv("MODEL_NAME") or None


_cache: dict = {"bundle": None}


def list_models() -> list[dict]:
    client = get_client()
    if not client.bucket_exists(bucket()):
        return []
    return [
        {
            "name": o.object_name,
            "size": o.size,
            "last_modified": o.last_modified.isoformat() if o.last_modified else None,
        }
        for o in client.list_objects(bucket(), recursive=True)
    ]


def _resolve_object(client: Minio) -> str:
    """Objeto del modelo a cargar.

    ``MODEL_OBJECT`` fija el objeto; si no esta definido, resuelve el
    serializado mas reciente bajo el prefijo ``MODEL_NAME/``.
    """
    exact = os.getenv("MODEL_OBJECT")
    if exact:
        return exact

    name = model_name()
    prefix = f"{name}/" if name else None
    candidates = [
        o
        for o in client.list_objects(bucket(), prefix=prefix, recursive=True)
        if o.object_name.lower().endswith(_SERIALISED)
    ]
    if not candidates:
        where = f"con prefijo '{prefix}' " if prefix else ""
        raise ModelNotAvailable(
            f"No hay objetos {_SERIALISED} {where}en el bucket '{bucket()}'"
        )
    latest = max(candidates, key=lambda o: (o.last_modified, o.object_name))
    return latest.object_name


def _read_object(client: Minio, object_name: str) -> bytes:
    resp = client.get_object(bucket(), object_name)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def _deserialise(object_name: str, data: bytes) -> Any:
    if object_name.lower().endswith(".joblib"):
        return joblib.load(io.BytesIO(data))
    try:
        return pickle.loads(data)
    except Exception:  # noqa: BLE001 - respaldo por si se publico con joblib
        return joblib.load(io.BytesIO(data))


def _read_metadata(client: Minio, object_name: str) -> dict:
    prefix = dirname(object_name)
    meta_object = f"{prefix}/{_METADATA_OBJECT}" if prefix else _METADATA_OBJECT
    try:
        raw = _read_object(client, meta_object)
    except S3Error:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def load_model(force: bool = False) -> LoadedModel:
    """Carga (y cachea) el modelo publicado en MinIO junto a su metadata."""
    cached = _cache.get("bundle")
    if cached is not None and not force:
        return cached

    client = get_client()
    if not client.bucket_exists(bucket()):
        raise ModelNotAvailable(f"El bucket '{bucket()}' no existe todavia")

    object_name = _resolve_object(client)
    model = _deserialise(object_name, _read_object(client, object_name))
    metadata = _read_metadata(client, object_name)

    bundle = LoadedModel(model=model, object_name=object_name, metadata=metadata)
    _cache["bundle"] = bundle
    return bundle
