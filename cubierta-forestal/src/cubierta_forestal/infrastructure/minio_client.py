"""Consumo de MinIO para la Inference API (s6)."""
from __future__ import annotations

import io
import os

import joblib
from minio import Minio


class ModelNotAvailable(RuntimeError):
    """Aun no hay un modelo publicado en el bucket."""


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



_SERIALISED = (".joblib", ".pkl", ".pickle")
_cache: dict = {"model": None, "name": None}


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
    exact = os.getenv("MODEL_OBJECT")
    if exact:
        return exact
    candidates = [
        o for o in client.list_objects(bucket(), recursive=True)
        if o.object_name.lower().endswith(_SERIALISED)
    ]
    if not candidates:
        raise ModelNotAvailable(f"No hay objetos {_SERIALISED} en el bucket '{bucket()}'")
    return max(candidates, key=lambda o: o.last_modified).object_name



def load_model(force: bool = False):
    if _cache["model"] is not None and not force:
        return _cache["model"], _cache["name"]
    client = get_client()
    if not client.bucket_exists(bucket()):
        raise ModelNotAvailable(f"El bucket '{bucket()}' no existe todavia")
    name = _resolve_object(client)
    resp = client.get_object(bucket(), name)
    try:
        data = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    try:
        model = joblib.load(io.BytesIO(data))
    except Exception:
        import pickle
        model = pickle.loads(data)
    _cache.update(model=model, name=name)
    return model, name