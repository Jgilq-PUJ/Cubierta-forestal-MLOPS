"""Consumo y publicacion de modelos en MinIO (s5).

La Inference API (s6) *lee* el modelo desde aqui y Jupyter (s4) lo *publica*
tras entrenar, de modo que ambos extremos comparten un unico contrato de bucket.
"""
from __future__ import annotations

import io
import json
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



def ensure_bucket() -> None:
    """Crea el bucket de modelos si aun no existe."""
    client = get_client()
    if not client.bucket_exists(bucket()):
        client.make_bucket(bucket())


def publish_model(model, object_name: str, metadata: dict | None = None) -> dict:
    """Serializa ``model`` con joblib y lo sube a MinIO (s4 -> s5).

    Si se pasan ``metadata`` se guarda un objeto ``<object_name>.metadata.json``
    al lado del modelo (no interfiere con la resolucion del modelo, que solo
    mira las extensiones serializadas de ``_SERIALISED``).

    Devuelve un resumen con el bucket, el objeto y su tamanio en bytes.
    """
    client = get_client()
    if not client.bucket_exists(bucket()):
        client.make_bucket(bucket())

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    data = buffer.getvalue()
    client.put_object(
        bucket(),
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type="application/octet-stream",
    )

    result = {"bucket": bucket(), "object": object_name, "size": len(data)}

    if metadata is not None:
        meta_name = f"{object_name}.metadata.json"
        meta_bytes = json.dumps(metadata, indent=2, default=str).encode("utf-8")
        client.put_object(
            bucket(),
            meta_name,
            io.BytesIO(meta_bytes),
            length=len(meta_bytes),
            content_type="application/json",
        )
        result["metadata_object"] = meta_name

    # Invalidar la cache local para que un siguiente load_model traiga el nuevo.
    _cache.update(model=None, name=None)
    return result


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