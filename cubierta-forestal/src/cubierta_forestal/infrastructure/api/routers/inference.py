"""Endpoints de inferencia: listar modelos de MinIO y predecir."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cubierta_forestal.infrastructure import minio_client

router = APIRouter(tags=["inference"])


class PredictRequest(BaseModel):
    instances: list[list[float]]


@router.get("/models")
def models():
    return {"bucket": minio_client.bucket(), "models": minio_client.list_models()}


@router.post("/reload")
def reload():
    try:
        _, name = minio_client.load_model(force=True)
    except minio_client.ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"reloaded": True, "model": name}


@router.post("/predict")
def predict(req: PredictRequest):
    try:
        model, name = minio_client.load_model()
    except minio_client.ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    X = np.asarray(req.instances, dtype=float)
    preds = model.predict(X)
    return {"model": name, "predictions": np.asarray(preds).tolist()}