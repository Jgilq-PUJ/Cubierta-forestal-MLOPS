"""Endpoints de inferencia: listar modelos de MinIO y predecir.

El modelo servido es un ``sklearn.pipeline.Pipeline`` con un
``ColumnTransformer`` adentro, asi que recibe un ``pandas.DataFrame`` con
columnas por nombre (no un array de floats). Las dos columnas categoricas
(``wilderness_area``, ``soil_type``) se normalizan con ``trim + lower`` antes de
predecir, igual que en la etapa ``processed`` con la que se entreno el modelo;
de lo contrario el one-hot las trata como categoria desconocida y la prediccion
se degrada en silencio.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from inference_api.infrastructure import minio_client

router = APIRouter(tags=["inference"])


class Instance(BaseModel):
    """Una fila de entrada con los 12 campos tipados del contrato del modelo."""

    elevation: int
    aspect: int
    slope: int
    horizontal_distance_to_hydrology: int
    vertical_distance_to_hydrology: int
    horizontal_distance_to_roadways: int
    hillshade_9am: int
    hillshade_noon: int
    hillshade_3pm: int
    horizontal_distance_to_fire_points: int
    wilderness_area: str
    soil_type: str


class PredictRequest(BaseModel):
    instances: list[Instance]


def get_bundle() -> minio_client.LoadedModel:
    """Dependencia: modelo cargado desde MinIO, o 503 si no hay ninguno."""
    try:
        return minio_client.load_model()
    except minio_client.ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _to_frame(instances: list[Instance], bundle: minio_client.LoadedModel) -> pd.DataFrame:
    categorical = set(bundle.categorical)
    rows = []
    for inst in instances:
        row = inst.model_dump()
        for col in categorical:
            value = row.get(col)
            if isinstance(value, str):
                row[col] = value.strip().lower()
        rows.append(row)
    # Ordena las columnas segun la lista `features` del metadata del modelo.
    return pd.DataFrame(rows, columns=bundle.features)


@router.get("/models")
def models():
    return {"bucket": minio_client.bucket(), "models": minio_client.list_models()}


@router.post("/reload")
def reload():
    try:
        bundle = minio_client.load_model(force=True)
    except minio_client.ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"reloaded": True, "model": bundle.info}


@router.post("/predict")
def predict(req: PredictRequest, bundle: minio_client.LoadedModel = Depends(get_bundle)):
    frame = _to_frame(req.instances, bundle)
    preds = bundle.model.predict(frame)
    return {"model": bundle.info, "predictions": [int(p) for p in preds]}
