"""Test de integracion opcional contra el MinIO del compose.

Se salta por defecto. Para ejecutarlo, levantar el stack
(``docker compose up -d minio minio-init``), publicar un modelo (correr el
notebook) y exportar ``RUN_MINIO_INTEGRATION=1`` con las variables
``MINIO_ENDPOINT`` / ``MINIO_ACCESS_KEY`` / ``MINIO_SECRET_KEY`` /
``MODELS_BUCKET`` / ``MODEL_NAME`` apuntando al stack:

    RUN_MINIO_INTEGRATION=1 MINIO_ENDPOINT=http://localhost:9000 \
        uv run --group api --group test pytest tests/test_integration_minio.py
"""
from __future__ import annotations

import os

import pytest

from inference_api.infrastructure import minio_client

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MINIO_INTEGRATION") != "1",
    reason="integracion con MinIO deshabilitada (exportar RUN_MINIO_INTEGRATION=1)",
)


def test_load_model_from_compose_minio():
    bundle = minio_client.load_model(force=True)
    assert bundle.model is not None
    info = bundle.info
    assert info["name"]
    assert info["trained_at"]
    assert bundle.features  # la lista viene del metadata.json


def test_predict_against_compose_minio():
    from fastapi.testclient import TestClient

    from inference_api.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={
                "instances": [
                    {
                        "elevation": 2596,
                        "aspect": 51,
                        "slope": 3,
                        "horizontal_distance_to_hydrology": 258,
                        "vertical_distance_to_hydrology": 0,
                        "horizontal_distance_to_roadways": 510,
                        "hillshade_9am": 221,
                        "hillshade_noon": 232,
                        "hillshade_3pm": 148,
                        "horizontal_distance_to_fire_points": 6279,
                        "wilderness_area": "Rawah",
                        "soil_type": "C2717",
                    }
                ]
            },
        )
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 1
    assert 0 <= preds[0] <= 6
