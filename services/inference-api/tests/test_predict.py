"""Tests del endpoint /predict con el modelo cargado desde un pickle de fixture."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inference_api.infrastructure import minio_client
from inference_api.main import app
from inference_api.routers import inference


def test_predict_ok(client, valid_instance):
    resp = client.post("/predict", json={"instances": [valid_instance]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == {
        "name": "cubierta-forestal",
        "dataset_version": "1",
        "trained_at": "20260926T183018Z",
    }
    assert len(body["predictions"]) == 1
    pred = body["predictions"][0]
    assert isinstance(pred, int)
    assert 0 <= pred <= 6


def test_normalization_matches_lowercase(client, valid_instance):
    messy = {**valid_instance, "wilderness_area": "  RAWAH "}
    clean = {**valid_instance, "wilderness_area": "rawah"}
    messy_pred = client.post("/predict", json={"instances": [messy]}).json()["predictions"]
    clean_pred = client.post("/predict", json={"instances": [clean]}).json()["predictions"]
    assert messy_pred == clean_pred


def test_instances_as_list_of_lists_is_422(client):
    payload = {
        "instances": [
            [2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279, "Rawah", "C2717"]
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_missing_field_is_422(client, valid_instance):
    incomplete = dict(valid_instance)
    del incomplete["elevation"]
    resp = client.post("/predict", json={"instances": [incomplete]})
    assert resp.status_code == 422


def test_wrong_type_is_422(client, valid_instance):
    bad = {**valid_instance, "elevation": "not-a-number"}
    resp = client.post("/predict", json={"instances": [bad]})
    assert resp.status_code == 422


def test_multiple_instances(client, valid_instance):
    other = {**valid_instance, "wilderness_area": "Neota", "soil_type": "C7745"}
    resp = client.post("/predict", json={"instances": [valid_instance, other]})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_predict_returns_503_when_no_model(monkeypatch, valid_instance):
    def _raise(*_args, **_kwargs):
        raise minio_client.ModelNotAvailable("El bucket 'models' no existe todavia")

    monkeypatch.setattr(minio_client, "load_model", _raise)
    app.dependency_overrides.pop(inference.get_bundle, None)
    with TestClient(app) as test_client:
        resp = test_client.post("/predict", json={"instances": [valid_instance]})
    assert resp.status_code == 503
    assert "models" in resp.json()["detail"]
