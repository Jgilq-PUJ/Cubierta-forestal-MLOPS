"""Fixtures de test.

Se construye un ``sklearn.pipeline.Pipeline`` con la misma forma que el del
notebook de entrenamiento (``ColumnTransformer`` con ``OneHotEncoder`` para las
categoricas), se serializa con ``pickle`` en ``tests/fixtures/model.pickle`` y se
acompana de un ``metadata.json``. Los tests cargan el modelo desde ese pickle,
sin depender de MinIO.

El modelo se entrena con las categoricas ya normalizadas (minusculas y sin
espacios), de modo que la normalizacion que aplica la API es lo que hace que
``"  RAWAH "`` y ``"rawah"`` produzcan la misma prediccion.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from inference_api.infrastructure import minio_client
from inference_api.main import app
from inference_api.routers import inference

NUMERIC = [
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
CATEGORICAL = ["wilderness_area", "soil_type"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "cover_type"

# Valores ya normalizados (minusculas, sin espacios), como en la etapa processed.
WILDERNESS = ["rawah", "neota", "comanche peak", "cache la poudre"]
SOIL = ["c2717", "c7745", "c8776", "c4703"]
CLASSES = list(range(7))  # 0..6

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_PICKLE = FIXTURES / "model.pickle"
METADATA_JSON = FIXTURES / "metadata.json"

OBJECT_NAME = "cubierta-forestal/1/20260926T183018Z/model.pickle"


def _build_pipeline() -> Pipeline:
    rng = np.random.default_rng(42)
    n = 400
    data = {col: rng.integers(0, 4000, n) for col in NUMERIC}
    data["wilderness_area"] = rng.choice(WILDERNESS, n)
    data["soil_type"] = rng.choice(SOIL, n)
    X = pd.DataFrame(data)[FEATURES]
    # Etiquetas en 0..6 (la clase 3 se omite, como en los datos reales).
    y = pd.Series(rng.choice([c for c in CLASSES if c != 3], n), name=TARGET)

    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=25, n_jobs=1, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def _ensure_fixture() -> tuple[Path, Path]:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if not MODEL_PICKLE.exists():
        model = _build_pipeline()
        with MODEL_PICKLE.open("wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if not METADATA_JSON.exists():
        with MODEL_PICKLE.open("rb") as fh:
            classes = sorted(int(c) for c in pickle.load(fh).classes_)
        metadata = {
            "model_name": "cubierta-forestal",
            "dataset_version": "1",
            "trained_at": "20260926T183018Z",
            "features": FEATURES,
            "categorical": CATEGORICAL,
            "target": TARGET,
            "classes": classes,
            "metrics": {"val_accuracy": 0.0, "val_f1_macro": 0.0},
            "estimator": "RandomForestClassifier",
        }
        METADATA_JSON.write_text(json.dumps(metadata, indent=2))
    return MODEL_PICKLE, METADATA_JSON


@pytest.fixture
def bundle() -> minio_client.LoadedModel:
    pkl, meta = _ensure_fixture()
    with pkl.open("rb") as fh:
        model = pickle.load(fh)
    metadata = json.loads(meta.read_text())
    return minio_client.LoadedModel(
        model=model, object_name=OBJECT_NAME, metadata=metadata
    )


@pytest.fixture
def client(bundle: minio_client.LoadedModel) -> TestClient:
    app.dependency_overrides[inference.get_bundle] = lambda: bundle
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def valid_instance() -> dict:
    return {
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
