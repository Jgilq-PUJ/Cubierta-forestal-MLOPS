"""Pruebas de la logica de entrenamiento (sin Postgres ni MinIO)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cubierta_forestal.application import training


def _synthetic_dataframe(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Dataset sintetico con el mismo esquema que ``covertype_raw``."""
    rng = np.random.default_rng(seed)
    data = {col: rng.integers(0, 4000, size=n) for col in training.NUMERIC_FEATURES}
    data["wilderness_area"] = rng.choice(["Rawah", "Neota", "Comanche"], size=n)
    data["soil_type"] = rng.choice([f"C{i}" for i in range(1, 6)], size=n)
    # Objetivo correlacionado con elevation para que el modelo aprenda algo.
    data["cover_type"] = (data["elevation"] // 800 % 7) + 1
    return pd.DataFrame(data)


def test_feature_order_matches_inference_contract():
    # 10 numericas + 2 categoricas, en el orden esperado por el vector de floats.
    assert training.FEATURE_ORDER == training.NUMERIC_FEATURES + training.CATEGORICAL_FEATURES
    assert len(training.FEATURE_ORDER) == 12


def test_train_model_produces_usable_pipeline():
    df = _synthetic_dataframe()
    trained = training.train_model(df, test_size=0.25, random_state=1)

    assert set(trained.metrics) >= {"accuracy", "f1_macro", "n_train", "n_test"}
    assert 0.0 <= trained.metrics["accuracy"] <= 1.0
    assert trained.n_samples == len(df)

    meta = trained.metadata()
    assert meta["feature_order"] == training.FEATURE_ORDER
    assert set(meta["categories"]) == set(training.CATEGORICAL_FEATURES)


def test_pipeline_predicts_on_plain_float_matrix():
    """El modelo publicado debe aceptar ``list[list[float]]`` como la s6."""
    df = _synthetic_dataframe()
    trained = training.train_model(df)

    # Reconstruir el vector numerico igual que lo haria el cliente de la API.
    X = training.build_feature_matrix(df.head(3), trained.categories)
    instances = X.tolist()  # list[list[float]]
    preds = trained.pipeline.predict(np.asarray(instances, dtype=float))

    assert len(preds) == 3


def test_unknown_category_is_ignored():
    df = _synthetic_dataframe()
    trained = training.train_model(df)

    unseen = df.head(2).copy()
    unseen["soil_type"] = "SOIL_NUNCA_VISTO"
    X = training.build_feature_matrix(unseen, trained.categories)
    # -1 marca la categoria desconocida; OneHotEncoder(handle_unknown='ignore').
    assert (X[:, training.FEATURE_ORDER.index("soil_type")] == -1).all()
    preds = trained.pipeline.predict(X)
    assert len(preds) == 2
