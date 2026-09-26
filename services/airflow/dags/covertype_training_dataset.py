"""s3 PostgreSQL · stage 2 -> stage 3: freeze a versioned training dataset.

Manual DAG (no schedule). One run takes everything in ``processed.covertype``
and writes an immutable snapshot into ``training.covertype_features`` under a
new ``dataset_version``, with the train/validation/test split decided here,
once, so every experiment on that version sees the same partitions.

Trigger with params (Airflow UI -> Trigger DAG w/ config):
    version         default "v<run timestamp>"
    train_fraction  default 0.70
    val_fraction    default 0.15   (test = 1 - train - val)
    seed            default 42
    notes           optional free text stored with the version
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.sdk import Param, dag, task

from covertype_pipeline.db import connect
from covertype_pipeline.stages import build_dataset


@dag(
    dag_id="covertype_training_dataset",
    description="Freezes processed.covertype into a versioned, split training dataset",
    schedule=None,
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    params={
        "version": Param("", type="string", description="Empty = v<run timestamp>"),
        "train_fraction": Param(0.70, type="number", minimum=0.01, maximum=0.99),
        "val_fraction": Param(0.15, type="number", minimum=0.0, maximum=0.99),
        "seed": Param(42, type="integer"),
        "notes": Param("", type="string"),
    },
    default_args={
        "owner": "cubierta-forestal",
        "retries": 0,
        "execution_timeout": timedelta(minutes=10),
    },
    tags=["training", "s3"],
)
def covertype_training_dataset():
    @task
    def freeze(params: dict | None = None, dag_run=None) -> dict:
        version = params["version"] or f"v{dag_run.run_after.strftime('%Y%m%dT%H%M%S')}"
        with connect() as conn:
            stats = build_dataset(
                conn,
                version,
                dag_run.run_id,
                float(params["train_fraction"]),
                float(params["val_fraction"]),
                int(params["seed"]),
                notes=params["notes"] or None,
            )
        print(f"dataset {stats}")
        return stats

    freeze()


covertype_training_dataset()
