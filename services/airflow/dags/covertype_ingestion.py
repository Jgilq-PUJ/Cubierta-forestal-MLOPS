"""s1 Data API -> s3 PostgreSQL · incremental ingestion of the covertype dataset.

Every 5 minutes the external Data API serves a random portion of its current
batch (10 batches in total, one per 5-minute window). Each DAG run:

    ensure_schema  ->  fetch_batch  ->  store_batch  ->  promote

and appends that portion to ``raw.covertype`` (stage 1), then promotes the
rows of this run into ``processed.covertype`` (stage 2: typed, normalised,
deduplicated). Stage 3 (``training.*``) is built on demand by the
``covertype_training_dataset`` DAG.

Runtime settings (Airflow Variables, editable from the UI without a restart,
useful if the Data API has to be redeployed somewhere else):
    data_api_url           default http://10.43.97.110:8080
    data_api_group_number  default 6
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
import requests
from airflow.sdk import Variable, dag, task
from airflow.sdk.exceptions import AirflowSkipException

from covertype_pipeline.db import connect
from covertype_pipeline.schema import CREATE_SCHEMA_SQL, RAW_TABLE
from covertype_pipeline.stages import promote_processed, store_rows

DEFAULT_DATA_API_URL = "http://10.43.97.110:8080"
DEFAULT_GROUP_NUMBER = 6


@dag(
    dag_id="covertype_ingestion",
    description="Pulls a random portion of the current Data API batch every 5 minutes into Postgres",
    schedule="*/5 * * * *",
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    # A run must not overlap the next 5-minute window
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=5),
    default_args={
        "owner": "cubierta-forestal",
        # The VM Data API may go down and be redeployed: 3 retries, 20 s apart.
        # Worst case per task: 4 attempts x 10 s timeout + 3 x 20 s = ~100 s,
        # well inside the 5-minute window of the current batch.
        "retries": 3,
        "retry_delay": timedelta(seconds=20),
        "execution_timeout": timedelta(minutes=1),
    },
    tags=["ingestion", "s1", "s3"],
)
def covertype_ingestion():
    @task
    def ensure_schema() -> None:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(CREATE_SCHEMA_SQL)

    @task
    def fetch_batch() -> dict:
        base_url = Variable.get("data_api_url", default=DEFAULT_DATA_API_URL).rstrip("/")
        group = int(Variable.get("data_api_group_number", default=DEFAULT_GROUP_NUMBER))

        # Connection errors, timeouts and 5xx raise -> Airflow retries the task
        response = requests.get(
            f"{base_url}/data", params={"group_number": group}, timeout=10
        )

        # 400 means the group already collected its 10 batches (or is invalid):
        # retrying will not change that, so skip instead of failing.
        if response.status_code == 400:
            raise AirflowSkipException(f"Data API returned 400: {response.text}")
        response.raise_for_status()

        payload = response.json()
        rows = payload["data"]
        if not rows:
            raise AirflowSkipException("Data API returned an empty portion")

        print(f"group={payload['group_number']} batch={payload['batch_number']} rows={len(rows)}")
        return payload

    @task
    def store_batch(payload: dict, dag_run=None) -> str:
        run_id = dag_run.run_id
        with connect() as conn:
            inserted = store_rows(conn, payload, run_id)
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {RAW_TABLE}")
                total = cur.fetchone()[0]
        print(f"inserted={inserted} batch={payload['batch_number']} total_rows={total}")
        return run_id

    @task
    def promote(run_id: str) -> dict:
        with connect() as conn:
            stats = promote_processed(conn, run_id)
        print(f"processed run={run_id} {stats}")
        return stats

    promote(ensure_schema() >> store_batch(fetch_batch()))


covertype_ingestion()
