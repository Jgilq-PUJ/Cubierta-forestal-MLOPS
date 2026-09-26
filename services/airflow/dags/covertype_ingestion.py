"""s1 Data API -> s3 PostgreSQL · incremental ingestion of the covertype dataset.

Every 5 minutes the external Data API serves a random portion of its current
batch (10 batches in total, one per 5-minute window). Each DAG run:

    ensure_table  ->  fetch_batch  ->  store_batch

and appends that portion to ``covertype_raw`` so the table grows run after run.

Runtime settings (Airflow Variables, editable from the UI without a restart,
useful if the Data API has to be redeployed somewhere else):
    data_api_url           default http://10.43.97.110:8080
    data_api_group_number  default 1
"""

from __future__ import annotations

import os
from datetime import timedelta

import pendulum
import psycopg2
import requests
from psycopg2.extras import execute_values

from airflow.exceptions import AirflowSkipException
from airflow.sdk import Variable, dag, task

DEFAULT_DATA_API_URL = "http://10.43.97.110:8080"
DEFAULT_GROUP_NUMBER = 6
TABLE = "covertype_raw"

# Order of the values in every row returned by /data
COLUMNS = [
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
    "wilderness_area",
    "soil_type",
    "cover_type",
]
TEXT_COLUMNS = {"wilderness_area", "soil_type"}

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id                                 BIGSERIAL PRIMARY KEY,
    elevation                          INTEGER NOT NULL,
    aspect                             INTEGER NOT NULL,
    slope                              INTEGER NOT NULL,
    horizontal_distance_to_hydrology   INTEGER NOT NULL,
    vertical_distance_to_hydrology     INTEGER NOT NULL,
    horizontal_distance_to_roadways    INTEGER NOT NULL,
    hillshade_9am                      INTEGER NOT NULL,
    hillshade_noon                     INTEGER NOT NULL,
    hillshade_3pm                      INTEGER NOT NULL,
    horizontal_distance_to_fire_points INTEGER NOT NULL,
    wilderness_area                    TEXT    NOT NULL,
    soil_type                          TEXT    NOT NULL,
    cover_type                         INTEGER NOT NULL,
    group_number                       INTEGER NOT NULL,
    batch_number                       INTEGER NOT NULL,
    dag_run_id                         TEXT    NOT NULL,
    ingested_at                        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {TABLE}_batch_idx ON {TABLE} (group_number, batch_number);
CREATE INDEX IF NOT EXISTS {TABLE}_run_idx ON {TABLE} (dag_run_id);
"""


def _connect():
    """Connection to the s3 business database (env set in docker-compose)."""
    return psycopg2.connect(
        host=os.environ["DATA_DB_HOST"],
        port=os.environ.get("DATA_DB_PORT", "5432"),
        dbname=os.environ["DATA_DB_NAME"],
        user=os.environ["DATA_DB_USER"],
        password=os.environ["DATA_DB_PASSWORD"],
    )


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
    def ensure_table() -> None:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)

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
    def store_batch(payload: dict, dag_run=None) -> int:
        run_id = dag_run.run_id
        records = []
        for row in payload["data"]:
            if len(row) != len(COLUMNS):
                raise ValueError(f"Expected {len(COLUMNS)} values, got {len(row)}: {row}")
            values = [
                value if column in TEXT_COLUMNS else int(value)
                for column, value in zip(COLUMNS, row)
            ]
            records.append((*values, payload["group_number"], payload["batch_number"], run_id))

        insert_sql = (
            f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}, group_number, batch_number, dag_run_id) "
            "VALUES %s"
        )
        # One transaction: delete-then-insert by run id keeps retries/re-runs
        # idempotent while every new run keeps accumulating rows.
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE} WHERE dag_run_id = %s", (run_id,))
            execute_values(cur, insert_sql, records)
            cur.execute(f"SELECT count(*) FROM {TABLE}")
            total = cur.fetchone()[0]

        print(f"inserted={len(records)} batch={payload['batch_number']} total_rows={total}")
        return len(records)

    ensure_table() >> store_batch(fetch_batch())


covertype_ingestion()
