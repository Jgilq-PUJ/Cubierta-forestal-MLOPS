"""s1 Data API -> s3 PostgreSQL · incremental ingestion of the covertype dataset.

Every 5 minutes the external Data API serves a random portion of its current
batch (10 batches in total, one per 5-minute window). Each DAG run does the
WHOLE process for exactly ONE request (the project forbids N requests per run):

    ensure_tables  ->  fetch_batch  ->  store_raw  ->  process_batch

populating the three data stages PostgreSQL must expose (project requirement:
"debe tener multiples etapas, almacenando informacion sin procesar, procesada y
lista para entrenamiento"):

    covertype_raw       (sin procesar)  raw rows exactly as the API returns them
    covertype_processed (procesada)     cleaned + typed + deduplicated rows
    covertype_training  (entrenamiento) VIEW with the all-numeric feature matrix
                                        (categoricals encoded) ready for Jupyter

Runtime settings (Airflow Variables, editable from the UI without a restart,
useful if the Data API has to be redeployed somewhere else):
    data_api_url           default http://10.43.97.110:8080
    data_api_group_number  default 6
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

# The three data stages the project requires PostgreSQL to expose.
TABLE = "covertype_raw"              # sin procesar
PROCESSED_TABLE = "covertype_processed"  # procesada
TRAINING_VIEW = "covertype_training"     # lista para entrenamiento

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

# Stage 2 · procesada: cleaned, typed and deduplicated rows. A UNIQUE row_hash
# drops exact-duplicate feature vectors across runs (ON CONFLICT DO NOTHING).
CREATE_PROCESSED_SQL = f"""
CREATE TABLE IF NOT EXISTS {PROCESSED_TABLE} (
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
    cover_type                         INTEGER NOT NULL CHECK (cover_type BETWEEN 1 AND 7),
    batch_number                       INTEGER NOT NULL,
    row_hash                           TEXT    NOT NULL UNIQUE,
    processed_at                       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {PROCESSED_TABLE}_batch_idx ON {PROCESSED_TABLE} (batch_number);
"""

# Feature columns fed to the model, in the exact order the Inference API expects.
FEATURE_COLUMNS = [c for c in COLUMNS if c != "cover_type" and c not in TEXT_COLUMNS]

# Stage 3 · lista para entrenamiento: all-numeric feature matrix + target.
# The two categoricals become 0-based integer codes over the whole processed
# table (dense_rank over the sorted distinct values) so the encoding is stable
# and identical to what training.build_feature_matrix produces (sorted -> index).
CREATE_TRAINING_VIEW_SQL = f"""
CREATE OR REPLACE VIEW {TRAINING_VIEW} AS
WITH wa AS (
    SELECT wilderness_area,
           dense_rank() OVER (ORDER BY wilderness_area) - 1 AS code
    FROM (SELECT DISTINCT wilderness_area FROM {PROCESSED_TABLE}) s
),
st AS (
    SELECT soil_type,
           dense_rank() OVER (ORDER BY soil_type) - 1 AS code
    FROM (SELECT DISTINCT soil_type FROM {PROCESSED_TABLE}) s
)
SELECT p.{', p.'.join(FEATURE_COLUMNS)},
       wa.code AS wilderness_area_code,
       st.code AS soil_type_code,
       p.cover_type
FROM {PROCESSED_TABLE} p
JOIN wa ON p.wilderness_area = wa.wilderness_area
JOIN st ON p.soil_type = st.soil_type;
"""

# INSERT ... SELECT that promotes one run's raw rows into the processed stage.
_HASH_EXPR = "md5(concat_ws('|', " + ", ".join(COLUMNS) + "))"
PROCESS_SQL = f"""
INSERT INTO {PROCESSED_TABLE} (
    {', '.join(COLUMNS)}, batch_number, row_hash
)
SELECT {', '.join(COLUMNS)}, batch_number, {_HASH_EXPR}
FROM {TABLE}
WHERE dag_run_id = %s
  AND cover_type BETWEEN 1 AND 7
  AND wilderness_area <> ''
  AND soil_type <> ''
ON CONFLICT (row_hash) DO NOTHING;
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
    def ensure_tables() -> None:
        """Create the three data stages (raw table, processed table, training view)."""
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_PROCESSED_SQL)
            cur.execute(CREATE_TRAINING_VIEW_SQL)

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
    def store_raw(payload: dict, dag_run=None) -> str:
        """Stage 1 · sin procesar: append this run's portion to covertype_raw."""
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

        print(f"raw inserted={len(records)} batch={payload['batch_number']} total_raw={total}")
        return run_id

    @task
    def process_batch(run_id: str) -> int:
        """Stage 2 · procesada: clean + dedup this run's raw rows into the processed table.

        Same DAG run, no extra API request: only transforms what store_raw wrote.
        Invalid rows (bad cover_type or empty categoricals) are dropped and exact
        duplicates are ignored via the UNIQUE row_hash.
        """
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(PROCESS_SQL, (run_id,))
            inserted = cur.rowcount
            cur.execute(f"SELECT count(*) FROM {PROCESSED_TABLE}")
            total = cur.fetchone()[0]
            cur.execute(f"SELECT count(*) FROM {TRAINING_VIEW}")
            trainable = cur.fetchone()[0]

        print(f"processed new={inserted} total_processed={total} trainable_rows={trainable}")
        return inserted

    ensure_tables() >> process_batch(store_raw(fetch_batch()))


covertype_ingestion()
