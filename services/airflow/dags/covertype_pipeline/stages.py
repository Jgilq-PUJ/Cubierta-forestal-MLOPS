"""raw -> processed -> training. Plain functions over a psycopg2 connection."""

from __future__ import annotations

from psycopg2.extras import execute_values

from .schema import (
    COLUMNS,
    FEATURES_TABLE,
    NUMERIC_COLUMNS,
    PROCESSED_TABLE,
    RAW_TABLE,
    TEXT_COLUMNS,
    VERSIONS_TABLE,
)

# Valid ranges of processed.covertype (mirrors its CHECK constraints).
# Note: cover_type is 0-indexed in this Data API (0..6), unlike the UCI 1..7.
_VALID_ROW = """
        aspect BETWEEN 0 AND 360
    AND slope  BETWEEN 0 AND 90
    AND horizontal_distance_to_hydrology   >= 0
    AND horizontal_distance_to_roadways    >= 0
    AND horizontal_distance_to_fire_points >= 0
    AND hillshade_9am  BETWEEN 0 AND 255
    AND hillshade_noon BETWEEN 0 AND 255
    AND hillshade_3pm  BETWEEN 0 AND 255
    AND cover_type BETWEEN 0 AND 6
"""

# Stage 2 · rows of one ingestion run, normalised and validated. Rows outside
# the valid ranges are filtered out (counted as rejected) instead of failing
# the whole insert; ON CONFLICT DO NOTHING collapses exact duplicates (unique
# feature vector index).
PROMOTE_SQL = f"""
INSERT INTO {PROCESSED_TABLE} (
    raw_id, {", ".join(NUMERIC_COLUMNS)}, wilderness_area, soil_type, cover_type, dag_run_id
)
SELECT
    id, {", ".join(NUMERIC_COLUMNS)},
    lower(trim(wilderness_area)), lower(trim(soil_type)), cover_type, dag_run_id
FROM {RAW_TABLE}
WHERE dag_run_id = %(run_id)s AND ({_VALID_ROW})
ORDER BY id
ON CONFLICT DO NOTHING
"""
REJECTED_SQL = f"""
SELECT count(*) FROM {RAW_TABLE}
WHERE dag_run_id = %(run_id)s AND NOT ({_VALID_ROW})
"""

# Stage 3 · u in [0, 1) derived from the first 8 hex chars of md5(id:seed):
# deterministic per row and seed, no session state involved.
FEATURE_COLUMNS = ", ".join(COLUMNS)
SPLIT_SQL = f"""
INSERT INTO {FEATURES_TABLE} (dataset_version, processed_id, split, {FEATURE_COLUMNS})
SELECT
    %(version)s,
    id,
    CASE
        WHEN u < %(train)s            THEN 'train'
        WHEN u < %(train)s + %(val)s  THEN 'validation'
        ELSE 'test'
    END,
    {FEATURE_COLUMNS}
FROM (
    SELECT p.*,
           ('x' || substr(md5(p.id::text || ':' || %(seed)s::text), 1, 8))::bit(32)::bigint
               / 4294967296.0 AS u
    FROM {PROCESSED_TABLE} p
) AS ranked
ORDER BY id
"""


def store_rows(conn, payload: dict, run_id: str) -> int:
    """Stage 1 · append the portion of ``payload`` to raw.covertype.

    Delete-then-insert by run id keeps retries/re-runs idempotent while every
    new run keeps accumulating rows. The processed rows of the same run are
    removed first so the raw delete does not hit the FK; rows already frozen
    in a training dataset make this fail on purpose (a snapshot never changes
    under the model that was trained on it).
    """
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
        f"INSERT INTO {RAW_TABLE} ({FEATURE_COLUMNS}, group_number, batch_number, dag_run_id) "
        "VALUES %s"
    )
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {PROCESSED_TABLE} WHERE dag_run_id = %s", (run_id,))
        cur.execute(f"DELETE FROM {RAW_TABLE} WHERE dag_run_id = %s", (run_id,))
        execute_values(cur, insert_sql, records)
    conn.commit()
    return len(records)


def promote_processed(conn, run_id: str) -> dict:
    """Stage 2 · raw -> processed for one ingestion run. Idempotent."""
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {PROCESSED_TABLE} WHERE dag_run_id = %s", (run_id,))
        cur.execute(PROMOTE_SQL, {"run_id": run_id})
        inserted = cur.rowcount
        cur.execute(REJECTED_SQL, {"run_id": run_id})
        rejected = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM {RAW_TABLE} WHERE dag_run_id = %s", (run_id,))
        raw_rows = cur.fetchone()[0]
    conn.commit()
    return {"raw_rows": raw_rows, "rejected": rejected, "inserted": inserted}


def build_dataset(
    conn,
    version: str,
    run_id: str,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    notes: str | None = None,
) -> dict:
    """Stage 3 · freeze processed.covertype as ``version``. Fails if it exists."""
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError(f"invalid split: train={train_fraction} val={val_fraction}")
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {VERSIONS_TABLE} "
            "(version, dag_run_id, row_count, train_fraction, val_fraction, split_seed, notes) "
            "VALUES (%s, %s, 0, %s, %s, %s, %s)",
            (version, run_id, train_fraction, val_fraction, seed, notes or None),
        )
        cur.execute(
            SPLIT_SQL,
            {"version": version, "train": train_fraction, "val": val_fraction, "seed": seed},
        )
        row_count = cur.rowcount
        cur.execute(
            f"UPDATE {VERSIONS_TABLE} SET row_count = %s WHERE version = %s",
            (row_count, version),
        )
        cur.execute(
            f"SELECT split, count(*) FROM {FEATURES_TABLE} WHERE dataset_version = %s GROUP BY split",
            (version,),
        )
        splits = dict(cur.fetchall())
    conn.commit()
    return {"version": version, "row_count": row_count, "splits": splits}
