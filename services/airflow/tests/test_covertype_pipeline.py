"""Integration tests for the raw -> processed -> training SQL helpers.

They run against a real PostgreSQL initialised with services/postgres/init
(the DDL is the contract). Environment: DATA_DB_HOST/PORT/NAME/USER/PASSWORD.

Run inside the Airflow image (no pytest there):

    python tests/test_covertype_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dags"))

from covertype_pipeline.db import connect  # noqa: E402
from covertype_pipeline.schema import COLUMNS  # noqa: E402
from covertype_pipeline.stages import build_dataset, promote_processed, store_rows  # noqa: E402

BASE_ROW = [2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279, "Rawah", "C2717", 5]


def _row(**overrides) -> list:
    row = dict(zip(COLUMNS, BASE_ROW))
    row.update(overrides)
    return [row[c] for c in COLUMNS]


def _payload(rows: list[list], batch: int = 0) -> dict:
    return {"group_number": 6, "batch_number": batch, "data": rows}


def _count(conn, table: str, where: str = "TRUE", params=()) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table} WHERE {where}", params)
        return cur.fetchone()[0]


def _reset(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE training.covertype_features, training.dataset_version, "
            "processed.covertype, raw.covertype RESTART IDENTITY"
        )
    conn.commit()


def test_store_rows_is_idempotent_per_run(conn) -> None:
    payload = _payload([_row(), _row(elevation=2600)])
    store_rows(conn, payload, "run-a")
    store_rows(conn, payload, "run-a")
    assert _count(conn, "raw.covertype") == 2
    assert _count(conn, "raw.covertype", "dag_run_id = %s", ("run-a",)) == 2


def test_promote_processed_normalises_dedups_and_rejects(conn) -> None:
    payload = _payload(
        [
            _row(wilderness_area="  Rawah "),   # kept, normalised
            _row(wilderness_area="rawah"),      # exact duplicate after normalisation
            _row(elevation=2700),               # kept
            _row(hillshade_noon=999),           # out of range -> rejected
            _row(cover_type=0),                 # classes are 0..6 in this API -> kept
            _row(cover_type=7),                 # -> rejected
        ]
    )
    store_rows(conn, payload, "run-b")
    stats = promote_processed(conn, "run-b")
    stats = promote_processed(conn, "run-b")  # idempotent re-run

    assert stats == {"raw_rows": 6, "rejected": 2, "inserted": 3}
    assert _count(conn, "processed.covertype", "dag_run_id = %s", ("run-b",)) == 3
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT wilderness_area FROM processed.covertype")
        assert cur.fetchall() == [("rawah",)]


def test_build_dataset_versions_and_splits_deterministically(conn) -> None:
    rows = [_row(elevation=2000 + i) for i in range(200)]
    store_rows(conn, _payload(rows), "run-c")
    promote_processed(conn, "run-c")

    first = build_dataset(conn, "v1", "train-run-1", 0.7, 0.15, 42)
    second = build_dataset(conn, "v2", "train-run-2", 0.7, 0.15, 42)

    assert first["row_count"] == 200
    assert sum(first["splits"].values()) == 200
    assert set(first["splits"]) == {"train", "validation", "test"}
    assert 120 <= first["splits"]["train"] <= 160

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM training.covertype_features a "
            "JOIN training.covertype_features b USING (processed_id) "
            "WHERE a.dataset_version = 'v1' AND b.dataset_version = 'v2' AND a.split <> b.split"
        )
        assert cur.fetchone()[0] == 0, "same seed must give the same split"

    try:
        build_dataset(conn, "v1", "train-run-3", 0.7, 0.15, 42)
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
    else:
        raise AssertionError("a dataset version must be immutable")


def test_raw_rows_frozen_by_a_dataset_cannot_be_rewritten(conn) -> None:
    store_rows(conn, _payload([_row()]), "run-d")
    promote_processed(conn, "run-d")
    build_dataset(conn, "v1", "train-run", 0.7, 0.15, 42)

    try:
        store_rows(conn, _payload([_row()]), "run-d")
    except psycopg2.errors.ForeignKeyViolation:
        conn.rollback()
    else:
        raise AssertionError("re-ingesting a run used by a dataset must fail loudly")


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for test in tests:
        conn = connect()
        try:
            _reset(conn)
            test(conn)
            conn.commit()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        finally:
            conn.close()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
