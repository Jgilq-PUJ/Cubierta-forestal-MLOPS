#!/usr/bin/env bash
# Runs the pipeline tests inside the Airflow image against a throwaway Postgres
# initialised with services/postgres/init. No local Python/Airflow needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NET=cf-test-net
PG=cf-test-postgres
cleanup() { docker rm -f "$PG" >/dev/null 2>&1 || true; docker network rm "$NET" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup
docker network create "$NET" >/dev/null
docker run -d --rm --name "$PG" --network "$NET" \
  -e POSTGRES_DB=cubierta_forestal -e POSTGRES_USER=app -e POSTGRES_PASSWORD=app \
  -v "$ROOT/services/postgres/init:/docker-entrypoint-initdb.d:ro" postgres:16 >/dev/null
for _ in $(seq 1 30); do
  docker exec "$PG" pg_isready -U app -d cubierta_forestal -q && break
  sleep 1
done
docker run --rm --network "$NET" \
  -e DATA_DB_HOST="$PG" -e DATA_DB_PORT=5432 -e DATA_DB_NAME=cubierta_forestal \
  -e DATA_DB_USER=app -e DATA_DB_PASSWORD=app \
  -v "$ROOT/services/airflow:/opt/airflow/project:ro" -w /opt/airflow/project \
  apache/airflow:3.3.2 python tests/test_covertype_pipeline.py
