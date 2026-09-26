"""Column contract of the covertype dataset as served by the Data API.

The DDL of the three stages lives in services/postgres/init/001_schemas.sql.
CREATE_SCHEMA_SQL only re-creates the raw table as a safety net for volumes
created before that file existed.
"""

RAW_TABLE = "raw.covertype"
PROCESSED_TABLE = "processed.covertype"
FEATURES_TABLE = "training.covertype_features"
VERSIONS_TABLE = "training.dataset_version"

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
TARGET = "cover_type"
NUMERIC_COLUMNS = [c for c in COLUMNS if c not in TEXT_COLUMNS and c != TARGET]

CREATE_SCHEMA_SQL = f"""
CREATE SCHEMA IF NOT EXISTS raw;
CREATE TABLE IF NOT EXISTS {RAW_TABLE} (
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
CREATE INDEX IF NOT EXISTS covertype_batch_idx ON {RAW_TABLE} (group_number, batch_number);
CREATE INDEX IF NOT EXISTS covertype_run_idx   ON {RAW_TABLE} (dag_run_id);
"""
