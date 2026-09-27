-- Cubierta Forestal · data stages (medallion layout)
--
--   raw        -> rows exactly as returned by the external Data API (Airflow writes)
--   processed  -> cleaned / typed / deduplicated rows              (Airflow writes)
--   training   -> versioned, split datasets ready for training     (Airflow writes, Jupyter reads)
--
-- Applied once by the postgres image entrypoint when the data volume is empty.
-- Every statement is idempotent so the file can also be re-applied by hand.

BEGIN;

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS processed;
CREATE SCHEMA IF NOT EXISTS training;

COMMENT ON SCHEMA raw       IS 'Stage 1: unprocessed rows from the external Data API';
COMMENT ON SCHEMA processed IS 'Stage 2: cleaned, typed and deduplicated rows';
COMMENT ON SCHEMA training  IS 'Stage 3: versioned datasets with train/validation/test split';

-- --------------------------------------------------------------------------
-- Stage 1 · raw
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.covertype (
    id                                 BIGSERIAL PRIMARY KEY,
    elevation                          INTEGER     NOT NULL,
    aspect                             INTEGER     NOT NULL,
    slope                              INTEGER     NOT NULL,
    horizontal_distance_to_hydrology   INTEGER     NOT NULL,
    vertical_distance_to_hydrology     INTEGER     NOT NULL,
    horizontal_distance_to_roadways    INTEGER     NOT NULL,
    hillshade_9am                      INTEGER     NOT NULL,
    hillshade_noon                     INTEGER     NOT NULL,
    hillshade_3pm                      INTEGER     NOT NULL,
    horizontal_distance_to_fire_points INTEGER     NOT NULL,
    wilderness_area                    TEXT        NOT NULL,
    soil_type                          TEXT        NOT NULL,
    cover_type                         INTEGER     NOT NULL,
    -- ingestion lineage
    group_number                       INTEGER     NOT NULL,
    batch_number                       INTEGER     NOT NULL,
    dag_run_id                         TEXT        NOT NULL,
    ingested_at                        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS covertype_batch_idx ON raw.covertype (group_number, batch_number);
CREATE INDEX IF NOT EXISTS covertype_run_idx   ON raw.covertype (dag_run_id);

-- --------------------------------------------------------------------------
-- Stage 2 · processed
-- Same grain as raw (one row per observation) but cleaned:
--   * numeric ranges validated by CHECK constraints
--   * categorical text normalised (trimmed, lower case)
--   * exact duplicates collapsed (unique index on the feature vector)
--   * raw_id keeps the link back to the source row for auditing
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS processed.covertype (
    id                                 BIGSERIAL PRIMARY KEY,
    raw_id                             BIGINT      NOT NULL REFERENCES raw.covertype (id),
    elevation                          INTEGER     NOT NULL,
    aspect                             SMALLINT    NOT NULL CHECK (aspect BETWEEN 0 AND 360),
    slope                              SMALLINT    NOT NULL CHECK (slope BETWEEN 0 AND 90),
    horizontal_distance_to_hydrology   INTEGER     NOT NULL CHECK (horizontal_distance_to_hydrology >= 0),
    vertical_distance_to_hydrology     INTEGER     NOT NULL,
    horizontal_distance_to_roadways    INTEGER     NOT NULL CHECK (horizontal_distance_to_roadways >= 0),
    hillshade_9am                      SMALLINT    NOT NULL CHECK (hillshade_9am  BETWEEN 0 AND 255),
    hillshade_noon                     SMALLINT    NOT NULL CHECK (hillshade_noon BETWEEN 0 AND 255),
    hillshade_3pm                      SMALLINT    NOT NULL CHECK (hillshade_3pm  BETWEEN 0 AND 255),
    horizontal_distance_to_fire_points INTEGER     NOT NULL CHECK (horizontal_distance_to_fire_points >= 0),
    wilderness_area                    TEXT        NOT NULL,
    soil_type                          TEXT        NOT NULL,
    cover_type                         SMALLINT    NOT NULL CHECK (cover_type BETWEEN 0 AND 6),
    -- lineage
    dag_run_id                         TEXT        NOT NULL,
    processed_at                       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS covertype_raw_id_idx ON processed.covertype (raw_id);
CREATE INDEX IF NOT EXISTS covertype_processed_run_idx ON processed.covertype (dag_run_id);
CREATE UNIQUE INDEX IF NOT EXISTS covertype_feature_vector_idx ON processed.covertype (
    elevation, aspect, slope,
    horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
    horizontal_distance_to_roadways,
    hillshade_9am, hillshade_noon, hillshade_3pm,
    horizontal_distance_to_fire_points,
    wilderness_area, soil_type, cover_type
);

-- --------------------------------------------------------------------------
-- Stage 3 · training
-- One immutable snapshot per dataset_version. The split is decided here, once,
-- so every experiment trained on the same version sees the same partitions.
-- Categorical columns stay as text: encoding belongs to the model pipeline so
-- the inference API applies exactly the same transformation.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training.dataset_version (
    version        TEXT        PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    dag_run_id     TEXT        NOT NULL,
    row_count      INTEGER     NOT NULL CHECK (row_count >= 0),
    train_fraction NUMERIC(4,3) NOT NULL CHECK (train_fraction > 0 AND train_fraction < 1),
    val_fraction   NUMERIC(4,3) NOT NULL CHECK (val_fraction  >= 0 AND val_fraction  < 1),
    split_seed     INTEGER     NOT NULL,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS training.covertype_features (
    id                                 BIGSERIAL   PRIMARY KEY,
    dataset_version                    TEXT        NOT NULL REFERENCES training.dataset_version (version),
    processed_id                       BIGINT      NOT NULL REFERENCES processed.covertype (id),
    split                              TEXT        NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    elevation                          INTEGER     NOT NULL,
    aspect                             SMALLINT    NOT NULL,
    slope                              SMALLINT    NOT NULL,
    horizontal_distance_to_hydrology   INTEGER     NOT NULL,
    vertical_distance_to_hydrology     INTEGER     NOT NULL,
    horizontal_distance_to_roadways    INTEGER     NOT NULL,
    hillshade_9am                      SMALLINT    NOT NULL,
    hillshade_noon                     SMALLINT    NOT NULL,
    hillshade_3pm                      SMALLINT    NOT NULL,
    horizontal_distance_to_fire_points INTEGER     NOT NULL,
    wilderness_area                    TEXT        NOT NULL,
    soil_type                          TEXT        NOT NULL,
    cover_type                         SMALLINT    NOT NULL,
    UNIQUE (dataset_version, processed_id)
);
CREATE INDEX IF NOT EXISTS covertype_features_version_split_idx
    ON training.covertype_features (dataset_version, split);

COMMIT;
