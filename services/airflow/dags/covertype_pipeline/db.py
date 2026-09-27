import os

import psycopg2


def connect():
    """Connection to the s3 business database (env set in docker-compose)."""
    return psycopg2.connect(
        host=os.environ["DATA_DB_HOST"],
        port=os.environ.get("DATA_DB_PORT", "5432"),
        dbname=os.environ["DATA_DB_NAME"],
        user=os.environ["DATA_DB_USER"],
        password=os.environ["DATA_DB_PASSWORD"],
    )
