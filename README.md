# Cubierta-forestal-MLOPS

Plataforma MLOps para clasificar el tipo de cobertura forestal (dataset
*Covertype*) siguiendo la arquitectura del enunciado. Todo el stack se levanta
con un unico `docker-compose`.

## Arquitectura

```
s1 Data API (externa, via VPN)  --->  s2 Airflow
s2 Airflow                     <--->  s3 PostgreSQL        (tabla covertype_raw)
s3 PostgreSQL                   --->  s4 Jupyter           (entrenamiento)
s4 Jupyter                      --->  s5 MinIO             (registro de modelos)
s5 MinIO                        --->  s6 Inference API     (sirve el modelo)
```

| Servicio | Rol | Puerto | Implementacion |
|----------|-----|--------|----------------|
| **s1** Data API | API externa del dataset (fuera del compose, se alcanza por VPN) | — | consumida por el DAG |
| **s2** Airflow | Ingesta incremental cada 5 min | 8080 | `services/airflow/dags/covertype_ingestion.py` |
| **s3** PostgreSQL | Base de negocio con `covertype_raw` | 5432 | servicio `postgres` |
| **s4** Jupyter | Entrena y publica el modelo | 8888 | `services/jupyter/notebooks/train_covertype.ipynb` |
| **s5** MinIO | Almacen de artefactos / modelos | 9000/9001 | servicio `minio`, bucket `models` |
| **s6** Inference API | Sirve el modelo publicado | 8000 | `cubierta_forestal.infrastructure.api` |

El paquete `cubierta-forestal` (en `cubierta-forestal/`) sigue una estructura
hexagonal y es compartido por Jupyter (s4) y la Inference API (s6). Ambos se
construyen desde el **mismo** `Dockerfile` / `uv.lock` (targets `jupyter` y
`api`) para que la version de scikit-learn que serializa el modelo sea la misma
que lo deserializa.

## Puesta en marcha

```bash
cp .env.example .env          # ajusta credenciales/puertos si hace falta
docker compose build          # construye los targets jupyter y api
docker compose up -d
```

1. **Airflow (s2)** — http://localhost:8080 (`airflow` / `airflow`). Activa el DAG
   `covertype_ingestion`; cada 5 minutos agrega una porcion del batch actual a
   `covertype_raw` en Postgres (s3).
2. **Jupyter (s4)** — http://localhost:8888 (token `cubierta`). Abre
   `notebooks/train_covertype.ipynb` y ejecutalo: lee Postgres, entrena un
   `RandomForestClassifier` y publica el modelo en MinIO (s5).
3. **MinIO (s5)** — consola en http://localhost:9001 (`minioadmin` /
   `minioadmin`). El modelo queda en el bucket `models`.
4. **Inference API (s6)** — http://localhost:8000/docs. Toma automaticamente el
   objeto mas reciente del bucket.

### Flujo de entrenamiento (s3 → s4 → s5)

El notebook es una capa fina sobre el paquete; la logica reutilizable vive en:

- `application/training.py` — entrenamiento puro (features, encoding, metricas).
- `infrastructure/postgres_client.py` — lectura de `covertype_raw` (s3).
- `infrastructure/minio_client.py` — `publish_model()` (s4→s5) y `load_model()` (s5→s6).
- `application/train_pipeline.py` — orquesta leer → entrenar → publicar:

```bash
# Dentro del contenedor de Jupyter (o cualquier entorno con el paquete):
python -m cubierta_forestal.application.train_pipeline
```

### Contrato de features con la Inference API

La Inference API recibe `instances: list[list[float]]`. El vector de entrada del
modelo tiene 12 posiciones en el orden de `training.FEATURE_ORDER`: las 10
columnas numericas del dataset, seguidas del codigo entero de `wilderness_area`
y de `soil_type`. El mapeo categoria→codigo se guarda en los metadatos del
modelo (`<objeto>.metadata.json`) para poder reconstruir el vector en el cliente.

Ejemplo de prediccion:

```bash
curl -s http://localhost:8000/models          # lista los modelos del bucket
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[2596,51,3,258,0,510,221,232,148,6279,0,0]]}'
```

## Pruebas

```bash
cd cubierta-forestal
uv run --group test pytest
```

Las pruebas de `tests/test_training.py` validan la logica de entrenamiento sin
necesidad de Postgres ni MinIO (incluye el round-trip de serializacion y el
contrato de `list[list[float]]`).
