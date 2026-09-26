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

## Inference API (s6)

Documentacion interactiva (Swagger) en http://localhost:8000/docs.

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| `GET`  | `/health`  | Liveness del servicio |
| `GET`  | `/models`  | Lista los objetos del bucket `models` |
| `POST` | `/reload`  | Recarga el modelo mas reciente sin reiniciar |
| `POST` | `/predict` | Predice `cover_type` para una o varias instancias |

### Contrato de features

La Inference API recibe `instances: list[list[float]]`. El vector de entrada del
modelo tiene **12 posiciones** en el orden de `training.FEATURE_ORDER`:

| Pos | Campo | Pos | Campo |
|-----|-------|-----|-------|
| 1 | elevation | 7 | hillshade_9am |
| 2 | aspect | 8 | hillshade_noon |
| 3 | slope | 9 | hillshade_3pm |
| 4 | horizontal_distance_to_hydrology | 10 | horizontal_distance_to_fire_points |
| 5 | vertical_distance_to_hydrology | 11 | **wilderness_area** (codigo entero) |
| 6 | horizontal_distance_to_roadways | 12 | **soil_type** (codigo entero) |

El mapeo categoria→codigo (posiciones 11 y 12) se guarda en los metadatos del
modelo (`<objeto>.metadata.json`) para reconstruir el vector en el cliente.

Ejemplo de prediccion:

```bash
curl -s http://localhost:8000/models          # lista los modelos del bucket
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[2596,51,3,258,0,510,221,232,148,6279,0,0]]}'
```

Respuesta:

```json
{ "model": "cubierta-forestal/covertype-rf-<ts>.joblib", "predictions": [5] }
```

El valor de `predictions` es la clase de cobertura forestal (1–7):

| Clase | Cobertura | Clase | Cobertura |
|-------|-----------|-------|-----------|
| 1 | Spruce/Fir | 5 | Aspen |
| 2 | Lodgepole Pine | 6 | Douglas-fir |
| 3 | Ponderosa Pine | 7 | Krummholz |
| 4 | Cottonwood/Willow | | |

> En PowerShell `curl` es un alias de `Invoke-WebRequest`; usa `curl.exe` o, mejor,
> `Invoke-RestMethod` con `ConvertTo-Json -Depth 5` para el `POST /predict`.

## Verificacion end-to-end

Flujo completo validado sobre el stack en ejecucion:

```
s2 Airflow ingesta   -> covertype_raw = 17.430 filas (s3 Postgres)
s4 Jupyter entrena   -> accuracy 0.9395 · f1_macro 0.9087
s4 -> s5 publica     -> models/cubierta-forestal/covertype-rf-<ts>.joblib
s6 /predict          -> 200 OK · predictions: [5]
```

## Pruebas

```bash
cd cubierta-forestal
uv run --group test pytest
```

Las pruebas de `tests/test_training.py` validan la logica de entrenamiento sin
necesidad de Postgres ni MinIO (incluye el round-trip de serializacion y el
contrato de `list[list[float]]`).
