# Inference API (s6)

Sirve el modelo de cobertura forestal publicado por el notebook de
entrenamiento en el bucket `models` de MinIO.

El artefacto sigue la convencion de rutas
`<model_name>/<dataset_version>/<trained_at>/` con dos objetos:

- `model.pickle` — un `sklearn.pipeline.Pipeline` (con un `ColumnTransformer`
  adentro) serializado con `pickle`.
- `metadata.json` — lista de `features`, `categorical`, `classes` y `metrics`.

## Configuracion (variables de entorno)

| Variable            | Default              | Descripcion                                             |
| ------------------- | -------------------- | ------------------------------------------------------- |
| `MINIO_ENDPOINT`    | `http://minio:9000`  | Endpoint de MinIO (`http://` / `https://`).             |
| `MINIO_ACCESS_KEY`  | `minioadmin`         | Access key.                                             |
| `MINIO_SECRET_KEY`  | `minioadmin`         | Secret key.                                             |
| `MODELS_BUCKET`     | `models`             | Bucket donde se publican los modelos.                   |
| `MODEL_NAME`        | (vacio)              | Prefijo para resolver el `.pickle` mas reciente.        |
| `MODEL_OBJECT`      | (vacio)              | Fija el objeto exacto; tiene prioridad sobre `MODEL_NAME`. |

El modelo se resuelve una vez y se cachea en memoria. `MODEL_OBJECT` fija el
objeto; si no esta definido, se toma el serializado mas reciente bajo el prefijo
`MODEL_NAME/`.

## Endpoints

- `GET /health` — liveness.
- `GET /models` — lista los objetos del bucket.
- `POST /reload` — recarga el modelo desde MinIO sin reiniciar; devuelve el
  nuevo `trained_at`.
- `POST /predict` — predice sobre una lista de instancias.

### `POST /predict`

El body se valida con Pydantic: 12 campos tipados (enteros para los numericos,
`str` para `wilderness_area` y `soil_type`). Un campo faltante o de tipo
incorrecto responde `422`. Si no hay modelo en el bucket, responde `503`.

Las dos columnas categoricas se normalizan con `trim + lower` antes de predecir
(igual que la etapa `processed`), asi `"  RAWAH "` y `"rawah"` producen la misma
prediccion.

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "instances": [
    {
      "elevation": 2596, "aspect": 51, "slope": 3,
      "horizontal_distance_to_hydrology": 258, "vertical_distance_to_hydrology": 0,
      "horizontal_distance_to_roadways": 510,
      "hillshade_9am": 221, "hillshade_noon": 232, "hillshade_3pm": 148,
      "horizontal_distance_to_fire_points": 6279,
      "wilderness_area": "Rawah", "soil_type": "C2717"
    }
  ]
}'
```

```json
{
  "model": {
    "name": "cubierta-forestal",
    "dataset_version": "1",
    "trained_at": "20260926T183018Z"
  },
  "predictions": [4]
}
```

## Desarrollo local

```bash
cd services/inference-api
uv sync --group api --group test
uv run --group api uvicorn inference_api.main:app --reload   # servidor
uv run --group api --group test pytest                       # tests
```

Los tests cargan el modelo desde un `.pickle` de fixture generado en
`tests/fixtures/` por `conftest.py` (no dependen de MinIO). El test de
integracion opcional contra el MinIO del compose se habilita con
`RUN_MINIO_INTEGRATION=1`.

## En Docker

```bash
docker compose up -d --build inference-api
```
