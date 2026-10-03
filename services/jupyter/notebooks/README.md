# Notebooks · Jupyter (parte derecha de la arquitectura)

Esta carpeta contiene el trabajo de entrenamiento que lee de **PostgreSQL** y prepara artefactos
para las etapas posteriores. **No** implementa MLflow, MinIO, FastAPI, Model Registry ni la API de
inferencia (los integran otros componentes).

## Notebooks

| Notebook | Propósito | Dependencias |
|---|---|---|
| `cubierta-forestal.ipynb` | Flujo mínimo existente: conexión → `training.covertype_features` → pipeline scikit-learn (RandomForest) → publicación a MinIO. **No modificado.** | entorno base del proyecto |
| `cubierta_forestal_experimentos.ipynb` | **Nuevo.** Flujo reproducible completo con **AutoGluon** restringido a **una sola familia de modelo** (LightGBM/`GBM`): validación de datos → Gold Layer → EDA → feature engineering → train/val/test → **25 configuraciones de hiperparámetros** → **25 experimentos** → métricas → `results_df` consolidado → artefactos → preparado para MLflow. | base + `requirements-training.txt` |

## Cómo ejecutar el notebook de experimentación

1. PostgreSQL del proyecto en marcha (`docker compose up postgres`) con datos ingeridos por los
   DAGs de Airflow (esquemas `raw` → `processed` → `training`).
2. Variables de entorno de conexión disponibles (ya inyectadas en el contenedor `jupyter`; en
   local, vía un archivo `.env` — ver sección 03 del notebook). **Nunca** se escriben credenciales
   en el código.
3. Dependencias de entrenamiento:
   ```bash
   pip install -r services/jupyter/requirements-training.txt
   ```
   (El notebook también las instala de forma idempotente en su celda de *bootstrap*.)
4. Ejecutar todas las celdas en orden. Para una prueba rápida del flujo, poner `QUICK_SMOKE = True`
   en la sección 14.

## Salidas (en `artifacts/`, relativo al directorio de trabajo del notebook)

```
artifacts/
├── configurations.json          # las 25 configuraciones (trazabilidad)
├── environment.json             # versiones utilizadas (reproducibilidad)
├── selected_model.json          # configuración ganadora + métricas de test
├── eda_overview.png             # (si matplotlib/seaborn disponibles)
├── comparison.png
├── models/
│   ├── config_01/ ... config_25/   # un TabularPredictor por configuración
├── results/
│   ├── results.csv              # tabla consolidada (25 filas)
│   └── results.parquet
└── mlflow_payloads/
    └── config_01.json ... config_25.json   # params + metrics + rutas, listos para MLflow
```

## Preparación para MLflow

Cada experimento deja un *payload* (`mlflow_payloads/config_XX.json`) con `params`, `metrics`,
`tags` y rutas de artefactos, y el modelo en `models/config_XX/`. La sección 20 del notebook
muestra el bucle `mlflow.start_run()` / `log_params` / `log_metrics` / `log_artifact` /
`log_model` que otro componente añadirá, sin ejecutarlo aquí.
