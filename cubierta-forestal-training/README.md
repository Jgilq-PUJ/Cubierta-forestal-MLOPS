# Cubierta Forestal · Notebook de entrenamiento (autónomo)

Flujo reproducible de Machine Learning para la **parte derecha** de la arquitectura MLOps:
PostgreSQL *(opcional)* → Gold Layer → EDA → Feature Engineering → Train/Val/Test →
**un solo tipo de modelo (AutoGluon · LightGBM)** → **25 configuraciones** → **25 experimentos** →
métricas → tabla consolidada → **preparado para MLflow**.

> **No implementa** MLflow, MinIO, FastAPI, Model Registry ni la API de inferencia: esos
> componentes se integran después. Este notebook deja listos los artefactos y la estructura.

## Por qué es "autónomo"

No depende de la infraestructura de ningún otro repositorio. La carga de datos
(`load_covertype()`):

1. intenta **PostgreSQL** si defines las variables de entorno y existe la tabla, y
2. si no hay base, **descarga el dataset oficial Covertype** con
   `sklearn.datasets.fetch_covtype` (reconstruyendo el esquema: `wilderness_area`/`soil_type`
   categóricas y `cover_type` en rango 0–6).

Así puedes ejecutarlo tal cual, en local con VS Code, sin más preparación que instalar las
dependencias.

## Contenido

```
cubierta-forestal-training/
├── cubierta_forestal_experimentos.ipynb   # notebook de 20 secciones
├── requirements.txt                        # dependencias
├── .env.example                            # variables de PostgreSQL (opcional)
├── .gitignore
└── README.md
```

## Cómo ejecutarlo en local (VS Code)

1. Abre la carpeta en VS Code (extensiones *Python* + *Jupyter*).
2. Crea y activa un entorno (recomendado Python 3.11/3.12):
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. *(Opcional)* Si vas a usar PostgreSQL, copia `.env.example` a `.env` y ajústalo. Si no, **omite
   este paso**: se usará `fetch_covtype` automáticamente.
4. Abre el notebook y **ejecuta todas las celdas en orden**. La primera vez, la celda de
   *bootstrap* instala lo que falte (incluido AutoGluon). Para una prueba rápida pon
   `QUICK_SMOKE = True` en la sección 14.

> La descarga de `fetch_covtype` (~75 MB la primera vez) requiere conexión a internet; luego queda
> en caché local de scikit-learn.

## Salidas (en `artifacts/`, junto al notebook)

```
artifacts/
├── configurations.json      # las 25 configuraciones
├── environment.json         # versiones utilizadas
├── selected_model.json      # configuración ganadora + métricas de test
├── models/config_01 … config_25/    # un modelo (TabularPredictor) por configuración
├── results/results.csv      # tabla consolidada (25 filas)
├── results/results.parquet
└── mlflow_payloads/config_01.json … config_25.json   # params+metrics+rutas, listos para MLflow
```

## Preparación para MLflow

Cada experimento deja un *payload* (`mlflow_payloads/config_XX.json`) con `params`, `metrics`,
`tags` y rutas de artefactos. La sección 20 del notebook muestra el bucle
`mlflow.start_run()` / `log_params` / `log_metrics` / `log_artifact(s)` que otro componente
añadirá, sin ejecutarlo aquí.

## Notas técnicas

- **Modelo único**: LightGBM (clave `GBM` en AutoGluon), intercambiable con `MODEL_FAMILY`.
- **25 configuraciones**: 5 learning rates × 5 niveles de capacidad/regularización (malla
  determinista, reproducible con `SEED`).
- **Métrica principal**: `f1_macro` (el target está desbalanceado).
- **Test aislado**: sólo se evalúa una vez, sobre la configuración ganadora.
