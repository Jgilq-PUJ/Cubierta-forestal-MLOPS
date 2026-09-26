from fastapi import FastAPI

app = FastAPI(title="Cubierta Forestal · Inference API", version="0.1.0")


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe used by the container HEALTHCHECK and compose."""
    return {"status": "ok"}
