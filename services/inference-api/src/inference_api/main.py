from fastapi import FastAPI

from inference_api.routers import inference

app = FastAPI(title="Cubierta Forestal · Inference API", version="0.1.0")


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(inference.router)