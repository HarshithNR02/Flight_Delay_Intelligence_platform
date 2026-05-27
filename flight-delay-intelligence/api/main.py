from fastapi import FastAPI
from api.predict import router

app = FastAPI(
    title="Flight Delay Intelligence Platform",
    description="Predict US domestic flight delays with SHAP explanations.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}
