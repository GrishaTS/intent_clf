# main (обновлённый)
from config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, health, predict, search, upload, metrics, retrain

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API для классификации входящих обращений",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # в продакшене ограничить список
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(upload.router)
app.include_router(search.router)
app.include_router(metrics.router)
app.include_router(retrain.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
