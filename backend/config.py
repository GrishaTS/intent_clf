# backend/settings.py
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Classification API"
    APP_VERSION: str = "1.0.0"

    # Безопасность
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 360

    # Модель
    MODEL_NAME: str = "intfloat/multilingual-e5-large"
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 16
    DEVICE: str = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"
    USE_CUDA: str = os.getenv("USE_CUDA", "0")  # "1" -> CUDA, иначе CPU

    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant_griga")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6555"))
    # Старая общая коллекция — оставлено для совместимости
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "requests")

    # Классификация
    KNN_NEIGHBORS: int = 5
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    RERANK_THRESHOLD: float = 0.2

    # Метрики (для /metrics/compute и /metrics/latest)
    METRICS_DIR: str = os.getenv("METRICS_DIR", "metrics")  # файлы вида {user_id}.json
    
    # Переобучение
    RETRAIN_DIR: str = "retrain"

    class Config:
        env_file = ".env"


settings = Settings()

# гарантируем наличие директории для метрик на старте
os.makedirs(settings.METRICS_DIR, exist_ok=True)
