# routers/predict.py
from fastapi import APIRouter, Depends, HTTPException
from services.auth import get_current_active_user
from services.classification import classifier
from services.embedding import embedder

from models.auth import User
from models.request_models import PredictionResponse, RequestItem

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: RequestItem,
    current_user: User = Depends(get_current_active_user),
):
    # 1) user_id для коллекции: UUID если есть, иначе username (fallback)
    user_id = getattr(current_user, "id", None) or current_user.username

    # 2) эмбеддинг (склеенный subject+description)
    combined_embedding = embedder.get_combined_embeddings(
        [request.subject], [request.description]
    )  # shape: (1, 2048)

    if combined_embedding is None or len(combined_embedding) == 0:
        # жёсткая защита: не отправляем пустой запрос в БД
        return PredictionResponse(predictions=[])

    # 3) предсказания (через KNN+reranker в коллекции пользователя)
    predictions = classifier.predict(
        user_id=user_id,
        subject=request.subject,
        description=request.description,
        query_vector=combined_embedding[0],
    )

    # 4) на стороне API возвращаем пустой список как валидный ответ,
    # чтобы клиентский код не делал max() по пустоте
    return PredictionResponse(predictions=predictions or [])