# routers/upload.py
import logging
from fastapi import APIRouter, Depends, HTTPException
from services.auth import get_current_active_user
from services.embedding import embedder
from services.vector_db import vector_db

from models.auth import User
from models.request_models import BatchRequestItem, UploadResponse

logger = logging.getLogger("upload_router")

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_data(
    request: BatchRequestItem, current_user: User = Depends(get_current_active_user)
):
    subjects = [item.subject for item in request.items]
    descriptions = [item.description for item in request.items]

    # Извлекаем эмбединги (склеенные subject+description)
    combined_embeddings = embedder.get_combined_embeddings(subjects, descriptions)

    # Подготавливаем payload
    payloads = []
    for i, item in enumerate(request.items):
        if not hasattr(item, "id") or item.id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required 'id' field for item at index {i}",
            )

        payloads.append(
            {
                "request_id": item.id,
                "subject": item.subject,
                "description": item.description,
                "class_name": getattr(item, "class_name", "unknown"),
                "task": getattr(item, "task", "unknown"),
            }
        )

    # Загружаем в Qdrant
    try:
        ids = vector_db.upload_vectors(
            user_id=current_user.id,
            vectors=combined_embeddings,
            payloads=payloads,
        )
        return UploadResponse(success=True, message="Данные успешно загружены", ids=ids)
    except Exception as e:
        logger.error(f"Error uploading vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading vectors: {e}")


@router.post("/clear_index", response_model=UploadResponse)
async def clear_index(current_user: User = Depends(get_current_active_user)):
    try:
        vector_db.clear_user_collection(user_id=current_user.id)
        return UploadResponse(success=True, message="Индекс успешно очищен", ids=[])
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing index: {e}")
