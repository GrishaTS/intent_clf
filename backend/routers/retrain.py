from __future__ import annotations

from fastapi import APIRouter, Depends, status
from services.auth import get_current_active_user
from models.auth import User
from models.retrain import (
    RetrainConfig,
    RetrainPushResponse,
    RetrainPullResponse,
    RetrainDeleteResponse,   # <-- NEW
)
from services.retrain import save_config, load_config, delete_config  # <-- add delete

router = APIRouter(prefix="/retrain", tags=["retrain"])

@router.post("/push", response_model=RetrainPushResponse, status_code=status.HTTP_201_CREATED)
async def push_retrain_config(
    cfg: RetrainConfig,
    current_user: User = Depends(get_current_active_user),
):
    user_id = getattr(current_user, "id", None) or current_user.username
    path, updated_at = save_config(user_id, cfg)
    return RetrainPushResponse(
        ok=True,
        user_id=str(user_id),
        path=str(path),
        updated_at=updated_at,
    )

@router.get("/pull", response_model=RetrainPullResponse)
async def pull_retrain_config(
    current_user: User = Depends(get_current_active_user),
):
    user_id = getattr(current_user, "id", None) or current_user.username
    try:
        cfg, path, updated_at = load_config(user_id)
    except FileNotFoundError:
        return RetrainPullResponse(ok=False, user_id=str(user_id))
    return RetrainPullResponse(
        ok=True,
        user_id=str(user_id),
        path=str(path),
        updated_at=updated_at,
        config=cfg,
    )

@router.delete("/delete", response_model=RetrainDeleteResponse)
async def delete_retrain_config(
    current_user: User = Depends(get_current_active_user),
):
    user_id = getattr(current_user, "id", None) or current_user.username
    path, existed = delete_config(user_id)
    return RetrainDeleteResponse(
        ok=True,
        user_id=str(user_id),
        path=str(path),
        deleted=bool(existed),
    )
