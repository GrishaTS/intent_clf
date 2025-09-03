# routers/metrics.py
from fastapi import APIRouter, Depends, HTTPException
from services.auth import get_current_active_user
from models.auth import User
from models.request_models import MetricsComputeRequest, MetricsResponse
from services.metrics import compute_metrics, read_latest_metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("/compute", response_model=MetricsResponse)
async def compute_metrics_endpoint(
    request: MetricsComputeRequest,
    current_user: User = Depends(get_current_active_user),
):
    user_id = getattr(current_user, "id", None) or current_user.username
    items = [it.dict() for it in request.items]  # pydantic v1
    data = compute_metrics(user_id=user_id, items=items)
    return data


@router.get("/latest", response_model=MetricsResponse)
async def latest_metrics_endpoint(
    current_user: User = Depends(get_current_active_user),
):
    user_id = getattr(current_user, "id", None) or current_user.username
    try:
        data = read_latest_metrics(user_id=user_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics not found for current user")
    return data
