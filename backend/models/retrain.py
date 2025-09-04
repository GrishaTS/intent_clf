# models/retrain.py
from __future__ import annotations

from typing import Dict, Literal, Optional, Union
from pydantic import BaseModel, Field, AnyHttpUrl, model_validator

AuthMethod = Literal["none", "basic", "bearer", "api_key", "headers_json"]
FilterMethod = Literal["by_freq", "by_quality"]

class DataAuth(BaseModel):
    method: AuthMethod = Field(default="none")

    # basic
    username: Optional[str] = None
    password: Optional[str] = None

    # bearer
    token: Optional[str] = None

    # api_key
    header: Optional[str] = None
    value: Optional[str] = None

    # headers_json
    headers: Optional[Dict[str, str]] = None
    headers_json: Optional[str] = None  # на всякий — если фронт пришлёт строку

    @model_validator(mode="after")
    def _validate_by_method(self):
        m = self.method
        if m == "basic":
            if not self.username or not self.password:
                raise ValueError("auth.basic: username/password required")
        elif m == "bearer":
            if not self.token:
                raise ValueError("auth.bearer: token required")
        elif m == "api_key":
            if not self.header or not self.value:
                raise ValueError("auth.api_key: header/value required")
        elif m == "headers_json":
            # приоритет готового dict
            if not self.headers:
                # fallback: попытаться распарсить headers_json если она вдруг пришла
                import json
                try:
                    parsed = json.loads(self.headers_json or "")
                    if not isinstance(parsed, dict) or not parsed:
                        raise ValueError
                    self.headers = {str(k): str(v) for k, v in parsed.items()}
                except Exception:
                    raise ValueError("auth.headers_json: headers dict required")
        return self

class DataAPI(BaseModel):
    url: AnyHttpUrl
    auth: DataAuth

class RetrainConfig(BaseModel):
    # источник/доступ
    data_api: DataAPI

    # колонки
    id_col: str
    subject_col: Optional[str] = None
    description_col: str
    class_col: str

    # фильтрация
    filter_method: FilterMethod
    top_n_values: int = Field(ge=1)
    min_samples: int = Field(ge=1)
    min_f1_score: float = Field(ge=0.0, le=1.0)

    # индекс
    clear_index_flag: bool = True

    # расписание — как пришло с фронта
    RUN_EVERY_DAYS: str
    ANCHOR_DATETIME_STR: str  # "YYYY-MM-DD HH:MM"
    RUN_ON_START: str         # "1" | "0"

    model_config = dict(extra="forbid")

class RetrainPushResponse(BaseModel):
    ok: bool
    user_id: str
    path: str
    updated_at: str  # iso

class RetrainPullResponse(BaseModel):
    ok: bool
    user_id: str
    path: Optional[str] = None
    updated_at: Optional[str] = None
    config: Optional[RetrainConfig] = None

class RetrainDeleteResponse(BaseModel):
    ok: bool
    user_id: str
    path: str
    deleted: bool