# services/retrain_storage.py
from __future__ import annotations

import json, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
from fastapi.encoders import jsonable_encoder  # <-- добавь

from config import settings
from models.retrain import RetrainConfig

def _retrain_dir() -> Path:
    d = getattr(settings, "RETRAIN_DIR", "retrain")
    return d if isinstance(d, Path) else Path(str(d))

def _user_file(user_id: str) -> Path:
    base = _retrain_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{user_id}.json"

def save_config(user_id: str, cfg: RetrainConfig) -> Tuple[Path, str]:
    path = _user_file(user_id)
    tmp = path.with_suffix(".json.tmp")

    # pydantic v2 → dict, затем jsonable_encoder приводим UUID/datetime/Path/np.*
    cfg_dict = cfg.model_dump(by_alias=False, exclude_none=False)
    cfg_dict = jsonable_encoder(cfg_dict)

    # страховка: принудительно строки в headers (на случай чужих типов)
    auth = cfg_dict.get("data_api", {}).get("auth", {})
    if isinstance(auth.get("headers"), dict):
        auth["headers"] = {str(k): str(v) for k, v in auth["headers"].items()}

    payload = {
        "_meta": {
            "user_id": str(user_id),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "ts": int(time.time()),
        },
        "config": cfg_dict,
    }

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    os.replace(tmp, path)
    return path, payload["_meta"]["updated_at"]

def load_config(user_id: str) -> Tuple[RetrainConfig, Path, str]:
    path = _user_file(user_id)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = RetrainConfig(**data["config"])
    updated_at = data.get("_meta", {}).get("updated_at", "")
    return cfg, path, updated_at

def delete_config(user_id: str) -> tuple[Path, bool]:
    """
    Удаляет retrain/<user_id>.json (и возможный .tmp).
    Возвращает (path, existed_before).
    """
    path = _user_file(user_id)
    tmp = path.with_suffix(".json.tmp")

    existed = path.exists()
    try:
        if path.exists():
            path.unlink()
    except FileNotFoundError:
        existed = False  # на всякий

    # подчистим возможный временный файл
    try:
        if tmp.exists():
            tmp.unlink()
    except FileNotFoundError:
        pass

    return path, existed