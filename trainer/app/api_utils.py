# api_utils_headless.py
from __future__ import annotations

import uuid
import logging
from typing import Callable, Optional, Tuple, List, Dict

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ====================== Helpers ======================

def _to_str_id(val) -> str:
    """Нормализует любое значение id к строке.
    NaN/None/пустые -> UUID; float-целое -> без '.0'."""
    try:
        if val is None or (isinstance(val, str) and val.strip() == "") or pd.isna(val):
            return str(uuid.uuid4())
        if isinstance(val, (float, np.floating)):
            if np.isfinite(val) and float(val).is_integer():
                return str(int(val))
            return f"{val:.15g}"
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        s = str(val).strip()
        return s if s else str(uuid.uuid4())
    except Exception:
        return str(uuid.uuid4())

def _safe_str(v: object, default: str) -> str:
    if v is None:
        return default
    try:
        s = str(v).strip()
        return s if s else default
    except Exception:
        return default

def _truncate(s: str, max_len: int) -> str:
    if s is None:
        return s
    return s if len(s) <= max_len else s[:max_len]

def _prepare_item(row: pd.Series) -> dict:
    """Собирает валидный item для /upload из строки DataFrame."""
    item_id = _to_str_id(row.get("id"))
    subject = _safe_str(row.get("subject"), "no_subject")
    description = _safe_str(row.get("description"), "no_description")
    class_name = _safe_str(row.get("class"), "Others")

    # Ограничиваем длину текстов
    subject = _truncate(subject, 500)
    description = _truncate(description, 5000)

    item = {
        "id": item_id,
        "subject": subject,
        "description": description,
        "class_name": class_name,
    }
    task = row.get("task")
    if task is not None and not (isinstance(task, float) and np.isnan(task)):
        item["task"] = _safe_str(task, "")
    return item

def _df_to_metrics_items(df: pd.DataFrame) -> List[Dict]:
    """Готовит payload для /metrics/compute из test_df."""
    items = []
    for _, row in df.iterrows():
        items.append({
            "id": _to_str_id(row.get("id")),
            "subject": _safe_str(row.get("subject"), "no_subject"),
            "description": _safe_str(row.get("description"), "no_description"),
            "label": _safe_str(row.get("class"), "Others"),
        })
    return items

# ====================== Low-level HTTP ======================

def _post_upload_items(api_url: str, headers: dict, items: List[dict]) -> Tuple[bool, List[str], str]:
    """
    Пытается отправить пачку items.
    Возвращает (ok, uploaded_ids, error_text). Если сервер не вернул ids, используем локальные id.
    """
    try:
        resp = requests.post(f"{api_url}/upload", json={"items": items}, headers=headers, timeout=60)
        if resp.status_code != 200:
            return False, [], f"{resp.status_code} {resp.text}"
        data = resp.json()
        if not isinstance(data, dict) or not data.get("success"):
            return False, [], f"Unexpected response: {data}"
        ids = data.get("ids")
        if isinstance(ids, list) and ids:
            return True, ids, ""
        return True, [it["id"] for it in items], ""
    except Exception as e:
        return False, [], str(e)

def _upload_with_fallback(
    api_url: str,
    headers: dict,
    items: List[dict],
    bad_ids: List[str]
) -> List[str]:
    """
    Загрузка с прогрессивным фолбэком: весь батч -> половинки -> одиночные.
    Возвращает список успешно загруженных id; неуспешные копит в bad_ids.
    """
    ok, ids, err = _post_upload_items(api_url, headers, items)
    if ok:
        return ids

    if len(items) == 1:
        bad_ids.append(items[0]["id"])
        log.warning("upload failed for single id=%s err=%s", items[0]["id"], err)
        return []

    mid = len(items) // 2
    left = items[:mid]
    right = items[mid:]
    uploaded_ids = []
    uploaded_ids.extend(_upload_with_fallback(api_url, headers, left, bad_ids))
    uploaded_ids.extend(_upload_with_fallback(api_url, headers, right, bad_ids))
    return uploaded_ids

# ====================== Public API ======================

def get_token(api_url: str, username: str, password: str) -> Optional[str]:
    """Получение токена авторизации (OAuth2 Password flow ожидается на /token)."""
    try:
        response = requests.post(
            f"{api_url}/token",
            data={"username": username, "password": password, "scope": "predict upload search"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        if response.status_code != 200:
            log.error("get_token failed: %s %s", response.status_code, response.text)
            return None
        data = response.json()
        return data.get("access_token")
    except requests.exceptions.RequestException as e:
        log.error("get_token connection error: %s", e)
        return None

def clear_index(token: str, api_url: str) -> bool:
    """Очистка индекса на бэкенде."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        resp = requests.post(f"{api_url}/clear_index", headers=headers, timeout=60)
        if resp.status_code != 200:
            log.error("clear_index failed: %s %s", resp.status_code, resp.text)
            return False
        data = resp.json()
        ok = bool(data.get("success"))
        if not ok:
            log.error("clear_index unexpected response: %s", data)
        return ok
    except Exception as e:
        log.error("clear_index exception: %s", e)
        return False

def upload_data(
    data: pd.DataFrame,
    token: str,
    api_url: str,
    *,
    batch_size: int = 50,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> List[str]:
    """
    Загрузка данных в систему (устойчиво; id всегда строка; fallback при ошибках).
    Возвращает список успешно загруженных id.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    n = len(data)
    total_batches = n // batch_size + (1 if n % batch_size > 0 else 0)
    uploaded_ids_total: List[str] = []
    bad_ids_total: List[str] = []

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n)
        batch_df = data.iloc[start_idx:end_idx]

        items: List[dict] = []
        seen_ids = set()
        for _, row in batch_df.iterrows():
            item = _prepare_item(row)
            if item["id"] in seen_ids:
                item["id"] = f"{item['id']}-{uuid.uuid4()}"
            seen_ids.add(item["id"])
            items.append(item)

        uploaded_ids = _upload_with_fallback(api_url, headers, items, bad_ids_total)
        uploaded_ids_total.extend(uploaded_ids)

        if progress_cb:
            progress_cb(min(1.0, (i + 1) / total_batches))

        if len(uploaded_ids) != len(items):
            log.warning(
                "batch %d/%d: uploaded %d/%d, skipped %d",
                i + 1, total_batches, len(uploaded_ids), len(items), len(items) - len(uploaded_ids)
            )

    if bad_ids_total:
        log.error("skipped records: %d (first 50) %s", len(bad_ids_total), bad_ids_total[:50])

    log.info("uploaded total: %d", len(uploaded_ids_total))
    return uploaded_ids_total

def compute_metrics_backend(test_df: pd.DataFrame, token: str, api_url: str) -> Optional[dict]:
    """POST /metrics/compute — считает метрики на сервере и сохраняет отчёт. Возвращает JSON с метриками."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    items = _df_to_metrics_items(test_df)
    try:
        resp = requests.post(f"{api_url}/metrics/compute", json={"items": items}, headers=headers, timeout=120)
        if resp.status_code != 200:
            log.error("metrics/compute failed: %s %s", resp.status_code, resp.text)
            return None
        return resp.json()
    except Exception as e:
        log.error("metrics/compute exception: %s", e)
        return None

def filter_high_quality_classes(
    df: pd.DataFrame,
    min_samples: int = 10,
    min_f1_score: float = 0.5,
    api_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    *,
    clear_before_upload: bool = True,
    test_size: float = 0.05,
    random_state: int = 42,
) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """
    Фильтрация классов по качеству с использованием backend /metrics/compute.
    Возвращает (df_high_quality, report_dict).
    df должен иметь колонки: id, subject (опц.), description, class.
    """

    df_processed = df.copy()

    # 1) Замена редких классов на "Others"
    class_counts = df_processed["class"].value_counts()
    rare_classes = class_counts[class_counts < min_samples].index.tolist()
    if rare_classes:
        df_processed["class"] = df_processed["class"].apply(lambda x: "Others" if x in rare_classes else x)

    # 2) train/test
    train_df, test_df = train_test_split(
        df_processed, test_size=test_size, random_state=random_state, stratify=df_processed["class"]
    )

    # 3) токен
    token = get_token(api_url, username, password)
    if not token:
        log.error("auth token not received")
        return None, None

    # 4) очистка индекса (по желанию)
    if clear_before_upload:
        if not clear_index(token, api_url):
            log.warning("clear_index failed (continuing anyway)")

    # 5) загрузка train
    upload_data(train_df, token, api_url)

    # 6) метрики по test
    metrics = compute_metrics_backend(test_df, token, api_url)
    if not metrics:
        return None, None

    report_dict = metrics.get("classification_report_dict", {})
    if not report_dict:
        return None, None

    # 7) отбор high-quality классов
    high_quality_classes: List[str] = []
    for class_name, m in report_dict.items():
        if class_name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        if m.get("f1-score", 0.0) >= min_f1_score:
            high_quality_classes.append(class_name)

    # 8) финальная фильтрация набора
    df_high_quality = df_processed.copy()
    df_high_quality["class"] = df_high_quality["class"].apply(
        lambda x: x if x in high_quality_classes else "Others"
    )

    return df_high_quality, report_dict
