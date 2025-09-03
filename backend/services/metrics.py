# backend/services/metrics.py
import json
import logging
import os
import time
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import settings  # единообразно с main/app
from services.embedding import embedder
from services.classification import classifier

logger = logging.getLogger("services.metrics")


# ---------------------------- ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ ----------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_empty(x: Any) -> bool:
    """Пусто ли обобщённо (без булевой правды numpy)."""
    if x is None:
        return True
    if isinstance(x, np.ndarray):
        return x.size == 0
    try:
        return len(x) == 0  # noqa: SIM118
    except TypeError:
        return False


def _is_empty_vec(x: Any) -> bool:
    """Проверка пустоты именно для эмбеддингов/векторов любых типов."""
    if x is None:
        return True
    if isinstance(x, np.ndarray):
        return x.size == 0
    if isinstance(x, (list, tuple)):
        return len(x) == 0
    return False


def _norm_label(s: str) -> str:
    """Нормализация строк меток, чтобы сопоставлять «похожие» вариации."""
    s = (s or "").strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _make_canonical_map(classes: List[str]) -> Dict[str, str]:
    """Карта: нормализованная строка -> каноническая метка из белого списка."""
    return {_norm_label(str(c)): str(c) for c in classes}


def _tolist(x: Any):
    """numpy -> python list, иначе вернуть как есть."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ---------------------------- НОРМАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ ----------------------------

def _coerce_pred_list(raw: Any, class_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Приводит произвольное предсказание модели к списку словарей вида:
        [{"class_name": <str>, "probability": <float>}, ...] (отсортировано по вероятности убыв.)
    Поддерживаются форматы:
      - список dict’ов с ключами {class_name|label|class, probability|score}
      - dict {"labels": [...], "scores": [...]}
      - кортеж/список (labels, scores)
      - чистый вектор вероятностей (list/ndarray)
      - одиночная строка (label) или индекс класса (int)
    """
    if raw is None:
        return []

    # Уже список dict’ов?
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and (
        "label" in raw[0] or "class_name" in raw[0] or "class" in raw[0]
    ):
        out = []
        for d in raw:
            lbl = d.get("class_name") or d.get("label") or d.get("class")
            prob = float(d.get("probability", d.get("score", 0.0)) or 0.0)
            out.append({"class_name": lbl, "probability": prob})
        out.sort(key=lambda x: x["probability"], reverse=True)
        return out

    # Dict с labels/scores
    if isinstance(raw, dict) and ("labels" in raw and "scores" in raw):
        labels = _tolist(raw["labels"])
        scores = [float(s) for s in _tolist(raw["scores"])]
        out = []
        for i, s in enumerate(scores):
            lbl = labels[i] if labels is not None else (class_names[i] if class_names and i < len(class_names) else str(i))
            out.append({"class_name": lbl, "probability": s})
        out.sort(key=lambda x: x["probability"], reverse=True)
        return out

    # Tuple/List (labels, scores)
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and (
        isinstance(raw[0], (list, tuple, np.ndarray)) and isinstance(raw[1], (list, tuple, np.ndarray))
    ):
        labels = _tolist(raw[0])
        scores = [float(s) for s in _tolist(raw[1])]
        out = []
        for i, s in enumerate(scores):
            lbl = labels[i] if labels is not None else (class_names[i] if class_names and i < len(class_names) else str(i))
            out.append({"class_name": lbl, "probability": s})
        out.sort(key=lambda x: x["probability"], reverse=True)
        return out

    # Чистый вектор вероятностей
    raw_list = _tolist(raw)
    if isinstance(raw_list, (list, tuple)) and raw_list and isinstance(raw_list[0], (float, int, np.floating, np.integer)):
        scores = [float(s) for s in raw_list]
        out = []
        for i, s in enumerate(scores):
            lbl = class_names[i] if class_names and i < len(class_names) else str(i)
            out.append({"class_name": lbl, "probability": s})
        out.sort(key=lambda x: x["probability"], reverse=True)
        return out

    # Одиночный top-1 как строка/индекс
    if isinstance(raw, str):
        return [{"class_name": raw, "probability": 1.0}]
    if isinstance(raw, (int, np.integer)) and class_names:
        idx = int(raw)
        if 0 <= idx < len(class_names):
            return [{"class_name": class_names[idx], "probability": 1.0}]

    # Фоллбэк — завернём как есть
    return [{"class_name": str(raw), "probability": 0.0}]


def _choose_top1(
    raw_pred: Any,
    canonical_map: Dict[str, str],
    class_names_in_order: Optional[List[str]] = None,
    threshold: Optional[float] = None,
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Возвращает (canonical_label, prob, reason_if_invalid)
      reason_if_invalid ∈ {None, "empty", "below_threshold", "not_in_classlist"}
    """
    items = _coerce_pred_list(raw_pred, class_names=class_names_in_order)
    if _is_empty(items):
        return None, None, "empty"

    best = items[0]
    raw_lbl = best.get("class_name")
    prob = float(best.get("probability", 0.0) or 0.0)

    if threshold is not None and prob < threshold:
        return None, prob, "below_threshold"

    if raw_lbl is None:
        return None, prob, "empty"

    canon = canonical_map.get(_norm_label(str(raw_lbl)))
    if canon is None:
        return None, prob, "not_in_classlist"

    return canon, prob, None


# ---------------------------- (УСТАРЕВШЕЕ) СОВМЕСТИМОСТЬ ----------------------------

def _normalize_preds(preds: Any) -> list:
    """
    Старый адаптер, оставлен на случай внешних вызовов. Новая логика — _coerce_pred_list/_choose_top1.
    """
    if _is_empty(preds):
        return []
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    if not isinstance(preds, (list, tuple)):
        if isinstance(preds, dict) and "predictions" in preds:
            preds = preds["predictions"]
        else:
            preds = [preds]
    if len(preds) == 0 and isinstance(preds, dict):
        labels = preds.get("labels")
        scores = preds.get("scores")
        if labels is not None and scores is not None:
            preds = list(zip(labels, scores))
    return list(preds)


def _top1_class(preds: Any) -> Optional[str]:
    """
    Упрощённый top-1 для обратной совместимости.
    """
    preds = _normalize_preds(preds)
    if _is_empty(preds):
        return None

    def get_pair(item) -> Tuple[Optional[str], float]:
        if isinstance(item, dict):
            cls = item.get("class_name") or item.get("label") or item.get("class")
            score = item.get("probability", item.get("score", 0.0))
            try:
                return cls, float(score)
            except Exception:
                return cls, 0.0
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            cls, score = item[0], item[1]
            try:
                return cls, float(score)
            except Exception:
                return cls, 0.0
        cls = getattr(item, "class_name", None) or getattr(item, "label", None) or getattr(item, "class_", None)
        score = getattr(item, "probability", getattr(item, "score", 0.0))
        try:
            return cls, float(score)
        except Exception:
            return cls, 0.0

    best_cls, best_score = None, float("-inf")
    for it in preds:
        cls, score = get_pair(it)
        if cls is None:
            continue
        if score > best_score:
            best_cls, best_score = cls, score
    return best_cls


# ---------------------------- ОСНОВНАЯ ЛОГИКА МЕТРИК ----------------------------

def compute_metrics(
    *,
    user_id: str,
    items: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Для каждого элемента:
      1) get_combined_embeddings([subject], [description]) -> (1, 2048)
      2) classifier.predict(user_id, subject, description, query_vector=<first_row>)
      3) нормализуем предсказание → top-1 → каноническая метка
      4) считаем метрики и сохраняем {METRICS_DIR}/{user_id}.json
    """
    t0 = time.time()

    items = list(items)
    n_total = len(items)

    # Пустой набор
    if n_total == 0:
        data = {
            "filename": f"{user_id}.json",
            "timestamp": _timestamp(),
            "accuracy": 0.0,
            "n_total": 0,
            "n_valid": 0,
            "n_invalid": 0,
            "classes": [],
            "confusion_matrix": [],
            "classification_report_dict": {},
            "classification_report_text": "Empty dataset",
        }
        _ensure_dir(settings.METRICS_DIR)
        _save_json(os.path.join(settings.METRICS_DIR, f"{user_id}.json"), data)
        return data

    # Истинные метки и белый список классов
    y_true = [str(it["label"]) for it in items]
    allowed_classes = sorted(list(set(y_true)))
    canonical_map = _make_canonical_map(allowed_classes)

    y_pred: List[Optional[str]] = []
    invalid = 0
    invalid_reasons = {"empty": 0, "below_threshold": 0, "not_in_classlist": 0, "exception": 0}

    for i, it in enumerate(items):
        subject = str(it.get("subject", "no_subject"))
        description = str(it["description"])  # KeyError если нет — корректно

        try:
            emb = embedder.get_combined_embeddings([subject], [description])  # ожидаемо (1, 2048)

            # ВАЖНО: без булевой правды numpy
            if _is_empty_vec(emb):
                y_pred.append(None)
                invalid += 1
                invalid_reasons["empty"] += 1
                continue

            # Достаём первую строку вне зависимости от типа
            if isinstance(emb, np.ndarray):
                if emb.ndim == 1:
                    query_vec = emb
                else:
                    if emb.shape[0] == 0:
                        y_pred.append(None)
                        invalid += 1
                        invalid_reasons["empty"] += 1
                        continue
                    query_vec = emb[0]
            elif isinstance(emb, (list, tuple)):
                if len(emb) == 0:
                    y_pred.append(None)
                    invalid += 1
                    invalid_reasons["empty"] += 1
                    continue
                query_vec = emb[0]
            else:
                # неизвестный тип — попробуем как есть
                query_vec = emb

            preds = classifier.predict(
                user_id=user_id,
                subject=subject,
                description=description,
                query_vector=query_vec,
            )

            # Универсальный top-1 + канонизация
            lbl, prob, why = _choose_top1(
                preds,
                canonical_map=canonical_map,
                class_names_in_order=allowed_classes,  # если модель вернула просто вектор
                threshold=None,  # при необходимости можно выставить порог
            )

            if lbl is None:
                invalid += 1
                if why in invalid_reasons:
                    invalid_reasons[why] += 1
                y_pred.append(None)
            else:
                y_pred.append(lbl)

            # Немного отладочной информации для первых примеров
            if i < 3:
                logger.debug("metrics.sample #%s raw_pred=%r -> top1=%r prob=%r why=%r", i, preds, lbl, prob, why)

        except Exception as e:
            logger.error(f"Prediction failed for item #{i}: {e}")
            y_pred.append(None)
            invalid += 1
            invalid_reasons["exception"] += 1

    # Отфильтровываем валидные индексы
    valid_idx = [i for i, yp in enumerate(y_pred) if yp is not None]
    y_true_valid = [y_true[i] for i in valid_idx]
    y_pred_valid = [y_pred[i] for i in valid_idx]

    # Если нет валидных — пишем диагностическую сводку
    if len(y_true_valid) == 0:
        data = {
            "filename": f"{user_id}.json",
            "timestamp": _timestamp(),
            "accuracy": 0.0,
            "n_total": n_total,
            "n_valid": 0,
            "n_invalid": n_total,
            "invalid_breakdown": invalid_reasons,
            "classes": allowed_classes,
            "confusion_matrix": [],
            "classification_report_dict": {},
            "classification_report_text": "No valid predictions!",
        }
        _ensure_dir(settings.METRICS_DIR)
        _save_json(os.path.join(settings.METRICS_DIR, f"{user_id}.json"), data)
        return data

    # Метрики только по валидным
    classes = sorted(list(set(y_true_valid + y_pred_valid)))
    acc = float(accuracy_score(y_true_valid, y_pred_valid))
    report_dict = classification_report(
        y_true_valid, y_pred_valid, labels=classes, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true_valid, y_pred_valid, labels=classes, zero_division=0
    )
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=classes)

    ts = _timestamp()
    data = {
        "filename": f"{user_id}.json",
        "timestamp": ts,
        "accuracy": acc,
        "n_total": n_total,
        "n_valid": len(y_true_valid),
        "n_invalid": invalid,
        "invalid_breakdown": invalid_reasons,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }

    _ensure_dir(settings.METRICS_DIR)
    out_path = os.path.join(settings.METRICS_DIR, f"{user_id}.json")
    _save_json(out_path, data)

    logger.info(
        "compute_metrics: user=%s total=%s valid=%s invalid=%s acc=%.4f time=%.2fs -> %s",
        user_id, n_total, len(y_true_valid), invalid, acc, time.time() - t0, out_path
    )
    return data


def read_latest_metrics(*, user_id: str) -> Dict[str, Any]:
    """
    Читает {METRICS_DIR}/{user_id}.json -> dict
    """
    path = os.path.join(settings.METRICS_DIR, f"{user_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found for user_id={user_id}")
    return _read_json(path)
