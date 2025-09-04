# frontend/api_utils.py  — refactored with i18n & decomposition

from __future__ import annotations

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split

from i18n import get_translations


# ================================
#   Logger
# ================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================================
#   i18n helpers
# ================================
def _TR():
    """Lazy translator per current UI language."""
    lang = st.session_state.get("lang", "ru")
    return get_translations(lang)

def _T(key: str, **fmt) -> str:
    """Translate with optional .format(**fmt). Falls back to key if missing (see i18n.t)."""
    text = _TR()[key]
    try:
        return text.format(**fmt) if fmt else text
    except Exception:
        return text


# ================================
#   Common text & id normalization
# ================================
def _to_str_id(val) -> str:
    """Normalize any id to string. Empty/NaN -> UUID; ints keep int string; floats keep compact form."""
    try:
        if val is None or (isinstance(val, str) and val.strip() == "") or pd.isna(val):
            return str(uuid.uuid4())
        if isinstance(val, (float, np.floating)):
            if np.isfinite(val) and float(val).is_integer():
                return str(int(val))
            return f"{val:.15g}"
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        return (str(val) or "").strip() or str(uuid.uuid4())
    except Exception:
        return str(uuid.uuid4())

def _safe_str(v: object, default: str) -> str:
    try:
        if v is None:
            return default
        s = str(v).strip()
        return s or default
    except Exception:
        return default

def _truncate(s: Optional[str], max_len: int) -> Optional[str]:
    if s is None:
        return s
    return s if len(s) <= max_len else s[:max_len]


# ================================
#   HTTP core
# ================================
def _headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def _json_post(url: str, payload: Any, headers: Dict[str, str], timeout: int = 60) -> Tuple[Optional[Dict[str, Any]], Optional[str], int]:
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        status = resp.status_code
        if status >= 400:
            return None, f"{status} {resp.text}", status
        return resp.json(), None, status
    except Exception as e:
        return None, str(e), 0

def _json_get(url: str, headers: Dict[str, str], timeout: int = 60) -> Tuple[Optional[Dict[str, Any]], Optional[str], int]:
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        status = resp.status_code
        if status >= 400:
            return None, f"{status} {resp.text}", status
        return resp.json(), None, status
    except Exception as e:
        return None, str(e), 0

def _ui_msg(level: str, key: str, **fmt) -> None:
    """level in {'error','warning','info','success'}; text from i18n."""
    msg = _T(key, **fmt)
    getattr(st, level)(msg)


# ================================
#   Upload helpers
# ================================
def _prepare_item(row: pd.Series) -> Dict[str, Any]:
    """Make a valid /upload item from DataFrame row."""
    item_id = _to_str_id(row.get("id"))
    subject = _safe_str(row.get("subject"), "no_subject")
    description = _safe_str(row.get("description"), "no_description")
    class_name = _safe_str(row.get("class"), "Others")

    subject = _truncate(subject, 500)
    description = _truncate(description, 5000)

    item = {"id": item_id, "subject": subject, "description": description, "class_name": class_name}
    if "task" in row and not pd.isna(row["task"]):
        item["task"] = _safe_str(row["task"], "")
    return item

def _df_to_metrics_items(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare payload for /metrics/compute from test_df."""
    items: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        items.append({
            "id": _to_str_id(row.get("id")),
            "subject": _safe_str(row.get("subject"), "no_subject"),
            "description": _safe_str(row.get("description"), "no_description"),
            "label": _safe_str(row.get("class"), "Others"),
        })
    return items

def _post_upload_items(api_url: str, headers: dict, items: list) -> Tuple[bool, List[str], str]:
    """Try to upload batch of items; returns (ok, uploaded_ids, err_text)."""
    data, err, status = _json_post(f"{api_url}/upload", {"items": items}, headers, timeout=60)
    if err:
        return False, [], err
    if not isinstance(data, dict) or not data.get("success"):
        return False, [], _T("api.upload.unexpected_response", data=str(data))
    ids = data.get("ids")
    return True, (ids if isinstance(ids, list) and ids else [it["id"] for it in items]), ""

def _upload_with_fallback(api_url: str, headers: dict, items: list, bad_ids: list) -> list:
    """Recursive upload with bisection fallback; accumulates bad ids."""
    ok, ids, err = _post_upload_items(api_url, headers, items)
    if ok:
        return ids
    if len(items) == 1:
        bad_ids.append(items[0]["id"])
        return []
    mid = len(items) // 2
    return _upload_with_fallback(api_url, headers, items[:mid], bad_ids) + \
           _upload_with_fallback(api_url, headers, items[mid:], bad_ids)


# ================================
#   API: Auth & User
# ================================
def get_token(api_url: str, username: str, password: str) -> Optional[str]:
    """POST /token — get auth token."""
    try:
        resp = requests.post(
            f"{api_url}/token",
            data={"username": username, "password": password, "scope": "predict upload search"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            _ui_msg("error", "api.token.failed_http", status=resp.status_code, text=resp.text)
            return None
        return (resp.json() or {}).get("access_token")
    except requests.exceptions.ConnectionError:
        _ui_msg("error", "api.conn.failed", url=api_url)
        return None
    except Exception as e:
        _ui_msg("error", "api.token.failed_generic", error=str(e))
        return None

def get_current_user(api_url: str, token: str):
    """GET /users/me — current user profile."""
    data, err, status = _json_get(f"{api_url}/users/me", _headers(token), timeout=30)
    if err:
        _ui_msg("error", "api.user.fetch_failed", status=status, text=err)
        return None
    return data


# ================================
#   API: Predict & Search
# ================================
def classify_request(subject: str, description: str, token: str, api_url: str):
    """POST /predict — single prediction."""
    payload = {
        "id": str(uuid.uuid4()),
        "subject": _safe_str(subject or "no_subject", "no_subject"),
        "description": _safe_str(description or "no_description", "no_description"),
    }
    data, err, status = _json_post(f"{api_url}/predict", payload, _headers(token), timeout=60)
    if err:
        _ui_msg("error", "api.predict.failed", status=status, text=err)
        return None
    return data

def search_similar(subject: str, description: str, token: str, api_url: str, limit: int = 10):
    """POST /search — similar docs."""
    payload = {
        "id": str(uuid.uuid4()),
        "subject": _safe_str(subject or "no_subject", "no_subject"),
        "description": _safe_str(description or "no_description", "no_description"),
        "limit": int(limit),
    }
    data, err, status = _json_post(f"{api_url}/search", payload, _headers(token), timeout=60)
    if err:
        _ui_msg("error", "api.search.failed", status=status, text=err)
        return None
    return data


# ================================
#   API: Index & Upload
# ================================
def clear_index(token: str, api_url: str) -> bool:
    """POST /clear_index — clear index."""
    logger.info("clear_index: request -> backend")
    data, err, status = _json_post(f"{api_url}/clear_index", {}, _headers(token), timeout=60)
    if err:
        logger.error("clear_index: %s", err)
        _ui_msg("error", "api.index.clear_failed", status=status, text=err)
        return False
    if not data.get("success"):
        logger.warning("clear_index: unexpected response %s", data)
        _ui_msg("warning", "api.index.clear_unexpected", data=str(data))
        return False
    _ui_msg("success", "api.index.cleared")
    return True

def upload_data(data: pd.DataFrame, token: str, api_url: str) -> List[str]:
    """
    Robust upload: normalize IDs, ensure uniqueness inside batch, fallback with bisection.
    Returns list of uploaded ids.
    """
    headers = _headers(token)
    batch_size = 50
    total = len(data)
    total_batches = total // batch_size + (1 if total % batch_size > 0 else 0)

    uploaded_ids_total: List[str] = []
    bad_ids_total: List[str] = []
    progress_bar = st.progress(0)

    for i in range(total_batches):
        start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, total)
        batch_data = data.iloc[start_idx:end_idx]

        seen_ids = set()
        items = []
        for _, row in batch_data.iterrows():
            item = _prepare_item(row)
            if item["id"] in seen_ids:
                item["id"] = f"{item['id']}-{uuid.uuid4()}"
            seen_ids.add(item["id"])
            items.append(item)

        uploaded_ids = _upload_with_fallback(api_url, headers, items, bad_ids_total)
        uploaded_ids_total.extend(uploaded_ids)

        if len(uploaded_ids) != len(items):
            _ui_msg("warning", "api.upload.batch_partial",
                    i=i + 1, n=total_batches, ok=len(uploaded_ids), total=len(items),
                    skipped=len(items) - len(uploaded_ids))

        progress_bar.progress(min(1.0, (i + 1) / total_batches))

    if bad_ids_total:
        _ui_msg("error", "api.upload.skipped_many", n=len(bad_ids_total), preview=str(bad_ids_total[:50]))
    _ui_msg("success", "api.upload.done", n=len(uploaded_ids_total))
    return uploaded_ids_total


# ================================
#   Legacy predict loop (kept for compat)
# ================================
def predict(data: pd.DataFrame, token: str, api_url: str):
    headers = _headers(token)
    predictions, empty_cnt, err_cnt = [], 0, 0
    progress_bar = st.progress(0)
    total_rows = len(data)

    for i, (_, row) in enumerate(data.iterrows()):
        payload = {
            "id": str(uuid.uuid4()),
            "subject": _safe_str(row.get("subject"), "no_subject"),
            "description": _safe_str(row.get("description"), "no_description"),
        }
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            preds = result.get("predictions")
            if preds is None:
                predictions.append(None); empty_cnt += 1
            elif isinstance(preds, dict):
                top = preds.get("class_name")
                predictions.append(top if top is not None else None)
                if top is None: empty_cnt += 1
            elif isinstance(preds, list) and preds and isinstance(preds[0], dict):
                top = preds[0].get("class_name")
                predictions.append(top if top is not None else None)
                if top is None: empty_cnt += 1
            else:
                predictions.append(None); empty_cnt += 1
        except Exception:
            err_cnt += 1
            predictions.append(None)

        progress_bar.progress(min(1.0, (i + 1) / total_rows))

    if empty_cnt or err_cnt:
        _ui_msg("warning", "api.predict.loop_warnings", n=empty_cnt + err_cnt, total=total_rows, empty=empty_cnt, errors=err_cnt)
    return predictions


# ================================
#   API: Backend metrics
# ================================
def compute_metrics_backend(test_df: pd.DataFrame, token: str, api_url: str) -> Optional[Dict[str, Any]]:
    """POST /metrics/compute — compute & persist metrics report for current user."""
    items = _df_to_metrics_items(test_df)
    data, err, status = _json_post(f"{api_url}/metrics/compute", {"items": items}, _headers(token), timeout=120)
    if err:
        _ui_msg("error", "api.metrics.compute_failed", status=status, text=err)
        return None
    return data

def get_last_metrics(token: str, api_url: str) -> Optional[Dict[str, Any]]:
    """GET /metrics/latest — last report for current user."""
    data, err, status = _json_get(f"{api_url}/metrics/latest", _headers(token), timeout=60)
    if status == 404:
        return None
    if err:
        _ui_msg("error", "api.metrics.latest_failed", status=status, text=err)
        return None
    return data


# ================================
#   Report rendering helpers
# ================================
def get_classification_report_df(report_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(report_dict).transpose()
    keep = ["precision", "recall", "f1-score", "support"]
    df = df[[c for c in keep if c in df.columns]]
    for c in ["precision", "recall", "f1-score"]:
        if c in df.columns:
            df[c] = df[c].round(2)
    if "support" in df.columns:
        df["support"] = df["support"].astype(int)
    return df

def plot_confusion_matrix(cm, class_names=None):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    if class_names is None and cm.shape[0] <= 20:
        class_names = [str(i) for i in range(cm.shape[0])]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
        ax=ax,
    )
    ax.set_xlabel(_T("data_upload.metrics.predicted"))
    ax.set_ylabel(_T("data_upload.metrics.true"))
    ax.set_title(_T("data_upload.metrics.cm_title"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_classification_metrics(report_dict: dict):
    classes = [k for k in report_dict.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    metrics_data = {"Class": [], "Metric": [], "Value": []}
    for cls in classes:
        for metric in ["precision", "recall", "f1-score"]:
            metrics_data["Class"].append(cls)
            metrics_data["Metric"].append(metric)
            metrics_data["Value"].append(report_dict[cls][metric])
    df = pd.DataFrame(metrics_data)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="Class", y="Value", hue="Metric", data=df, ax=ax)
    ax.set_title(_T("data_upload.metrics.by_class_plot_title"))
    ax.set_ylabel(_T("data_upload.metrics.metric_label"))
    ax.set_xlabel(_T("data_upload.metrics.class_label"))
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=_T("data_upload.metrics.metric_label"))
    plt.tight_layout()
    return fig


# ================================
#   High-quality filter pipeline
# ================================
def filter_high_quality_classes(
    df: pd.DataFrame,
    min_samples: int = 10,
    min_f1_score: float = 0.5,
    api_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
):
    """
    Quality-driven filtering using backend /metrics/compute.
    Returns (filtered_df, report_dict) or (None, None) on error.
    """
    st.subheader(_T("data_upload.filter.quality_analysis.title"))

    df_processed = df.copy()
    class_counts = df_processed["class"].value_counts()
    rare = class_counts[class_counts < min_samples].index.tolist()

    df_processed["class"] = df_processed["class"].apply(lambda x: "Others" if x in rare else x)

    with st.expander(_T("data_upload.after_filter_stats.title"), expanded=True):
        new_counts = df_processed["class"].value_counts()
        stats_df = pd.DataFrame({
            _T("data_upload.common.value"): new_counts.index,
            _T("data_upload.common.count"): new_counts.values,
            _T("data_upload.common.percent"): (new_counts.values / new_counts.sum() * 100).round(2),
        })
        st.dataframe(stats_df)
        fig, ax = plt.subplots(figsize=(10, 6))
        new_counts.plot(kind="bar", ax=ax)
        ax.set_title(_T("data_upload.after_filter_stats.chart_title"))
        ax.set_xlabel(_T("data_upload.common.value"))
        ax.set_ylabel(_T("data_upload.common.count"))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    train_df, test_df = train_test_split(df_processed, test_size=0.05, random_state=42, stratify=df_processed["class"])

    token = get_token(api_url, username, password)
    if not token:
        _ui_msg("error", "api.token.failed_for_quality")
        return None, None

    with st.spinner(_T("data_upload.info.clearing")):
        clear_index(token, api_url)

    with st.spinner(_T("data_upload.spinner.upload")):
        upload_data(train_df, token, api_url)

    with st.spinner(_T("data_upload.spinner.predict")):
        metrics = compute_metrics_backend(test_df, token, api_url)
        if not metrics:
            _ui_msg("error", "api.metrics.compute_empty")
            return None, None

    report_dict = metrics.get("classification_report_dict", {})
    if not report_dict:
        _ui_msg("error", "api.metrics.report_empty")
        return None, None

    high_quality = []
    for cls, m in report_dict.items():
        if cls in ["accuracy", "macro avg", "weighted avg"]:
            continue
        if m.get("f1-score", 0.0) >= float(min_f1_score):
            high_quality.append(cls)

    with st.expander(_T("data_upload.metrics.by_class_plot"), expanded=True):
        metrics_df = pd.DataFrame([
            {
                _T("data_upload.metrics.class_label"): name,
                "Precision": report_dict[name].get("precision", 0.0),
                "Recall": report_dict[name].get("recall", 0.0),
                "F1-score": report_dict[name].get("f1-score", 0.0),
                "Support": report_dict[name].get("support", 0),
                _T("data_upload.filter.quality_label"): _T("data_upload.quality.high") if name in high_quality else _T("data_upload.quality.low"),
            }
            for name in report_dict.keys() if name not in ["accuracy", "macro avg", "weighted avg"]
        ])
        metrics_df = metrics_df.sort_values("F1-score", ascending=False)
        for c in ["Precision", "Recall", "F1-score"]:
            metrics_df[c] = metrics_df[c].round(3)
        st.dataframe(metrics_df)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = metrics_df[_T("data_upload.filter.quality_label")].map({
            _T("data_upload.quality.high"): "green",
            _T("data_upload.quality.low"): "red",
        })
        ax.bar(metrics_df[_T("data_upload.metrics.class_label")], metrics_df["F1-score"], color=colors)
        ax.axhline(y=min_f1_score, color="red", linestyle="--", alpha=0.7)
        ax.text(0, min_f1_score + 0.01, _T("data_upload.filter.f1_threshold", thr=min_f1_score), color="red")
        ax.set_title(_T("data_upload.metrics.by_class_plot_title"))
        ax.set_xlabel(_T("data_upload.metrics.class_label"))
        ax.set_ylabel("F1-score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    df_processed["class"] = df_processed["class"].apply(lambda x: x if x in high_quality else "Others")
    df_high_quality = df_processed

    with st.expander(_T("data_upload.filter.quality_stats_after"), expanded=True):
        cnt = df_high_quality["class"].value_counts()
        stats_df = pd.DataFrame({
            _T("data_upload.common.value"): cnt.index,
            _T("data_upload.common.count"): cnt.values,
            _T("data_upload.common.percent"): (cnt.values / cnt.sum() * 100).round(2),
        })
        st.dataframe(stats_df)
        fig, ax = plt.subplots(figsize=(10, 6))
        cnt.plot(kind="bar", ax=ax)
        ax.set_title(_T("data_upload.after_filter_stats.chart_title"))
        ax.set_xlabel(_T("data_upload.common.value"))
        ax.set_ylabel(_T("data_upload.common.count"))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    st.info(_T("data_upload.quality.summary.total_classes", n=len(class_counts)))
    st.info(_T("data_upload.quality.summary.after_rare", n=df_processed["class"].nunique()))
    st.info(_T("data_upload.quality.summary.high", thr=min_f1_score, n=len(high_quality)))
    st.info(_T("data_upload.quality.summary.rows_kept", kept=len(df_high_quality), total=len(df_processed),
               pct=f"{len(df_high_quality)/max(1,len(df_processed))*100:.1f}%"))

    return df_high_quality, report_dict


# ================================
#   Retrain API
# ================================
def _validate_retrain_cfg(cfg: Dict[str, Any]) -> Optional[str]:
    """Light client-side shape check before /retrain/push."""
    try:
        if not isinstance(cfg, dict):
            return _T("api.retrain.err.cfg_not_dict")
        required_root = [
            "data_api", "id_col", "subject_col", "description_col", "class_col",
            "filter_method", "top_n_values", "min_samples", "min_f1_score",
            "clear_index_flag", "RUN_EVERY_DAYS", "ANCHOR_DATETIME_STR", "RUN_ON_START",
        ]
        for k in required_root:
            if k not in cfg:
                return _T("api.retrain.err.missing_field", field=k)

        da = cfg["data_api"]
        if not isinstance(da, dict):
            return _T("api.retrain.err.data_api_not_dict")
        if "url" not in da or not isinstance(da["url"], str) or not da["url"]:
            return _T("api.retrain.err.data_api_url_required")
        if "auth" not in da or not isinstance(da["auth"], dict):
            return _T("api.retrain.err.data_api_auth_required")
        if "method" not in da["auth"]:
            return _T("api.retrain.err.data_api_auth_method_required")

        if cfg["filter_method"] not in ("by_freq", "by_quality"):
            return _T("api.retrain.err.bad_filter_method")

        if not isinstance(cfg["top_n_values"], int) or cfg["top_n_values"] < 1:
            return _T("api.retrain.err.top_n")
        if not isinstance(cfg["min_samples"], int) or cfg["min_samples"] < 1:
            return _T("api.retrain.err.min_samples")
        if not isinstance(cfg["min_f1_score"], (int, float)) or not (0.0 <= float(cfg["min_f1_score"]) <= 1.0):
            return _T("api.retrain.err.min_f1")
        if not isinstance(cfg["clear_index_flag"], bool):
            return _T("api.retrain.err.clear_index_flag")
        if not isinstance(cfg["RUN_EVERY_DAYS"], str) or not cfg["RUN_EVERY_DAYS"].isdigit():
            return _T("api.retrain.err.run_every_days")
        if not isinstance(cfg["ANCHOR_DATETIME_STR"], str) or len(cfg["ANCHOR_DATETIME_STR"]) < 10:
            return _T("api.retrain.err.anchor_dt")
        if cfg["RUN_ON_START"] not in ("0", "1"):
            return _T("api.retrain.err.run_on_start")
    except Exception as e:
        return _T("api.retrain.err.validation_generic", error=str(e))
    return None

def retrain_push(cfg: Dict[str, Any], token: str, api_url: str) -> Optional[Dict[str, Any]]:
    """POST /retrain/push — save user's retrain config."""
    err = _validate_retrain_cfg(cfg)
    if err:
        _ui_msg("error", "api.retrain.invalid_cfg", err=err)
        return None
    data, http_err, status = _json_post(f"{api_url}/retrain/push", cfg, _headers(token), timeout=60)
    if http_err or not data:
        _ui_msg("error", "api.retrain.push_failed", status=status, text=http_err or "")
        return None
    if data.get("ok"):
        _ui_msg("success", "api.retrain.saved", path=data.get("path", ""), ts=data.get("updated_at", ""))
    else:
        _ui_msg("warning", "api.retrain.push_unexpected", data=str(data))
    return data

def retrain_pull(token: str, api_url: str) -> Optional[Dict[str, Any]]:
    """GET /retrain/pull — get stored config for current user."""
    data, err, status = _json_get(f"{api_url}/retrain/pull", _headers(token), timeout=30)
    if status == 404:
        return None
    if err or not isinstance(data, dict):
        _ui_msg("error", "api.retrain.pull_failed", status=status, text=err or str(data))
        return None
    return data

def retrain_delete(token: str, api_url: str) -> Optional[Dict[str, Any]]:
    """DELETE /retrain/delete — delete stored config."""
    try:
        resp = requests.delete(f"{api_url}/retrain/delete", headers=_headers(token), timeout=30)
        if resp.status_code != 200:
            _ui_msg("error", "api.retrain.delete_failed", status=resp.status_code, text=resp.text)
            return None
        return resp.json()
    except Exception as e:
        _ui_msg("error", "api.retrain.delete_failed_generic", error=str(e))
        return None
