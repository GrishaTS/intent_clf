# frontend/api_utils.py  (обновлено под backend /metrics)

import uuid
import requests
import streamlit as st
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # оставляем для локального сплита
import json
import os
from datetime import datetime

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------- Helpers ----------------------
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
        return str(val).strip() or str(uuid.uuid4())
    except Exception:
        return str(uuid.uuid4())

def _safe_str(v: object, default: str) -> str:
    """Приводит значение к строке, возвращает default для None/пустых/ошибок."""
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

def _prepare_item(row) -> dict:
    """Собирает валидный item для /upload из строки DataFrame."""
    item_id = _to_str_id(row["id"] if "id" in row else None)
    subject = _safe_str(row["subject"] if "subject" in row and not pd.isna(row["subject"]) else "no_subject", "no_subject")
    description = _safe_str(row["description"] if "description" in row and not pd.isna(row["description"]) else "no_description", "no_description")
    class_name = _safe_str(row["class"] if "class" in row and not pd.isna(row["class"]) else "Others", "Others")

    # Ограничиваем длину текстов
    subject = _truncate(subject, 500)
    description = _truncate(description, 5000)

    item = {
        "id": item_id,
        "subject": subject,
        "description": description,
        "class_name": class_name,
    }
    if "task" in row and not pd.isna(row["task"]):
        item["task"] = _safe_str(row["task"], "")
    return item

def _df_to_metrics_items(df: pd.DataFrame) -> list[dict]:
    """Готовит payload для /metrics/compute из test_df."""
    items = []
    for _, row in df.iterrows():
        items.append({
            "id": _to_str_id(row["id"] if "id" in row else None),
            "subject": _safe_str(row["subject"] if "subject" in row and not pd.isna(row["subject"]) else "no_subject", "no_subject"),
            "description": _safe_str(row["description"] if "description" in row and not pd.isna(row["description"]) else "no_description", "no_description"),
            "label": _safe_str(row["class"] if "class" in row and not pd.isna(row["class"]) else "Others", "Others"),
        })
    return items

def _post_upload_items(api_url: str, headers: dict, items: list):
    """
    Пытается отправить пачку items.
    Возвращает (ok: bool, uploaded_ids: list[str], error_text: str).
    Если сервер не вернул ids, используем локальные id.
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

def _upload_with_fallback(api_url: str, headers: dict, items: list, bad_ids: list) -> list:
    """
    Загрузка с прогрессивным фолбэком: весь батч -> половинки -> одиночные.
    Возвращает список успешно загруженных id; неуспешные копит в bad_ids.
    """
    ok, ids, err = _post_upload_items(api_url, headers, items)
    if ok:
        return ids

    if len(items) == 1:
        bad_ids.append(items[0]["id"])
        return []

    mid = len(items) // 2
    left = items[:mid]
    right = items[mid:]
    uploaded_ids = []
    uploaded_ids.extend(_upload_with_fallback(api_url, headers, left, bad_ids))
    uploaded_ids.extend(_upload_with_fallback(api_url, headers, right, bad_ids))
    return uploaded_ids

# ---------------------- API calls ----------------------
def get_token(api_url, username, password):
    """Получение токена авторизации"""
    try:
        response = requests.post(
            f"{api_url}/token",
            data={
                "username": username,
                "password": password,
                "scope": "predict upload search",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code != 200:
            return None
        return response.json()["access_token"]
    except requests.exceptions.ConnectionError:
        st.error(f"Не удалось подключиться к API по адресу {api_url}")
        return None

def get_current_user(api_url: str, token: str):
    """Запрос текущего пользователя по токену."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        resp = requests.get(f"{api_url}/users/me", headers=headers, timeout=30)
        if resp.status_code != 200:
            st.error(f"Ошибка при запросе профиля пользователя: {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        st.error(f"Ошибка подключения к API (/users/me): {str(e)}")
        return None

def classify_request(subject, description, token, api_url):
    """Сингл-предсказание через /predict"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "id": str(uuid.uuid4()),
        "subject": _safe_str(subject if subject else "no_subject", "no_subject"),
        "description": _safe_str(description if description else "no_description", "no_description"),
    }
    try:
        response = requests.post(f"{api_url}/predict", json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при классификации: {str(e)}")
        return None

def search_similar(subject, description, token, api_url, limit=10):
    """Поиск похожих документов"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "id": str(uuid.uuid4()),
        "subject": _safe_str(subject if subject else "no_subject", "no_subject"),
        "description": _safe_str(description if description else "no_description", "no_description"),
        "limit": int(limit),
    }
    try:
        response = requests.post(f"{api_url}/search", json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при поиске: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            st.error(f"Ответ сервера: {e.response.text}")
        return None

def clear_index(token, api_url):
    """Очистка индекса"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        logger.info("Отправка запроса на очистку индекса")
        response = requests.post(f"{api_url}/clear_index", headers=headers, timeout=60)
        if response.status_code != 200:
            logger.error(f"Ошибка при очистке индекса: {response.text}")
            st.error(f"Ошибка при очистке индекса: {response.text}")
            return False
        result = response.json()
        if result.get("success"):
            logger.info("Индекс успешно очищен")
            st.success("Индекс успешно очищен")
            return True
        else:
            logger.warning(f"API вернул неожиданный ответ при очистке индекса: {result}")
            st.warning(f"API вернул неожиданный ответ при очистке индекса: {result}")
            return False
    except Exception as e:
        logger.error(f"Ошибка при очистке индекса: {str(e)}")
        st.error(f"Ошибка при очистке индекса: {str(e)}")
        return False

def upload_data(data, token, api_url):
    """Загрузка данных в систему (устойчиво; id всегда строка; fallback при ошибках)."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    batch_size = 50
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)

    uploaded_ids_total: list[str] = []
    bad_ids_total: list[str] = []
    progress_bar = st.progress(0)

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = data.iloc[start_idx:end_idx]

        items = []
        seen_ids = set()
        for _, row in batch_data.iterrows():
            item = _prepare_item(row)
            if item["id"] in seen_ids:
                item["id"] = f"{item['id']}-{uuid.uuid4()}"
            seen_ids.add(item["id"])
            items.append(item)

        uploaded_ids = _upload_with_fallback(api_url, headers, items, bad_ids_total)
        uploaded_ids_total.extend(uploaded_ids)

        if len(uploaded_ids) != len(items):
            st.warning(f"Батч {i + 1}/{total_batches}: загружено {len(uploaded_ids)}/{len(items)}, "
                       f"пропущено {len(items) - len(uploaded_ids)}")

        progress_bar.progress(min(1.0, (i + 1) / total_batches))

    if bad_ids_total:
        st.error(f"Пропущено записей: {len(bad_ids_total)}. Проблемные id (первые 50): {bad_ids_total[:50]}")

    st.success(f"Загружено {len(uploaded_ids_total)} записей")
    return uploaded_ids_total

def predict(data, token, api_url):
    """
    Старый путь получения предсказаний с фронта (оставлено для обратной совместимости).
    Новый рекомендованный путь для оценки качества — /metrics/compute.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    predictions = []
    progress_bar = st.progress(0)
    total_rows = len(data)
    empty_cnt = 0
    err_cnt = 0

    for i, (index, row) in enumerate(data.iterrows()):
        try:
            payload = {
                "id": str(uuid.uuid4()),
                "subject": _safe_str(row["subject"] if "subject" in row and not pd.isna(row["subject"]) else "no_subject", "no_subject"),
                "description": _safe_str(row["description"] if "description" in row and not pd.isna(row["description"]) else "no_description", "no_description"),
            }

            response = requests.post(f"{api_url}/predict", json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()

            preds = result.get("predictions")
            if preds is None:
                predictions.append(None); empty_cnt += 1
            elif isinstance(preds, dict):
                top = preds.get("class_name")
                predictions.append(top if top is not None else None)
                if top is None: empty_cnt += 1
            elif isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
                top = preds[0].get("class_name")
                predictions.append(top if top is not None else None)
                if top is None: empty_cnt += 1
            else:
                predictions.append(None); empty_cnt += 1

        except requests.exceptions.RequestException:
            err_cnt += 1
            predictions.append(None)
        except Exception:
            err_cnt += 1
            predictions.append(None)

        progress_bar.progress(min(1.0, (i + 1) / total_rows))

    if empty_cnt or err_cnt:
        st.warning(f"Предупреждение: пустых/ошибочных ответов {empty_cnt + err_cnt} из {total_rows} "
                   f"(empty={empty_cnt}, errors={err_cnt})")

    return predictions

# ---------------------- Backend metrics API ----------------------
def compute_metrics_backend(test_df: pd.DataFrame, token: str, api_url: str) -> dict | None:
    """POST /metrics/compute — считает метрики на сервере и сохраняет {user_id}.json"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    items = _df_to_metrics_items(test_df)
    try:
        resp = requests.post(f"{api_url}/metrics/compute", json={"items": items}, headers=headers, timeout=120)
        if resp.status_code != 200:
            st.error(f"Ошибка /metrics/compute: {resp.status_code} {resp.text}")
            return None
        st.error(f"{resp.json()}")
        return resp.json()
    except Exception as e:
        st.error(f"Ошибка запроса /metrics/compute: {e}")
        return None

def get_last_metrics(token: str, api_url: str) -> dict | None:
    """GET /metrics/latest — получить последний отчёт для текущего пользователя"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        resp = requests.get(f"{api_url}/metrics/latest", headers=headers, timeout=60)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            st.error(f"Ошибка /metrics/latest: {resp.status_code} {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        st.error(f"Ошибка запроса /metrics/latest: {e}")
        return None

# ---------------------- Report rendering helpers ----------------------
def get_classification_report_df(report_dict: dict) -> pd.DataFrame:
    """Преобразование словаря classification_report в DataFrame"""
    df = pd.DataFrame(report_dict).transpose()
    columns_to_keep = ['precision', 'recall', 'f1-score', 'support']
    df = df[[col for col in columns_to_keep if col in df.columns]]
    for col in ['precision', 'recall', 'f1-score']:
        if col in df.columns:
            df[col] = df[col].round(2)
    if 'support' in df.columns:
        df['support'] = df['support'].astype(int)
    return df

def plot_confusion_matrix(cm, class_names=None):
    """Построение матрицы ошибок из списка/np.array"""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    if class_names is None and cm.shape[0] <= 20:
        class_names = [str(i) for i in range(cm.shape[0])]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )

    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_classification_metrics(report_dict):
    """Создание визуализации per-class метрик из dict"""
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_data = {'Class': [], 'Metric': [], 'Value': []}
    for cls in classes:
        for metric in ['precision', 'recall', 'f1-score']:
            metrics_data['Class'].append(cls)
            metrics_data['Metric'].append(metric)
            metrics_data['Value'].append(report_dict[cls][metric])

    df = pd.DataFrame(metrics_data)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Class', y='Value', hue='Metric', data=df, ax=ax)

    plt.title('Метрики классификации по классам')
    plt.ylabel('Значение')
    plt.xlabel('Класс')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Метрика')
    plt.tight_layout()
    return fig

# ---------------------- High-quality filter pipeline ----------------------
def filter_high_quality_classes(df, min_samples=10, min_f1_score=0.5, api_url=None, username=None, password=None):
    """
    Фильтрация классов по качеству с использованием backend /metrics/compute.
    """
    st.subheader("Анализ качества классификации по классам")

    # Копируем исходный DataFrame
    df_processed = df.copy()

    # Подсчитываем количество образцов для каждого класса
    class_counts = df_processed["class"].value_counts()

    # Определяем редкие классы
    rare_classes = class_counts[class_counts < min_samples].index.tolist()

    # Заменяем редкие классы на "Others"
    df_processed["class"] = df_processed["class"].apply(
        lambda x: "Others" if x in rare_classes else x
    )

    # Статистика после фильтрации редких
    with st.expander("Статистика после фильтрации редких классов", expanded=True):
        new_counts = df_processed["class"].value_counts()
        new_stats_df = pd.DataFrame({
            'Значение': new_counts.index,
            'Количество': new_counts.values,
            'Процент': (new_counts.values / new_counts.sum() * 100).round(2)
        })
        st.dataframe(new_stats_df)
        fig, ax = plt.subplots(figsize=(10, 6))
        new_counts.plot(kind='bar', ax=ax)
        plt.title('Распределение значений после фильтрации редких классов')
        plt.xlabel('Значение')
        plt.ylabel('Количество')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    # Разбиваем данные
    train_df, test_df = train_test_split(
        df_processed, test_size=0.05, random_state=42, stratify=df_processed["class"]
    )

    # Токен
    token = get_token(api_url, username, password)
    if not token:
        st.error("Не удалось получить токен для доступа к API")
        return None, None

    # Очищаем индекс
    with st.spinner("Очистка существующего индекса..."):
        clear_index(token, api_url)

    # Загружаем train
    with st.spinner("Загрузка обучающих данных в систему..."):
        upload_data(train_df, token, api_url)

    # Считаем метрики на backend по test
    with st.spinner("Расчёт метрик на тестовой выборке (backend)..."):
        metrics = compute_metrics_backend(test_df, token, api_url)
        if not metrics:
            st.error("Не удалось получить метрики с бэкенда.")
            return None, None

    report_dict = metrics.get("classification_report_dict", {})
    if not report_dict:
        st.error("Пустой отчёт классификации.")
        return None, None

    # Фильтруем классы с высоким F1-score
    high_quality_classes = []
    for class_name, m in report_dict.items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        if m.get('f1-score', 0.0) >= min_f1_score:
            high_quality_classes.append(class_name)

    # Визуализация метрик
    with st.expander("Метрики по классам", expanded=True):
        metrics_df = pd.DataFrame([
            {
                'Класс': name,
                'Precision': report_dict[name].get('precision', 0.0),
                'Recall': report_dict[name].get('recall', 0.0),
                'F1-score': report_dict[name].get('f1-score', 0.0),
                'Support': report_dict[name].get('support', 0),
                'Качество': 'Высокое' if name in high_quality_classes else 'Низкое'
            }
            for name in report_dict.keys()
            if name not in ['accuracy', 'macro avg', 'weighted avg']
        ])
        metrics_df = metrics_df.sort_values('F1-score', ascending=False)
        for col in ['Precision', 'Recall', 'F1-score']:
            metrics_df[col] = metrics_df[col].round(3)
        st.dataframe(metrics_df)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = metrics_df['Качество'].map({'Высокое': 'green', 'Низкое': 'red'})
        ax.bar(metrics_df['Класс'], metrics_df['F1-score'], color=colors)
        ax.axhline(y=min_f1_score, color='red', linestyle='--', alpha=0.7)
        ax.text(0, min_f1_score + 0.01, f'Порог F1-score: {min_f1_score}', color='red')
        plt.title('F1-score по классам')
        plt.xlabel('Класс')
        plt.ylabel('F1-score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    # Финальная фильтрация набора
    df_processed["class"] = df_processed["class"].apply(
        lambda x: x if x in high_quality_classes else "Others"
    )
    df_high_quality = df_processed

    with st.expander("Статистика после фильтрации по качеству", expanded=True):
        high_quality_counts = df_high_quality["class"].value_counts()
        high_quality_stats_df = pd.DataFrame({
            'Значение': high_quality_counts.index,
            'Количество': high_quality_counts.values,
            'Процент': (high_quality_counts.values / high_quality_counts.sum() * 100).round(2)
        })
        st.dataframe(high_quality_stats_df)
        fig, ax = plt.subplots(figsize=(10, 6))
        high_quality_counts.plot(kind='bar', ax=ax)
        plt.title('Распределение значений после фильтрации по качеству')
        plt.xlabel('Значение')
        plt.ylabel('Количество')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    st.info(f"Исходное количество классов: {len(class_counts)}")
    st.info(f"Количество классов после фильтрации редких: {df_processed['class'].nunique()}")
    st.info(f"Количество классов с высоким качеством (F1-score >= {min_f1_score}): {len(high_quality_classes)}")
    st.info(f"Количество записей в отфильтрованном наборе: {len(df_high_quality)} из {len(df_processed)} ({len(df_high_quality)/len(df_processed)*100:.1f}%)")

    return df_high_quality, report_dict
