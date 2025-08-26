import uuid
import requests
import streamlit as st
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
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
                return str(int(val))          # 78617.0 -> "78617"
            return f"{val:.15g}"              # 7.8617e+04 -> "78617"
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
    """Функция для получения токена авторизации"""
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
            st.error(f"Ошибка аутентификации: {response.text}")
            return None

        return response.json()["access_token"]
    except requests.exceptions.ConnectionError:
        st.error(f"Не удалось подключиться к API по адресу {api_url}")
        return None

def classify_request(subject, description, token, api_url):
    """Функция для классификации запроса"""
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
    """Поиск похожих документов на основе темы и описания"""
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
    """Очистка индекса перед загрузкой новых данных"""
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

    # Базовый размер батча; при проблемах fallback дробит дальше
    batch_size = 50
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)

    uploaded_ids_total: list[str] = []
    bad_ids_total: list[str] = []
    progress_bar = st.progress(0)

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = data.iloc[start_idx:end_idx]

        # Сбор и нормализация элементов
        items = []
        seen_ids = set()
        for _, row in batch_data.iterrows():
            item = _prepare_item(row)
            # уникальность id в рамках батча
            if item["id"] in seen_ids:
                item["id"] = f"{item['id']}-{uuid.uuid4()}"
            seen_ids.add(item["id"])
            items.append(item)

        # Пытаемся загрузить; при ошибке дробим
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
    """Получение предсказаний (устойчиво к пустым/неверным ответам)."""
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
            # допускаем словарь, список или пусто
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

# ---------------------- Metrics & plots ----------------------
def calculate_metrics(y_true, y_pred):
    """Вычисление метрик качества классификации (устойчиво к пустым)."""
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if not y_true_valid:
        logger.error("Нет валидных предсказаний для оценки")
        return None, None, None, None, []

    acc = accuracy_score(y_true_valid, y_pred_valid)

    # zero_division=0, чтобы убрать UndefinedMetricWarning
    report_dict = classification_report(y_true_valid, y_pred_valid, output_dict=True, zero_division=0)
    report_text = classification_report(y_true_valid, y_pred_valid, zero_division=0)
    cm = confusion_matrix(y_true_valid, y_pred_valid)
    classes = sorted(list(set(y_true_valid + y_pred_valid)))

    return acc, report_dict, report_text, cm, classes

def get_classification_report_df(report_dict):
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
    """Построение матрицы ошибок"""
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
    """Создание визуализации метрик классификации"""
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_data = {'Class': [], 'Metric': [], 'Value': []}
    for cls in classes:
        for metric in ['precision', 'recall', 'f1-score']:
            metrics_data['Class'].append(cls)
            metrics_data['Metric'].append(metric)
            metrics_data['Value'].append(report_dict[cls][metric])

    df = pd.DataFrame(metrics_data)
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = {'precision': 'blue', 'recall': 'green', 'f1-score': 'red'}
    sns.barplot(x='Class', y='Value', hue='Metric', data=df, palette=palette, ax=ax)

    plt.title('Метрики классификации по классам')
    plt.ylabel('Значение')
    plt.xlabel('Класс')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Метрика')
    plt.tight_layout()
    return fig

def save_metrics(metrics_dir, filename, acc, report_dict, report_text, cm, class_names=None):
    """Сохранение метрик в файл"""
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"metrics_{timestamp}.json"

    metrics_data = {
        "filename": filename,
        "timestamp": timestamp,
        "accuracy": float(acc),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "classes": class_names if class_names else []
    }

    with open(os.path.join(metrics_dir, metrics_filename), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=4)

    fig_cm = plot_confusion_matrix(cm, class_names)
    fig_cm.savefig(os.path.join(metrics_dir, f"confusion_matrix_{timestamp}.png"))

    fig_metrics = plot_classification_metrics(report_dict)
    fig_metrics.savefig(os.path.join(metrics_dir, f"classification_metrics_{timestamp}.png"))

    return metrics_filename

# ---------------------- High-quality filter pipeline ----------------------
def filter_high_quality_classes(df, min_samples=10, min_f1_score=0.5, api_url=None, username=None, password=None):
    """
    Фильтрация классов с высоким качеством классификации.
    """
    st.subheader("Анализ качества классификации по классам")
    st.write(f'''Суть метода в том чтобы оставить те классы на которых мы уже хорошо работаем. 
             Для этого мы сначала выносим очень редкие наблюдения меньше {min_samples} в отдельную категорию
             Затем строим на этих данных модель и оставляем только те классы на которых качество получилось больше {min_f1_score} и уже после этого загружаем в базу только те классы на которых хорошо рабоатет''')

    # Копируем исходный DataFrame
    df_processed = df.copy()

    # Подсчитываем количество образцов для каждого класса
    class_counts = df_processed["class"].value_counts()

    # Определяем редкие классы (менее min_samples образцов)
    rare_classes = class_counts[class_counts < min_samples].index.tolist()

    # Заменяем редкие классы на "Others"
    df_processed["class"] = df_processed["class"].apply(
        lambda x: "Others" if x in rare_classes else x
    )

    # Отображаем статистику после замены
    with st.expander("Статистика после фильтрации редких классов", expanded=True):
        st.write(' ')
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

    # Разбиваем данные на обучающую и тестовую выборки
    train_df, test_df = train_test_split(
        df_processed, test_size=0.05, random_state=42, stratify=df_processed["class"]
    )

    # Получаем токен для доступа к API
    token = get_token(api_url, username, password)
    if not token:
        st.error("Не удалось получить токен для доступа к API")
        return None, None

    # Очищаем индекс перед загрузкой
    with st.spinner("Очистка существующего индекса..."):
        clear_index(token, api_url)

    # Загружаем обучающую выборку
    with st.spinner("Загрузка обучающих данных в систему..."):
        upload_data(train_df, token, api_url)

    # Получаем предсказания для тестовой выборки
    with st.spinner("Получение предсказаний для тестовой выборки..."):
        preds = predict(test_df, token, api_url)
        if all(p is None for p in preds):
            st.error("API не вернул ни одного валидного предсказания для тестовой выборки — пропускаю оценку.")
            return None, None

    # Вычисляем метрики
    y_true = test_df["class"].tolist()
    y_pred = preds

    # Валидные предсказания
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    report_dict = classification_report(y_true_valid, y_pred_valid, output_dict=True, zero_division=0)

    # Фильтруем классы с высоким F1-score
    high_quality_classes = []
    for class_name, metrics in report_dict.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            if metrics['f1-score'] >= min_f1_score:
                high_quality_classes.append(class_name)

    # Отображаем метрики по классам
    with st.expander("Метрики по классам", expanded=True):
        metrics_df = pd.DataFrame([
            {
                'Класс': class_name,
                'Precision': report_dict[class_name]['precision'],
                'Recall': report_dict[class_name]['recall'],
                'F1-score': report_dict[class_name]['f1-score'],
                'Support': report_dict[class_name]['support'],
                'Качество': 'Высокое' if class_name in high_quality_classes else 'Низкое'
            }
            for class_name in report_dict.keys()
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']
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

    # Фильтруем данные, оставляя только классы с высоким качеством
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
    st.info(f"Количество классов после фильтрации редких: {len(new_counts)}")
    st.info(f"Количество классов с высоким качеством (F1-score >= {min_f1_score}): {len(high_quality_classes)}")
    st.info(f"Количество записей в отфильтрованном наборе: {len(df_high_quality)} из {len(df_processed)} ({len(df_high_quality)/len(df_processed)*100:.1f}%)")

    return df_high_quality, report_dict
