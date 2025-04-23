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

    # Генерируем ID для запроса
    item_id = str(uuid.uuid4())

    payload = {
        "id": item_id,  # Обязательное поле!
        "subject": subject if subject else "no_subject",
        "description": description if description else "no_description",
    }

    try:
        response = requests.post(f"{api_url}/predict", json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result
    except Exception as e:
        st.error(f"Ошибка при классификации: {str(e)}")
        return None

def search_similar(subject, description, token, api_url, limit=10):
    """Поиск похожих документов на основе темы и описания"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "id": str(uuid.uuid4()),
        "subject": subject if subject else "no_subject",
        "description": description if description else "no_description",
        "limit": limit,
    }

    try:
        response = requests.post(f"{api_url}/search", json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result
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
        response = requests.post(f"{api_url}/clear_index", headers=headers)
        
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
    """Загрузка данных в систему"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Разбиваем данные на батчи по 100 записей
    batch_size = 100
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)

    uploaded_ids = []
    progress_bar = st.progress(0)

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = data.iloc[start_idx:end_idx]

        # Подготавливаем данные для загрузки
        items = []
        for _, row in batch_data.iterrows():
            # Используем оригинальный ID из данных, если он есть
            item_id = row.get("id", str(uuid.uuid4()))

            item = {
                "id": item_id,
                "subject": row["subject"] if not pd.isna(row["subject"]) else "no_subject",
                "description": row["description"] if not pd.isna(row["description"]) else "no_description",
                "class_name": row["class"],
            }

            # Добавляем task, если есть в данных
            if "task" in row and not pd.isna(row["task"]):
                item["task"] = row["task"]

            items.append(item)

        payload = {"items": items}

        try:
            # Отправляем запрос на загрузку
            response = requests.post(f"{api_url}/upload", json=payload, headers=headers)

            if response.status_code != 200:
                st.error(f"Ошибка при загрузке батча {i + 1}/{total_batches}: {response.text}")
                continue
                
            result = response.json()
            if result.get("success") and "ids" in result:
                uploaded_ids.extend(result["ids"])
            else:
                st.warning(f"Предупреждение: API вернул неожиданный ответ для батча {i + 1}/{total_batches}: {result}")

        except Exception as e:
            st.error(f"Ошибка при загрузке батча {i + 1}/{total_batches}: {str(e)}")

        # Обновляем прогресс
        progress_value = min(1.0, (i + 1) / total_batches)
        progress_bar.progress(progress_value)

    st.success(f"Загружено {len(uploaded_ids)} записей")
    return uploaded_ids

def predict(data, token, api_url):
    """Получение предсказаний для тестовых данных"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    predictions = []
    progress_bar = st.progress(0)
    total_rows = len(data)

    for i, (index, row) in enumerate(data.iterrows()):
        try:
            payload = {
                "id": str(uuid.uuid4()),  # Добавляем ID, так как он обязателен
                "subject": row["subject"] if not pd.isna(row["subject"]) else "no_subject",
                "description": row["description"] if not pd.isna(row["description"]) else "no_description",
            }

            logger.info(f"Отправка запроса для записи {index}")
            response = requests.post(f"{api_url}/predict", json=payload, headers=headers)

            response.raise_for_status()

            result = response.json()
            logger.debug(f"Получен ответ: {result}")

            if not isinstance(result, dict) or "predictions" not in result:
                raise ValueError("Неожиданный формат ответа от API")

            predictions_list = result["predictions"]
            if not isinstance(predictions_list, list) or not predictions_list:
                raise ValueError("Список предсказаний пуст или имеет неверный формат")
                
            top_prediction = predictions_list[0]["class_name"]
            predictions.append(top_prediction)
            logger.info(f"Предсказание для записи {index}: {top_prediction}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при отправке запроса для записи {index}: {str(e)}")
            st.error(f"Ошибка при отправке запроса для записи {index}: {str(e)}")
            predictions.append(None)
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Ошибка при обработке ответа API для записи {index}: {str(e)}")
            st.error(f"Ошибка при обработке ответа API для записи {index}: {str(e)}")
            predictions.append(None)
        except Exception as e:
            logger.error(f"Неожиданная ошибка для записи {index}: {str(e)}")
            st.error(f"Неожиданная ошибка для записи {index}: {str(e)}")
            predictions.append(None)

        # Обновляем прогресс (используем i вместо index)
        progress_value = min(1.0, (i + 1) / total_rows)
        progress_bar.progress(progress_value)

    return predictions

# Функции из evaluate.py, перенесенные в api_utils.py
def calculate_metrics(y_true, y_pred):
    """Вычисление метрик качества классификации"""
    # Удаляем записи, где предсказание не было получено
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if not y_true_valid:
        logger.error("Ошибка: Нет валидных предсказаний для оценки")
        return None, None, None, None

    # Вычисляем метрики
    acc = accuracy_score(y_true_valid, y_pred_valid)
    
    # Получаем отчет о классификации в виде словаря
    report_dict = classification_report(y_true_valid, y_pred_valid, output_dict=True)
    
    # Также сохраняем текстовую версию для обратной совместимости
    report_text = classification_report(y_true_valid, y_pred_valid)
    
    cm = confusion_matrix(y_true_valid, y_pred_valid)
    
    # Получаем уникальные классы для отображения в матрице ошибок
    classes = sorted(list(set(y_true_valid + y_pred_valid)))
    
    return acc, report_dict, report_text, cm, classes

def get_classification_report_df(report_dict):
    """Преобразование словаря classification_report в DataFrame"""
    # Преобразуем словарь в DataFrame и транспонируем для лучшего отображения
    df = pd.DataFrame(report_dict).transpose()
    
    # Удаляем ненужные столбцы, если они есть
    columns_to_keep = ['precision', 'recall', 'f1-score', 'support']
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    # Округляем числовые значения для лучшей читаемости
    for col in ['precision', 'recall', 'f1-score']:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    # Преобразуем support в целое число
    if 'support' in df.columns:
        df['support'] = df['support'].astype(int)
    
    return df

def plot_confusion_matrix(cm, class_names=None):
    """Построение матрицы ошибок"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Если имена классов не указаны, используем уникальные значения
    if class_names is None and cm.shape[0] <= 20:  # Ограничиваем для читаемости
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # Создаем тепловую карту
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
    # Извлекаем данные для визуализации
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Создаем DataFrame для визуализации
    metrics_data = {
        'Class': [],
        'Metric': [],
        'Value': []
    }
    
    for cls in classes:
        for metric in ['precision', 'recall', 'f1-score']:
            metrics_data['Class'].append(cls)
            metrics_data['Metric'].append(metric)
            metrics_data['Value'].append(report_dict[cls][metric])
    
    df = pd.DataFrame(metrics_data)
    
    # Создаем визуализацию
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Используем разные цвета для разных метрик
    palette = {
        'precision': 'blue',
        'recall': 'green',
        'f1-score': 'red'
    }
    
    # Создаем сгруппированную столбчатую диаграмму
    sns.barplot(
        x='Class',
        y='Value',
        hue='Metric',
        data=df,
        palette=palette,
        ax=ax
    )
    
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
    
    # Сохраняем визуализации
    fig_cm = plot_confusion_matrix(cm, class_names)
    fig_cm.savefig(os.path.join(metrics_dir, f"confusion_matrix_{timestamp}.png"))
    
    fig_metrics = plot_classification_metrics(report_dict)
    fig_metrics.savefig(os.path.join(metrics_dir, f"classification_metrics_{timestamp}.png"))
    
    return metrics_filename
