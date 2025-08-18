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


def filter_high_quality_classes(df, min_samples=10, min_f1_score=0.5, api_url=None, username=None, password=None):
    """
    Функция для фильтрации классов с высоким качеством классификации.
    
    Параметры:
    df (DataFrame): DataFrame с данными для обработки
    min_samples (int): Минимальное количество образцов для сохранения класса
    min_f1_score (float): Минимальное значение F1-score для сохранения класса
    api_url, username, password: Параметры для подключения к API
    
    Возвращает:
    DataFrame с отфильтрованными данными и словарь с метриками по классам
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
    
    # Заменяем редкие классы на "Другое"
    df_processed["class"] = df_processed["class"].apply(
        lambda x: "Другое" if x in rare_classes else x
    )
    
    # Отображаем статистику после замены
    with st.expander("Статистика после фильтрации редких классов", expanded=True):
        st.write(' ')
        new_counts = df_processed["class"].value_counts()
        
        # Создаем DataFrame для отображения
        new_stats_df = pd.DataFrame({
            'Значение': new_counts.index,
            'Количество': new_counts.values,
            'Процент': (new_counts.values / new_counts.sum() * 100).round(2)
        })
        
        st.dataframe(new_stats_df)
        
        # Визуализация нового распределения
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
    
    # Вычисляем метрики
    y_true = test_df["class"].tolist()
    y_pred = preds
    
    # Удаляем None из предсказаний
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    
    # Получаем отчет о классификации в виде словаря
    report_dict = classification_report(y_true_valid, y_pred_valid, output_dict=True)
    
    # Фильтруем классы с высоким F1-score
    high_quality_classes = []
    
    for class_name, metrics in report_dict.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            if metrics['f1-score'] >= min_f1_score:
                high_quality_classes.append(class_name)
    
    # Отображаем метрики по классам
    with st.expander("Метрики по классам", expanded=True):
        # Создаем DataFrame для отображения метрик
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
        
        # Сортируем по F1-score
        metrics_df = metrics_df.sort_values('F1-score', ascending=False)
        
        # Округляем числовые значения
        for col in ['Precision', 'Recall', 'F1-score']:
            metrics_df[col] = metrics_df[col].round(3)
        
        st.dataframe(metrics_df)
        
        # Визуализация F1-score по классам
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Создаем цветовую палитру в зависимости от качества
        colors = metrics_df['Качество'].map({'Высокое': 'green', 'Низкое': 'red'})
        
        # Создаем столбчатую диаграмму
        bars = ax.bar(metrics_df['Класс'], metrics_df['F1-score'], color=colors)
        
        # Добавляем горизонтальную линию для порога
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
                    lambda x: x if x in high_quality_classes else "Другое"
                )
    df_high_quality = df_processed

    
    # Отображаем статистику после фильтрации по качеству
    with st.expander("Статистика после фильтрации по качеству", expanded=True):
        high_quality_counts = df_high_quality["class"].value_counts()
        
        # Создаем DataFrame для отображения
        high_quality_stats_df = pd.DataFrame({
            'Значение': high_quality_counts.index,
            'Количество': high_quality_counts.values,
            'Процент': (high_quality_counts.values / high_quality_counts.sum() * 100).round(2)
        })
        
        st.dataframe(high_quality_stats_df)
        
        # Визуализация распределения после фильтрации по качеству
        fig, ax = plt.subplots(figsize=(10, 6))
        high_quality_counts.plot(kind='bar', ax=ax)
        plt.title('Распределение значений после фильтрации по качеству')
        plt.xlabel('Значение')
        plt.ylabel('Количество')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Отображаем информацию о результатах фильтрации
    st.info(f"Исходное количество классов: {len(class_counts)}")
    st.info(f"Количество классов после фильтрации редких: {len(new_counts)}")
    st.info(f"Количество классов с высоким качеством (F1-score >= {min_f1_score}): {len(high_quality_classes)}")
    st.info(f"Количество записей в отфильтрованном наборе: {len(df_high_quality)} из {len(df_processed)} ({len(df_high_quality)/len(df_processed)*100:.1f}%)")
    
    # Возвращаем отфильтрованный DataFrame и метрики
    return df_high_quality, report_dict
