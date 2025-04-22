import uuid
import requests
import streamlit as st
import logging

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
        progress_bar.progress((i + 1) / total_batches)

    st.success(f"Загружено {len(uploaded_ids)} записей")
    return uploaded_ids

def predict(data, token, api_url):
    """Получение предсказаний для тестовых данных"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    predictions = []
    progress_bar = st.progress(0)

    for index, row in data.iterrows():
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

        # Обновляем прогресс
        progress_bar.progress((index + 1) / len(data))

    return predictions
