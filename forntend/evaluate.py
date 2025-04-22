import argparse
import json
import logging
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Настройки API
# API_URL = "http://localhost:8190"
# USERNAME = "admin"
# PASSWORD = "secret"


def get_token():
    """Получение токена аутентификации"""
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={
                "username": USERNAME,
                "password": PASSWORD,
                "scope": "predict upload search",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise Exception(f"Ошибка аутентификации: {response.text}")

        return response.json()["access_token"]
    except requests.exceptions.ConnectionError:
        raise Exception(f"Не удалось подключиться к API по адресу {API_URL}")


def upload_data(data, token):
    """Загрузка данных в систему"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Разбиваем данные на батчи по 100 записей
    batch_size = 100
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)

    uploaded_ids = []

    for i in tqdm(range(total_batches), desc="Загрузка данных"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = data.iloc[start_idx:end_idx]

        # Подготавливаем данные для загрузки
        items = []
        for _, row in batch_data.iterrows():
            # Используем оригинальный ID из данных, если он есть
            item_id = row.get("id", str(row.name))

            item = {
                "id": item_id,
                "subject": row["subject"]
                if not pd.isna(row["subject"])
                else "no_subject",
                "description": row["description"]
                if not pd.isna(row["description"])
                else "no_description",
                "class_name": row["class"],
            }

            # Добавляем task, если есть в данных
            if "task" in row and not pd.isna(row["task"]):
                item["task"] = row["task"]

            items.append(item)

        payload = {"items": items}

        try:
            # Отправляем запрос на загрузку
            response = requests.post(f"{API_URL}/upload", json=payload, headers=headers)

            if response.status_code != 200:
                print(
                    f"Ошибка при загрузке батча {i + 1}/{total_batches}: {response.text}"
                )
                continue

            result = response.json()
            if result.get("success") and "ids" in result:
                uploaded_ids.extend(result["ids"])
            else:
                print(
                    f"Предупреждение: API вернул неожиданный ответ для батча {i + 1}/{total_batches}: {result}"
                )

        except Exception as e:
            print(f"Ошибка при загрузке батча {i + 1}/{total_batches}: {str(e)}")

        # Небольшая пауза, чтобы не перегружать API
        time.sleep(0.1)

    print(f"Загружено {len(uploaded_ids)} записей")
    return uploaded_ids


def predict(data, token):
    """Получение предсказаний для тестовых данных"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    predictions = []

    for index, row in tqdm(
        data.iterrows(), total=len(data), desc="Получение предсказаний"
    ):
        try:
            payload = {
                "id": str(uuid.uuid4()),  # Добавляем ID, так как он обязателен
                "subject": row["subject"]
                if not pd.isna(row["subject"])
                else "no_subject",
                "description": row["description"]
                if not pd.isna(row["description"])
                else "no_description",
            }

            logger.info(f"Отправка запроса для записи {index}")
            response = requests.post(
                f"{API_URL}/predict", json=payload, headers=headers
            )

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
            predictions.append(None)
        except (KeyError, IndexError, ValueError) as e:
            logger.error(
                f"Ошибка при обработке ответа API для записи {index}: {str(e)}"
            )
            predictions.append(None)
        except Exception as e:
            logger.error(f"Неожиданная ошибка для записи {index}: {str(e)}")
            predictions.append(None)

        time.sleep(0.1)

    return predictions


def evaluate(y_true, y_pred):
    """Оценка качества классификации"""
    # Удаляем записи, где предсказание не было получено
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if not y_true_valid:
        print("Ошибка: Нет валидных предсказаний для оценки")
        return None, None

    # Вычисляем метрики
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    report = classification_report(y_true_valid, y_pred_valid, output_dict=True)

    # Выводим результаты
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true_valid, y_pred_valid))

    # Строим матрицу ошибок
    cm = confusion_matrix(y_true_valid, y_pred_valid)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(set(y_true_valid)),
        yticklabels=sorted(set(y_true_valid)),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to 'confusion_matrix.png'")

    # Сохраняем детальный отчет в JSON
    with open("evaluation_report.json", "w") as f:
        json.dump({"accuracy": accuracy, "classification_report": report}, f, indent=4)
    print("Detailed report saved to 'evaluation_report.json'")

    return accuracy, report


def search_test(query, token, limit=5):
    """Тестирование функции поиска"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {"query": query, "limit": limit}

    try:
        response = requests.post(f"{API_URL}/search", json=payload, headers=headers)

        if response.status_code != 200:
            print(f"Ошибка при поиске: {response.text}")
            return None

        return response.json()
    except Exception as e:
        print(f"Ошибка при выполнении поиска: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка данных и оценка качества классификации"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Путь к CSV файлу с данными"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Доля тестовой выборки"
    )
    parser.add_argument(
        "--search_query", type=str, default=None, help="Запрос для тестирования поиска"
    )
    parser.add_argument(
        "--api_url", type=str, default="http://localhost:8190", help="URL API сервера"
    )
    args = parser.parse_args()

    # Устанавливаем URL API
    global API_URL
    API_URL = args.api_url

    # Загружаем данные
    print(f"Загрузка данных из {args.data}...")
    data = pd.read_csv(args.data)

    # Проверяем наличие необходимых колонок
    required_columns = ["subject", "description", "class"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"В данных отсутствует колонка '{col}'")

    # Разделяем на обучающую и тестовую выборки
    train_data, test_data = train_test_split(
        data, test_size=args.test_size, random_state=42, stratify=data["class"]
    )

    print(f"Размер обучающей выборки: {len(train_data)}")
    print(f"Размер тестовой выборки: {len(test_data)}")

    # Получаем токен
    print("Получение токена...")
    token = get_token()

    # Загружаем обучающие данные
    print("Загрузка обучающих данных...")
    uploaded_ids = upload_data(train_data, token)

    # Даем системе время на индексацию
    print("Ожидание индексации данных (10 секунд)...")
    time.sleep(10)

    # Получаем предсказания для тестовой выборки
    print("Получение предсказаний для тестовой выборки...")
    predictions = predict(test_data, token)

    # Оцениваем качество
    print("Оценка качества классификации...")
    evaluate(test_data["class"].tolist(), predictions)

    # Тестируем поиск, если указан запрос
    if args.search_query:
        print(f"\nТестирование поиска с запросом: '{args.search_query}'")
        search_results = search_test(args.search_query, token)
        if search_results and "results" in search_results:
            print(f"Найдено {len(search_results['results'])} результатов:")
            for i, result in enumerate(search_results["results"]):
                print(
                    f"{i + 1}. {result['subject']} (Класс: {result['class_name']}, Оценка: {result['score']:.4f})"
                )


if __name__ == "__main__":
    main()
