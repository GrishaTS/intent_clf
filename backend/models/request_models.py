# backend/models/request_models.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ====== Базовые запросы/ответы ======

class RequestItem(BaseModel):
    id: str = Field(..., description="Идентификатор заявки")
    subject: str = Field(..., description="Тема обращения")
    description: str = Field(..., description="Описание обращения")
    class_name: Optional[str] = Field(None, description="Класс обращения (если известен)")
    task: Optional[str] = Field(None, description="Задача обращения (если известна)")

    class Config:
        schema_extra = {
            "example": {
                "id": "INC0027099",
                "subject": "Старая версия 1С клиента. Садовники д.4 к.2",
                "description": "Удалить старые версии 1С клиента на ПК коменданта.",
                "class_name": "Сопровождение сервисов сотрудника",
                "task": "1С клиент",
            }
        }


class BatchRequestItem(BaseModel):
    items: List[RequestItem] = Field(..., description="Список обращений для обработки")

    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "INC1001",
                        "subject": "Проблема с доступом к системе",
                        "description": "Не могу войти в личный кабинет после смены пароля",
                        "class_name": "Технические проблемы",
                    },
                    {
                        "id": "INC1002",
                        "subject": "Ошибка при оплате",
                        "description": "При попытке оплаты возникает ошибка сервера",
                        "class_name": "Финансовые вопросы",
                    },
                ]
            }
        }


class PredictionResult(BaseModel):
    class_name: str = Field(..., description="Предсказанный класс")
    probability: float = Field(..., description="Вероятность класса")


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult] = Field(..., description="Список предсказаний с вероятностями")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"class_name": "Технические проблемы", "probability": 0.85},
                    {"class_name": "Финансовые вопросы", "probability": 0.10},
                    {"class_name": "Другое", "probability": 0.05},
                ]
            }
        }


class UploadResponse(BaseModel):
    success: bool = Field(..., description="Статус операции")
    message: str = Field(..., description="Сообщение о результате")
    ids: Optional[List[str]] = Field(None, description="Идентификаторы загруженных документов")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Данные успешно загружены",
                "ids": ["1", "2", "3"],
            }
        }


# ====== Поиск ======

class SearchRequest(BaseModel):
    subject: str = Field("", description="Тема для поиска")
    description: str = Field("", description="Описание для поиска")
    limit: int = Field(10, description="Количество результатов")


class SearchResult(BaseModel):
    id: str = Field(..., description="Идентификатор объекта в хранилище (Qdrant)")
    request_id: str = Field(..., description="Идентификатор исходной заявки")
    subject: str = Field(..., description="Тема обращения")
    description: str = Field(..., description="Описание обращения")
    class_name: str = Field(..., description="Класс обращения")
    score: float = Field(..., description="Оценка релевантности")

    class Config:
        schema_extra = {
            "example": {
                "id": "qdrant_123",
                "request_id": "INC1001",
                "subject": "Проблема с доступом к системе",
                "description": "Не могу войти в личный кабинет после смены пароля",
                "class_name": "Технические проблемы",
                "score": 0.92,
            }
        }


class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="Результаты поиска")


# ====== Compute Metrics ======

class MetricsItem(BaseModel):
    id: str = Field(..., description="Идентификатор записи")
    description: str = Field(..., description="Текст для инференса (обязателен)")
    subject: Optional[str] = Field("no_subject", description="Опциональная тема")
    label: str = Field(..., description="Истинный класс (target)")

    class Config:
        schema_extra = {
            "example": {
                "id": "INC2001",
                "subject": "Не открывается почта",
                "description": "Outlook выдает ошибку при запуске",
                "label": "Технические проблемы",
            }
        }


class MetricsComputeRequest(BaseModel):
    items: List[MetricsItem] = Field(..., description="Тестовый датасет для оценки качества")

    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "INC2001",
                        "subject": "Не открывается почта",
                        "description": "Outlook выдает ошибку при запуске",
                        "label": "Технические проблемы",
                    },
                    {
                        "id": "INC2002",
                        "subject": "Возврат средств",
                        "description": "Оплата прошла дважды",
                        "label": "Финансовые вопросы",
                    },
                ]
            }
        }


class MetricsResponse(BaseModel):
    filename: str = Field(..., description="Имя файла с метриками (user_id.json)")
    timestamp: str = Field(..., description="Время вычисления (YYYYMMDD_HHMMSS)")
    accuracy: float = Field(..., description="Accuracy на тестовом наборе")
    n_total: int = Field(..., description="Всего объектов")
    n_valid: int = Field(..., description="Число валидных предсказаний")
    n_invalid: int = Field(..., description="Число невалидных предсказаний")
    classes: List[str] = Field(..., description="Список классов")
    confusion_matrix: List[List[int]] = Field(..., description="Матрица ошибок (числа)")
    classification_report_dict: Dict[str, Any] = Field(..., description="Отчет sklearn classification_report")
    classification_report_text: Optional[str] = Field(None, description="Текстовый отчет (если нужен)")

    class Config:
        schema_extra = {
            "example": {
                "filename": "u_ac97ef2c5677.json",
                "timestamp": "20250902_231500",
                "accuracy": 0.78,
                "n_total": 500,
                "n_valid": 495,
                "n_invalid": 5,
                "classes": ["Технические проблемы", "Финансовые вопросы", "Другое"],
                "confusion_matrix": [[180, 12, 8], [15, 160, 5], [10, 6, 99]],
                "classification_report_dict": {
                    "Технические проблемы": {"precision": 0.88, "recall": 0.90, "f1-score": 0.89, "support": 200},
                    "Финансовые вопросы": {"precision": 0.86, "recall": 0.85, "f1-score": 0.86, "support": 180},
                    "Другое": {"precision": 0.84, "recall": 0.80, "f1-score": 0.82, "support": 120},
                    "accuracy": 0.78,
                    "macro avg": {"precision": 0.86, "recall": 0.85, "f1-score": 0.86, "support": 500},
                    "weighted avg": {"precision": 0.86, "recall": 0.86, "f1-score": 0.86, "support": 500},
                },
                "classification_report_text": "…",
            }
        }
