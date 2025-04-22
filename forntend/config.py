import os

# Конфигурационные параметры
API_URL = os.getenv("API_URL", "http://localhost:8190") 
USERNAME = os.getenv("USERNAME", "admin") 
PASSWORD = os.getenv("PASSWORD", "secret")

# Примеры запросов для выбора
DEFAULT_EXAMPLES = [
    {
        "id": "INC0027099",
        "description": "Удалить старые версии 1С клиента на пк коменданта.",
        "subject": "Старая версия 1С клиента. Садовники д.4 к.2",
        "class": "Сопровождение сервисов сотрудника",
        "task": "1С клиент",
    }
    # Можно добавить больше примеров в будущем
]
