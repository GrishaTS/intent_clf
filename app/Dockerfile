FROM python:3.9-slim

WORKDIR /app

# Копирование файлов
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Добавление директории в PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

