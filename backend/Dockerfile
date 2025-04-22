FROM python:3.9-slim

WORKDIR /app

# Копирование файлов
COPY . .

RUN pip install uv
RUN uv sync


# Добавление директории в PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Запуск приложения
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

