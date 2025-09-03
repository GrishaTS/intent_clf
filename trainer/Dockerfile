FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Moscow

RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

WORKDIR /app/app
CMD ["python", "-u", "scheduler.py"]
