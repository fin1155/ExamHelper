# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY static ./static

EXPOSE 8000

ENV OLLAMA_BASE_URL="http://ollama:11434"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
