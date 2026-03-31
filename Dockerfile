FROM python:3.11-slim

# Системные зависимости:
# antiword — читает старый .doc (OLE2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    antiword \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — чтобы слой кешировался при изменении кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY . .

# Проверяем что pipeline пакет на месте
RUN python -c "from pipeline.classifier import classify; print('pipeline OK')"

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
