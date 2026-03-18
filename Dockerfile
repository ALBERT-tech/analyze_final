FROM python:3.11-slim

# Системные зависимости:
# antiword — читает старый .doc (OLE2)
# libgl1   — нужен Docling для обработки PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    antiword \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — чтобы слой кешировался при изменении кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY . .

# Docling скачивает модели при первом запуске.
# Делаем это на этапе сборки образа, чтобы не ждать при старте контейнера.
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()" || true

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
