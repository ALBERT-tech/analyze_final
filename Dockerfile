FROM python:3.11-slim

# Системные зависимости:
# antiword — читает старый .doc (OLE2)
# unar     — распаковка .rar архивов (свободная альтернатива unrar,
#            лежит в основном репозитории, не требует non-free)
RUN apt-get update && apt-get install -y --no-install-recommends \
    antiword \
    unar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — чтобы слой кешировался при изменении кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY . .

# Проверяем что pipeline пакет на месте
RUN python -c "from pipeline.classifier import classify; print('pipeline OK')"

# Непривилегированный пользователь (security-аудит H6): приложение принимает
# произвольные файлы/архивы — нельзя крутить парсеры от root.
# uid 10001 фиксирован, чтобы совпадать с владельцем тома /opt/analyze_final/data на хосте.
RUN useradd -m -u 10001 appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# --proxy-headers + --forwarded-allow-ips "*" — доверяем X-Forwarded-For от nginx
# (порт контейнера слушает только 127.0.0.1, снаружи недоступен), чтобы
# rate-limit логина работал по реальному IP клиента, а не по адресу nginx.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips "*"
