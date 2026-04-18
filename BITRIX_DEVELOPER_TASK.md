# Задача для разработчика Битрикс24

## Что нужно сделать

Робот в сделке: менеджер загружает ZIP с тендерной документацией → робот отправляет файл на анализ → получает текстовый отчёт → вставляет в комментарий к сделке.

## Подготовка

### 1. Пользовательское поле в сделке
- Название: "Тендерная документация"
- Тип: файл
- Код поля: например `UF_CRM_TENDER_DOCS`

### 2. Исходящий вебхук (или REST-приложение)
Нужен доступ к REST API для:
- `disk.file.get` — получить прямую ссылку на файл
- `crm.timeline.comment.add` — добавить комментарий к сделке

---

## Логика бизнес-процесса

### Триггер
Изменение поля `UF_CRM_TENDER_DOCS` в сделке (файл загружен).

### Шаг 1: Получить прямую ссылку на файл

Файловое поле сделки содержит ID файла на диске Битрикс.

```
GET https://bitrix.rossilber.com/rest/1/<webhook-token>/disk.file.get?id={FILE_ID}
```

Ответ (JSON):
```json
{
  "result": {
    "ID": 90708,
    "NAME": "documents.zip",
    "DOWNLOAD_URL": "https://bitrix.rossilber.com/rest/1/<token>/disk.file.download?id=90708&token=xxxx",
    ...
  }
}
```

Нам нужно поле `result.DOWNLOAD_URL`.

### Шаг 2: Отправить на анализ

```
POST http://85.239.34.39/api/analyze

Headers:
  Content-Type: application/json
  X-API-Key: <ключ будет предоставлен администратором>

Body:
{
  "files": ["{DOWNLOAD_URL из шага 1}"],
  "mode": "short",
  "callback_url": "https://bitrix.rossilber.com/rest/1/<webhook-token>/bizproc.event.send?event_token={EVENT_TOKEN}",
  "external_id": "{ID сделки}"
}
```

Ответ (мгновенный):
```json
{
  "task_id": "xxx",
  "status": "processing",
  "message": "Принято 1 файлов. Результат на callback."
}
```

### Шаг 3: Дождаться callback

Сервер анализатора сам отправит POST на `callback_url` через 30-60 секунд.

Тело callback:
```json
{
  "task_id": "xxx",
  "external_id": "12345",
  "status": "done",
  "text": "АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ\n\n1. НОМЕР И НАИМЕНОВАНИЕ ЗАКУПКИ\n   ...\n\n2. ЗАКУПКА ДЛЯ СМП\n   ...",
  "results": [...]
}
```

Или ошибка:
```json
{
  "task_id": "xxx",
  "external_id": "12345",
  "status": "error",
  "detail": "текст ошибки"
}
```

### Шаг 4: Записать результат в комментарий сделки

При получении callback с `"status": "done"`:

```
POST https://bitrix.rossilber.com/rest/1/<webhook-token>/crm.timeline.comment.add
Content-Type: application/json

{
  "fields": {
    "ENTITY_ID": {external_id из callback},
    "ENTITY_TYPE": "deal",
    "COMMENT": "{text из callback}"
  }
}
```

При `"status": "error"`:
```json
{
  "fields": {
    "ENTITY_ID": "{external_id}",
    "ENTITY_TYPE": "deal",
    "COMMENT": "Ошибка анализа документации: {detail из callback}"
  }
}
```

---

## Альтернатива без callback (polling)

Если callback сложно реализовать, можно опрашивать статус:

После шага 2 (получили `task_id`) — ждать 60 сек, потом:

```
GET http://85.239.34.39/status/{task_id}
Headers: X-API-Key: <ключ>
```

Ответ тот же формат что и callback. Если `"status": "processing"` — подождать ещё 30 сек и повторить. Если `"done"` — записать комментарий.

---

## Важные моменты

1. **URL файла** — обязательно через `disk.file.get` → `DOWNLOAD_URL`. Обычная ссылка из браузера (show_file.php) не работает — требует авторизацию.

2. **API-ключ** — выдаётся администратором через панель. Передаётся в заголовке `X-API-Key`.

3. **external_id** — передавайте ID сделки. Он вернётся в callback без изменений, чтобы знать к какой сделке привязать ответ.

4. **Формат файлов** — ZIP, PDF, DOCX, DOC, XLSX, XLS, HTML, TXT. Рекомендация: загружать ZIP со всей документацией.

5. **Время обработки** — 30-90 секунд в зависимости от объёма.

6. **Лимиты** — дневной лимит токенов на пользователя (настраивается администратором). При превышении — ответ 429.

---

## Тестирование (curl)

Проверить что файл скачивается:
```bash
curl -sI "DOWNLOAD_URL_ИЗ_DISK_FILE_GET"
# Должен быть Content-Type: application/zip (или другой файловый тип)
# НЕ text/html
```

Отправить на анализ:
```bash
curl -X POST http://85.239.34.39/api/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <ключ>" \
  -d '{"files": ["DOWNLOAD_URL"], "mode": "short", "external_id": "TEST_001"}'
```

Проверить статус:
```bash
curl http://85.239.34.39/status/<task_id> -H "X-API-Key: <ключ>"
```
