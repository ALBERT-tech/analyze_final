# Анализатор тендерной документации

FastAPI-сервис с Docling для анализа тендерных документов.

## Стек

- **FastAPI** — веб-сервер
- **Docling** — извлечение текста из PDF, DOCX, HTML
- **DeepSeek** — LLM-анализ (через LiteLLM-прокси)
- **uvicorn** — ASGI-сервер

## Структура проекта

```
tender-analyzer/
├── .gitignore
├── .env.example        ← скопируйте в .env и заполните
├── requirements.txt
├── main.py             ← FastAPI + оркестрация пайплайна
├── pipeline/
│   ├── extractor.py    ← Docling + распаковка ZIP
│   ├── classifier.py   ← тип документа
│   ├── splitter.py     ← нарезка на разделы
│   └── router.py       ← маршрутизация + сборка контекста
├── prompts/
│   └── default.txt     ← промпт по умолчанию (редактируется)
└── static/
    └── index.html      ← фронтенд
```

## Быстрый старт

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/ваш-репо/tender-analyzer
cd tender-analyzer

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Установите зависимости
pip install -r requirements.txt
# Docling при первом запуске скачает модели (~1-2 GB)

# 4. Настройте секреты
cp .env.example .env
# Отредактируйте .env — вставьте API_KEY

# 5. Запустите
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Откройте http://localhost:8000

## Деплой на Railway

1. Создайте проект на [railway.app](https://railway.app)
2. Подключите GitHub-репозиторий
3. В разделе **Variables** добавьте переменные из `.env.example`
4. Railway задеплоит автоматически при `git push`

## Переменные окружения

| Переменная          | Описание                         | По умолчанию                              |
|---------------------|----------------------------------|-------------------------------------------|
| `API_KEY`           | Ключ API DeepSeek/LiteLLM        | —                                         |
| `API_URL`           | URL прокси-сервера               | https://litellm.tokengate.ru/v1/...       |
| `MODEL`             | Идентификатор модели             | deepseek/deepseek-chat                    |
| `MAX_TOKENS`        | Макс. токенов в ответе           | 2000                                      |
| `MAX_CONTEXT_CHARS` | Макс. символов контекста         | 100000 (~25K токенов)                     |

## Пайплайн обработки

```
ZIP / PDF / DOCX / HTML
   ↓
[1] Распаковка ZIP (включая вложенные, до 3 уровней)
   ↓
[2] Docling: извлечение текста → Markdown с заголовками
   ↓
[3] Классификация: КОНТРАКТ | ТЗ | ИЗВЕЩЕНИЕ | ТРЕБОВАНИЯ | МУСОР
   ↓
[4] Нарезка по Markdown-заголовкам → карта {раздел → текст}
   ↓
[5] Ранжирование разделов по ключевым словам → сборка контекста
   ↓
[6] DeepSeek API → JSON с ответами и цитатами
   ↓
[7] Отчёт (браузер: TXT / PDF через печать)
```

## Известные ограничения

- `.doc` (старый OLE2) не поддерживается — конвертируйте в `.docx`
- Лимит контекста: ~25K токенов (ограничение прокси)
- Сканированные PDF без текстового слоя обрабатываются хуже
