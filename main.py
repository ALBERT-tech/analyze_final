"""
main.py — FastAPI сервер анализатора тендерной документации.

Запуск:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Эндпоинты:
    GET  /           → HTML интерфейс
    POST /analyze    → загрузка файлов, анализ, возврат JSON
    GET  /prompt     → текущий промпт по умолчанию
"""

import json
import logging
import re
import tempfile
import shutil
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

from pipeline.extractor import extract_files
from pipeline.classifier import classify
from pipeline.splitter import split_into_sections
from pipeline.router import build_context

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Конфиг ──────────────────────────────────────────────────
API_KEY           = os.getenv("API_KEY", "")
API_URL           = os.getenv("API_URL", "https://litellm.tokengate.ru/v1/chat/completions")
MODEL             = os.getenv("MODEL", "deepseek/deepseek-chat")
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "2000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "100000"))

DEFAULT_PROMPT_PATH = Path("prompts/default.txt")

# ── FastAPI ──────────────────────────────────────────────────
app = FastAPI(title="Анализатор тендерной документации")


def load_default_prompt() -> str:
    if DEFAULT_PROMPT_PATH.exists():
        return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    return "Проанализируй документацию и ответь на вопросы.\n\n{{CONTRACT_TEXT}}"


# ── Эндпоинты ────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — возвращает JSON, не HTML."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("static/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="static/index.html не найден")
    html = html_path.read_text(encoding="utf-8")
    # Встраиваем промпт прямо в HTML — без fetch
    prompt = load_default_prompt().replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = html.replace("__DEFAULT_PROMPT__", prompt)
    return HTMLResponse(content=html)


@app.head("/")
async def index_head():
    """Render health check — HEAD на / отвечаем 200."""
    return HTMLResponse(content="", status_code=200)


@app.get("/prompt")
async def get_default_prompt():
    return {"prompt": load_default_prompt()}


@app.post("/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    prompt: str = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не переданы")

    # Финальный промпт
    prompt_template = prompt.strip() if prompt and prompt.strip() else load_default_prompt()
    if "{{CONTRACT_TEXT}}" not in prompt_template:
        raise HTTPException(status_code=400, detail="В промпте отсутствует метка {{CONTRACT_TEXT}}")

    # Временная директория для загрузок
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Сохраняем загруженные файлы
        saved_paths: list[Path] = []
        for upload in files:
            dest = tmp_dir / upload.filename
            content = await upload.read()
            dest.write_bytes(content)
            saved_paths.append(dest)
            logger.info(f"Получен файл: {upload.filename} ({len(content)} байт)")

        # ── Шаг 1+2: Распаковка + извлечение текста ─────────
        logger.info(f"[PIPELINE] Начало обработки: {len(saved_paths)} файлов")
        extracted = extract_files(saved_paths)

        logger.info(f"[PIPELINE] Извлечено: {len(extracted)} файлов, "
                    f"пропущено: {sum(1 for f in extracted if f.skipped)}")

        if not extracted:
            raise HTTPException(status_code=422, detail="Не удалось извлечь текст ни из одного файла")

        # Собираем информацию о пропущенных файлах
        skipped = [{"name": f.name, "reason": f.skip_reason} for f in extracted if f.skipped]
        valid = [f for f in extracted if not f.skipped and f.text.strip()]

        if not valid:
            detail = "Нет пригодных для анализа файлов."
            if skipped:
                detail += " Пропущены: " + "; ".join(f["name"] + " — " + f["reason"] for f in skipped)
            raise HTTPException(status_code=422, detail=detail)

        # ── Шаг 3: Классификация + нарезка ──────────────────
        docs = []
        for f in valid:
            doc_type = classify(f.text)
            sections = split_into_sections(f.text)
            docs.append({
                "name": f.name,
                "type": doc_type,
                "sections": sections,
                "char_count": len(f.text),
            })
            logger.info(f"{f.name}: тип={doc_type}, разделов={len(sections)}")

        # ── Шаг 4: Сборка контекста ──────────────────────────
        context, truncated, context_chars = build_context(docs, max_chars=MAX_CONTEXT_CHARS)

        if not context:
            raise HTTPException(status_code=422, detail="Не удалось сформировать контекст из документов")

        logger.info(f"[PIPELINE] Контекст: {context_chars} символов, обрезан={truncated}")

        # ── Шаг 5: DeepSeek API ──────────────────────────────
        final_prompt = prompt_template.replace("{{CONTRACT_TEXT}}", context)
        logger.info(f"[PIPELINE] Отправка в API: модель={MODEL}, промпт={len(final_prompt)} симв.")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": final_prompt}],
                    "temperature": 0.1,
                    "max_tokens": MAX_TOKENS,
                },
            )

        if response.status_code != 200:
            err_body = response.text[:500]
            logger.error(f"API ошибка {response.status_code}: {err_body}")
            raise HTTPException(status_code=502, detail=f"Ошибка API модели: {err_body}")

        data = response.json()
        raw_content = data["choices"][0]["message"]["content"].strip()
        logger.info(f"[PIPELINE] API ответ получен ({len(raw_content)} символов)")

        # Чистим Markdown-обёртки
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_content)
        clean_content = json_match.group(1).strip() if json_match else raw_content

        try:
            parsed = json.loads(clean_content)
            if not isinstance(parsed, list):
                raise ValueError("Ответ не является списком")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[PIPELINE] Ошибка парсинга JSON: {e}\nКонтент: {clean_content[:300]}")
            raise HTTPException(status_code=502, detail="Модель вернула некорректный JSON. Попробуйте ещё раз.")

        logger.info(f"[PIPELINE] JSON распарсен, элементов: {len(parsed)}")

        # ── Формируем ответ ──────────────────────────────────
        return JSONResponse({
            "results": parsed,
            "meta": {
                "files": [
                    {"name": d["name"], "type": d["type"], "chars": d["char_count"]}
                    for d in docs
                ],
                "skipped": skipped,
                "context_chars": context_chars,
                "truncated": truncated,
            },
        })

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Статика монтируется ПОСЛЕ роутов, чтобы /prompt и другие эндпоинты не перехватывались
app.mount("/static", StaticFiles(directory="static"), name="static")
