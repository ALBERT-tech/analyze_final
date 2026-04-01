"""
main.py — FastAPI сервер анализатора тендерной документации.

Запуск:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Эндпоинты:
    GET  /              → HTML интерфейс
    POST /analyze       → загрузка файлов (браузер), возврат task_id + polling
    GET  /status/{id}   → текущий статус задачи (polling)
    POST /api/analyze   → JSON с URL файлов (Битрикс и др.), callback по готовности
    GET  /prompt        → текущий промпт по умолчанию
"""

import asyncio
import json
import logging
import re
import tempfile
import shutil
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from pipeline.extractor import extract_files
from pipeline.classifier import classify
from pipeline.splitter import split_into_sections
from pipeline.router import build_context

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# -- Конфиг ---------------------------------------------------------------
API_KEY           = os.getenv("API_KEY", "")
API_URL           = os.getenv("API_URL", "https://litellm.tokengate.ru/v1/chat/completions")
MODEL             = os.getenv("MODEL", "deepseek/deepseek-chat")
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "2000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "80000"))

DEFAULT_PROMPT_PATH = Path("prompts/default.txt")

# -- Хранилище задач (in-memory) ------------------------------------------
tasks: dict[str, dict] = {}
TASK_TTL = 1800  # 30 мин — потом удаляем


def cleanup_old_tasks():
    """Удаляет задачи старше TASK_TTL."""
    now = time.time()
    expired = [tid for tid, t in tasks.items() if now - t.get("created", 0) > TASK_TTL]
    for tid in expired:
        # Удаляем временные файлы если остались
        tmp_dir = tasks[tid].get("tmp_dir")
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        del tasks[tid]
    if expired:
        logger.info(f"[CLEANUP] Удалено {len(expired)} старых задач")


# -- FastAPI ---------------------------------------------------------------
app = FastAPI(title="Анализатор тендерной документации")


def load_default_prompt() -> str:
    if DEFAULT_PROMPT_PATH.exists():
        return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    return "Проанализируй документацию и ответь на вопросы.\n\n{{CONTRACT_TEXT}}"


# -- Модель запроса для API (Битрикс и др.) --------------------------------

class ApiAnalyzeRequest(BaseModel):
    files: list[str]                # URL-ссылки на файлы
    callback_url: str | None = None # Куда отправить результат
    prompt: str | None = None       # Кастомный промпт (необязательно)


def format_results_text(results: list[dict]) -> str:
    """Форматирует результаты в читаемый текст для Битрикса."""
    lines = ["АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ", "=" * 40, ""]
    for i, item in enumerate(results, 1):
        title = item.get("title", "—")
        answer = item.get("answer", "—")
        citation = item.get("citation", "")
        lines.append(f"{i}. {title}")
        lines.append(f"   Ответ: {answer}")
        if citation and citation.strip() not in ("—", ""):
            lines.append(f"   Цитата: {citation}")
        lines.append("")
    return "\n".join(lines)


# -- Фоновая обработка задачи ---------------------------------------------

async def process_task(task_id: str, saved_paths: list[Path], prompt_template: str, tmp_dir: Path):
    """Весь pipeline в фоне. Обновляет tasks[task_id] на каждом шаге."""
    try:
        task = tasks[task_id]

        # -- Шаг 1+2: Извлечение текста --
        task.update(status="processing", step="extracting", detail="Извлечение текста из документов...")
        logger.info(f"[TASK {task_id[:8]}] Начало обработки: {len(saved_paths)} файлов")

        extracted = await asyncio.to_thread(extract_files, saved_paths)

        logger.info(f"[TASK {task_id[:8]}] Извлечено: {len(extracted)} файлов, "
                    f"пропущено: {sum(1 for f in extracted if f.skipped)}")

        if not extracted:
            task.update(status="error", detail="Не удалось извлечь текст ни из одного файла")
            return

        skipped = [{"name": f.name, "reason": f.skip_reason} for f in extracted if f.skipped]
        valid = [f for f in extracted if not f.skipped and f.text.strip()]

        if not valid:
            detail = "Нет пригодных для анализа файлов."
            if skipped:
                detail += " Пропущены: " + "; ".join(f["name"] + " -- " + f["reason"] for f in skipped)
            task.update(status="error", detail=detail)
            return

        # -- Шаг 3: Классификация + нарезка --
        task.update(step="classifying", detail="Классификация и нарезка на разделы...")

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
            logger.info(f"[TASK {task_id[:8]}] {f.name}: тип={doc_type}, разделов={len(sections)}")

        # -- Шаг 4: Сборка контекста --
        task.update(step="context", detail="Сборка контекста для модели...")

        context, truncated, context_chars = build_context(docs, max_chars=MAX_CONTEXT_CHARS)

        if not context:
            task.update(status="error", detail="Не удалось сформировать контекст из документов")
            return

        logger.info(f"[TASK {task_id[:8]}] Контекст: {context_chars} символов, обрезан={truncated}")

        # -- Шаг 5: DeepSeek API --
        task.update(step="api", detail="Запрос к DeepSeek...")

        final_prompt = prompt_template.replace("{{CONTRACT_TEXT}}", context)
        logger.info(f"[TASK {task_id[:8]}] Отправка в API: модель={MODEL}, промпт={len(final_prompt)} симв.")

        async with httpx.AsyncClient(timeout=300.0) as client:
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
            logger.error(f"[TASK {task_id[:8]}] API ошибка {response.status_code}: {err_body}")
            # Понятное сообщение при превышении контекста
            if "max_n" in err_body or "context" in err_body.lower() or "token" in err_body.lower():
                task.update(status="error", detail=(
                    "Документация слишком большая для модели. "
                    "Попробуйте загрузить меньше файлов или только ключевые документы "
                    "(контракт + ТЗ), без вспомогательных."
                ))
            else:
                task.update(status="error", detail=f"Ошибка API модели: {err_body[:200]}")
            return

        data = response.json()
        raw_content = data["choices"][0]["message"]["content"].strip()
        logger.info(f"[TASK {task_id[:8]}] API ответ получен ({len(raw_content)} символов)")

        # -- Шаг 6: Парсинг JSON --
        task.update(step="parsing", detail="Формирование отчёта...")

        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_content)
        clean_content = json_match.group(1).strip() if json_match else raw_content

        try:
            parsed = json.loads(clean_content)
            if not isinstance(parsed, list):
                raise ValueError("Ответ не является списком")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[TASK {task_id[:8]}] Ошибка парсинга JSON: {e}\nКонтент: {clean_content[:300]}")
            task.update(status="error", detail="Модель вернула некорректный JSON. Попробуйте ещё раз.")
            return

        logger.info(f"[TASK {task_id[:8]}] JSON распарсен, элементов: {len(parsed)}")

        # -- Готово --
        meta = {
            "files": [{"name": d["name"], "type": d["type"], "chars": d["char_count"]} for d in docs],
            "skipped": skipped,
            "context_chars": context_chars,
            "truncated": truncated,
        }
        task.update(status="done", results=parsed, meta=meta)

        # -- Callback (для Битрикса и др.) --
        callback_url = task.get("callback_url")
        if callback_url:
            try:
                text_report = format_results_text(parsed)
                async with httpx.AsyncClient(timeout=30.0) as cb_client:
                    await cb_client.post(callback_url, json={
                        "task_id": task_id,
                        "status": "done",
                        "results": parsed,
                        "text": text_report,
                        "meta": meta,
                    })
                logger.info(f"[TASK {task_id[:8]}] Callback отправлен на {callback_url}")
            except Exception as cb_err:
                logger.error(f"[TASK {task_id[:8]}] Ошибка callback: {cb_err}")

    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] Необработанная ошибка: {e}", exc_info=True)
        tasks[task_id].update(status="error", detail=f"Внутренняя ошибка сервера: {str(e)[:200]}")

        # Callback при ошибке
        callback_url = tasks[task_id].get("callback_url")
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as cb_client:
                    await cb_client.post(callback_url, json={
                        "task_id": task_id,
                        "status": "error",
                        "detail": str(e)[:200],
                    })
            except Exception:
                pass

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- Эндпоинты ------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("static/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="static/index.html не найден")
    html = html_path.read_text(encoding="utf-8")
    prompt = load_default_prompt().replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = html.replace("__DEFAULT_PROMPT__", prompt)
    return HTMLResponse(content=html)


@app.head("/")
async def index_head():
    return HTMLResponse(content="", status_code=200)


@app.get("/prompt")
async def get_default_prompt():
    return {"prompt": load_default_prompt()}


@app.post("/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    prompt: str = Form(None),
):
    """Принимает файлы, запускает фоновую обработку, возвращает task_id."""
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не переданы")

    # Чистим старые задачи
    cleanup_old_tasks()

    # Промпт
    prompt_template = prompt.strip() if prompt and prompt.strip() else load_default_prompt()
    if "{{CONTRACT_TEXT}}" not in prompt_template:
        raise HTTPException(status_code=400, detail="В промпте отсутствует метка {{CONTRACT_TEXT}}")

    # Сохраняем файлы
    tmp_dir = Path(tempfile.mkdtemp())
    saved_paths: list[Path] = []
    for upload in files:
        dest = tmp_dir / upload.filename
        content = await upload.read()
        dest.write_bytes(content)
        saved_paths.append(dest)
        logger.info(f"Получен файл: {upload.filename} ({len(content)} байт)")

    # Создаём задачу
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "step": "uploading",
        "detail": "Файлы загружены, начинаем обработку...",
        "created": time.time(),
        "tmp_dir": str(tmp_dir),
    }

    # Запускаем в фоне
    asyncio.create_task(process_task(task_id, saved_paths, prompt_template, tmp_dir))

    logger.info(f"[TASK {task_id[:8]}] Задача создана, файлов: {len(saved_paths)}")
    return JSONResponse({"task_id": task_id})


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Возвращает текущий статус задачи."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    if task["status"] == "done":
        return JSONResponse({
            "status": "done",
            "results": task.get("results"),
            "meta": task.get("meta"),
        })
    elif task["status"] == "error":
        return JSONResponse({
            "status": "error",
            "detail": task.get("detail", "Неизвестная ошибка"),
        })
    else:
        return JSONResponse({
            "status": "processing",
            "step": task.get("step", ""),
            "detail": task.get("detail", ""),
        })


@app.post("/api/analyze")
async def api_analyze(req: ApiAnalyzeRequest):
    """
    API для Битрикса и внешних систем.
    Принимает JSON с URL файлов, скачивает, обрабатывает, отправляет результат на callback_url.
    """
    if not req.files:
        raise HTTPException(status_code=400, detail="Список файлов пуст")

    cleanup_old_tasks()

    # Промпт
    prompt_template = req.prompt.strip() if req.prompt and req.prompt.strip() else load_default_prompt()
    if "{{CONTRACT_TEXT}}" not in prompt_template:
        raise HTTPException(status_code=400, detail="В промпте отсутствует метка {{CONTRACT_TEXT}}")

    # Скачиваем файлы по URL
    tmp_dir = Path(tempfile.mkdtemp())
    saved_paths: list[Path] = []

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as dl_client:
            for url in req.files:
                try:
                    resp = await dl_client.get(url)
                    resp.raise_for_status()
                except Exception as e:
                    logger.error(f"[API] Не удалось скачать {url}: {e}")
                    continue

                # Имя файла из URL
                filename = url.split("/")[-1].split("?")[0]
                if not filename:
                    filename = f"file_{len(saved_paths)}"
                dest = tmp_dir / filename
                dest.write_bytes(resp.content)
                saved_paths.append(dest)
                logger.info(f"[API] Скачан: {filename} ({len(resp.content)} байт)")

    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при скачивании файлов: {str(e)[:200]}")

    if not saved_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Не удалось скачать ни одного файла")

    # Создаём задачу
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "step": "uploading",
        "detail": "Файлы скачаны, начинаем обработку...",
        "created": time.time(),
        "tmp_dir": str(tmp_dir),
        "callback_url": req.callback_url,
    }

    # Запускаем в фоне
    asyncio.create_task(process_task(task_id, saved_paths, prompt_template, tmp_dir))

    logger.info(f"[API TASK {task_id[:8]}] Задача создана, файлов: {len(saved_paths)}, "
                f"callback: {req.callback_url or 'нет'}")

    return JSONResponse({
        "task_id": task_id,
        "status": "processing",
        "message": f"Принято {len(saved_paths)} файлов. "
                   + ("Результат будет отправлен на callback_url." if req.callback_url
                      else "Проверяйте статус через GET /status/" + task_id),
    })


# Статика монтируется ПОСЛЕ роутов
app.mount("/static", StaticFiles(directory="static"), name="static")
