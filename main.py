"""
main.py — FastAPI-сервер анализатора тендерной документации.

Архитектура v3: 2 параллельных запроса (Риски + Параметры), весь текст, без классификации.

Эндпоинты:
    GET  /              — HTML-интерфейс
    POST /analyze       — загрузка файлов, возврат task_id + polling
    GET  /status/{id}   — статус задачи
    POST /api/analyze   — JSON с URL файлов (Битрикс), callback
    GET  /prompt        — дефолтный промпт
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
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Response, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from pipeline.extractor import extract_files
from pipeline.classifier import classify
from pipeline.splitter import split_into_sections
import auth as auth_module

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# -- Конфиг ---------------------------------------------------------------
API_KEY             = os.getenv("API_KEY", "")
API_URL             = os.getenv("API_URL", "https://api.agentplatform.ru/v1/chat/completions")
MODEL               = os.getenv("MODEL", "google/gemini-2.0-flash-001")
# Резервный провайдер (fallback при 429)
FALLBACK_API_KEY    = os.getenv("FALLBACK_API_KEY", "")
FALLBACK_API_URL    = os.getenv("FALLBACK_API_URL", "https://api.timeweb.ai/v1/chat/completions")
FALLBACK_MODEL      = os.getenv("FALLBACK_MODEL", "gemini/gemini-2.0-flash")
MAX_TOKENS_PER_CALL = int(os.getenv("MAX_TOKENS_PER_CALL", "12000"))
MAX_CONTEXT_CHARS   = int(os.getenv("MAX_CONTEXT_CHARS", "500000"))

DEFAULT_PROMPT_PATH = Path("prompts/default.txt")
SESSION_SECRET      = os.getenv("SESSION_SECRET", "tender-analyzer-secret-key-change-me")

# -- Сессии (cookie → user_id) ---------------------------------------------
sessions: dict[str, int] = {}  # session_token → user_id
RISKS_PROMPT_PATH   = Path("prompts/risks.txt")
PARAMS_PROMPT_PATH  = Path("prompts/params.txt")
SHORT_PROMPT_PATH   = Path("prompts/short.txt")
REFINE_PROMPT_PATH  = Path("prompts/refine.txt")

# -- Хранилище задач -------------------------------------------------------
tasks: dict[str, dict] = {}
TASK_TTL = 1800


def cleanup_old_tasks():
    now = time.time()
    expired = [tid for tid, t in tasks.items() if now - t.get("created", 0) > TASK_TTL]
    for tid in expired:
        tmp_dir = tasks[tid].get("tmp_dir")
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        del tasks[tid]
    if expired:
        logger.info(f"[CLEANUP] Удалено {len(expired)} старых задач")


# -- FastAPI ---------------------------------------------------------------
app = FastAPI(title="Анализатор тендерной документации v3")

# Инициализация БД
auth_module.init_db()

# Пути без авторизации
PUBLIC_PATHS = {"/auth/login", "/health", "/openapi.json", "/docs", "/favicon.ico"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Публичные пути
        if path in PUBLIC_PATHS or path.startswith("/static/"):
            return await call_next(request)

        # Страница логина
        if path == "/login":
            return await call_next(request)

        # API-доступ через X-API-Key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user = auth_module.get_user_by_api_key(api_key)
            if not user:
                return JSONResponse({"detail": "Неверный API-ключ"}, status_code=401)
            request.state.user = user
            return await call_next(request)

        # Веб-доступ через cookie
        session_token = request.cookies.get("session")
        if session_token and session_token in sessions:
            user_id = sessions[session_token]
            user = auth_module.get_user_by_id(user_id)
            if user and user["active"]:
                request.state.user = user
                return await call_next(request)

        # Не авторизован → редирект на логин (для веб) или 401 (для API)
        if path.startswith("/api/") or path.startswith("/analyze") or path.startswith("/status/"):
            return JSONResponse({"detail": "Требуется авторизация"}, status_code=401)

        return RedirectResponse("/login", status_code=302)


app.add_middleware(AuthMiddleware)


# -- Утилиты ---------------------------------------------------------------

def _try_parse_json(content: str, task_id: str) -> list | None:
    """Парсинг JSON-массива с авторемонтом."""
    # Попытка 1
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Попытка 2: обрезка до последнего }
    try:
        last_brace = content.rfind("}")
        if last_brace > 0:
            repaired = content[:last_brace + 1].rstrip().rstrip(",") + "\n]"
            start = repaired.find("[")
            if start >= 0:
                parsed = json.loads(repaired[start:])
                if isinstance(parsed, list):
                    logger.warning(f"[TASK {task_id[:8]}] JSON отремонтирован (обрезан)")
                    return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Попытка 3: regex
    try:
        objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
        if objects:
            results = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    if "title" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
            if results:
                logger.warning(f"[TASK {task_id[:8]}] JSON отремонтирован (regex, {len(results)} объектов)")
                return results
    except Exception:
        pass

    logger.error(f"[TASK {task_id[:8]}] JSON не распарсен:\n{content[:500]}")
    return None


def load_default_prompt() -> str:
    if DEFAULT_PROMPT_PATH.exists():
        return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    return "{{CONTRACT_TEXT}}"


def load_prompt(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Промпт не найден: {path}")


# -- Вызов API -------------------------------------------------------------

async def _try_api(url: str, key: str, model: str, prompt_text: str,
                   task_id: str, label: str, provider: str) -> tuple[list | None, int]:
    """Один вызов API. Возвращает (parsed_or_None, http_status)."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.1,
                    "max_tokens": MAX_TOKENS_PER_CALL,
                },
            )
        if response.status_code != 200:
            logger.error(f"[TASK {task_id[:8]}] [{label}] [{provider}] API ошибка {response.status_code}: {response.text[:300]}")
            return None, response.status_code

        raw = response.json()["choices"][0]["message"]["content"].strip()
        logger.info(f"[TASK {task_id[:8]}] [{label}] [{provider}] Ответ ({len(raw)} симв.)")
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        clean = m.group(1).strip() if m else raw
        parsed = _try_parse_json(clean, task_id)
        if parsed:
            logger.info(f"[TASK {task_id[:8]}] [{label}] [{provider}] Распарсено {len(parsed)} объектов")
        return parsed, 200
    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] [{label}] [{provider}] Ошибка: {e}")
        return None, 0


async def call_api(prompt_text: str, task_id: str, label: str) -> list | None:
    """Вызов LLM API с fallback при 429. Отмечает в tasks[task_id] использование резерва."""
    logger.info(f"[TASK {task_id[:8]}] [{label}] Отправка ({len(prompt_text)} симв.)")

    # Попытка 1: основной провайдер
    parsed, status = await _try_api(API_URL, API_KEY, MODEL, prompt_text, task_id, label, "MAIN")

    # Fallback при 429 или сетевой ошибке
    if parsed is None and status in (0, 429, 502, 503, 504) and FALLBACK_API_KEY:
        logger.warning(f"[TASK {task_id[:8]}] [{label}] Основной провайдер недоступен ({status}), переключаюсь на резервный")
        parsed, status = await _try_api(FALLBACK_API_URL, FALLBACK_API_KEY, FALLBACK_MODEL,
                                        prompt_text, task_id, label, "FALLBACK")
        if parsed and task_id in tasks:
            tasks[task_id]["fallback_used"] = True

    return parsed


# -- Сборка контекста ------------------------------------------------------

def build_full_context(docs: list[dict], max_chars: int) -> tuple[str, int]:
    """Собирает весь текст всех документов в один контекст."""
    parts = []
    total = 0

    for doc in docs:
        header = f"\n\n{'='*60}\nДОКУМЕНТ: {doc['name']} (тип: {doc['type']})\n{'='*60}\n"
        text = "\n".join(s.text for s in doc["sections"])

        block = header + text
        if total + len(block) > max_chars:
            remaining = max_chars - total - 100
            if remaining > 500:
                block = block[:remaining] + "\n...[обрезано]"
                parts.append(block)
                total += len(block)
            break

        parts.append(block)
        total += len(block)

    return "".join(parts).strip(), total


# -- Форматирование отчёта -------------------------------------------------

def format_report(risks: list, params: list, warnings: list, meta: dict) -> str:
    """Текстовый отчёт: Блок 1 (Риски) + Блок 2 (Параметры)."""
    lines = ["АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ", ""]

    # Обработанные файлы
    files = meta.get("files", [])
    if files:
        lines.append("Обработанные файлы:")
        for f in files:
            lines.append(f'  + {f["name"]} ({f["type"]}, {f["chars"]} симв.)')
        lines.append("")

    # Пропущенные файлы
    skipped = meta.get("skipped", [])
    if skipped:
        lines.append("Не обработаны:")
        for s in skipped:
            lines.append(f'  - {s["name"]} — {s["reason"]}')
        lines.append("")

    if warnings:
        for w in warnings:
            lines.append(f"!!! {w}")
        lines.append("")

    # Блок 1
    if risks:
        lines.append("=" * 40)
        lines.append("БЛОК 1 — РИСКИ (стоп-сигналы)")
        lines.append("=" * 40)
        lines.append("")
        for i, item in enumerate(risks, 1):
            lines.append(f"{i}. {item.get('title', '')}")
            lines.append(f"   {item.get('answer', '')}")
            for src in item.get("sources", []):
                doc = src.get("doc", "")
                ref = src.get("ref", "")
                cit = src.get("citation", "")
                if cit:
                    lines.append(f"   [{doc}, {ref} — «{cit}»]")
            # Фолбэк для старого формата citation
            if not item.get("sources") and item.get("citation", "").strip() not in ("", "—", "-"):
                lines.append(f"   [{item['citation']}]")
            lines.append("")

    # Блок 2
    if params:
        lines.append("=" * 40)
        lines.append("БЛОК 2 — ИЗВЛЕКАЕМЫЕ ПАРАМЕТРЫ")
        lines.append("=" * 40)
        lines.append("")
        for i, item in enumerate(params, 1):
            lines.append(f"{i}. {item.get('title', '')}")
            lines.append(f"   {item.get('answer', '')}")
            for src in item.get("sources", []):
                doc = src.get("doc", "")
                ref = src.get("ref", "")
                cit = src.get("citation", "")
                if cit:
                    lines.append(f"   [{doc}, {ref} — «{cit}»]")
            if not item.get("sources") and item.get("citation", "").strip() not in ("", "—", "-"):
                lines.append(f"   [{item['citation']}]")
            lines.append("")

    return "\n".join(lines).strip()


# -- Основной pipeline (2 параллельных запроса) ----------------------------

async def _send_error_callback(task_id: str, detail: str):
    """Отправляет callback с ошибкой если callback_url задан."""
    task = tasks.get(task_id, {})
    callback_url = task.get("callback_url")
    if callback_url:
        try:
            async with httpx.AsyncClient(timeout=30.0) as cb:
                await cb.post(callback_url, json={
                    "task_id": task_id,
                    "external_id": task.get("external_id"),
                    "status": "error",
                    "detail": detail,
                })
            logger.info(f"[TASK {task_id[:8]}] Error callback отправлен")
        except Exception as e:
            logger.error(f"[TASK {task_id[:8]}] Ошибка error callback: {e}")


async def process_task_v3(task_id: str, saved_paths: list[Path], tmp_dir: Path):
    """Pipeline v3: весь текст → 2 параллельных запроса (Риски + Параметры)."""
    try:
        task = tasks[task_id]

        # -- Шаг 1: Извлечение текста --
        task.update(status="processing", step="extracting", detail="Извлечение текста из документов...")
        logger.info(f"[TASK {task_id[:8]}] Начало обработки: {len(saved_paths)} файлов")

        extracted = await asyncio.to_thread(extract_files, saved_paths)

        if not extracted:
            detail = "Не удалось извлечь текст ни из одного файла"
            task.update(status="error", detail=detail)
            await _send_error_callback(task_id, detail)
            return

        skipped = [{"name": f.name, "reason": f.skip_reason} for f in extracted if f.skipped]
        valid = [f for f in extracted if not f.skipped and f.text.strip()]

        if not valid:
            detail = "Нет пригодных для анализа файлов."
            task.update(status="error", detail=detail)
            await _send_error_callback(task_id, detail)
            return

        logger.info(f"[TASK {task_id[:8]}] Извлечено: {len(valid)} файлов, пропущено: {len(skipped)}")

        # -- Шаг 2: Нарезка на разделы (для логов и мета) --
        task.update(step="classifying", detail="Обработка документов...")

        docs = []
        for f in valid:
            doc_type = classify(f.text, f.name)
            sections = split_into_sections(f.text)
            docs.append({
                "name": f.name,
                "type": doc_type,
                "sections": sections,
                "char_count": len(f.text),
            })
            logger.info(f"[TASK {task_id[:8]}] {f.name}: тип={doc_type}, {len(f.text)} симв.")

        # -- Шаг 3: Один контекст из всех документов --
        task.update(step="context", detail="Сборка контекста...")

        context, context_chars = build_full_context(docs, MAX_CONTEXT_CHARS)

        if not context:
            detail = "Не удалось сформировать контекст"
            task.update(status="error", detail=detail)
            await _send_error_callback(task_id, detail)
            return

        logger.info(f"[TASK {task_id[:8]}] Контекст: {context_chars} симв. (~{context_chars//4} токенов)")

        # -- Шаг 4: 2 параллельных запроса --
        task.update(step="api", detail="Запросы к модели (2 параллельных)...")

        try:
            risks_prompt = load_prompt(RISKS_PROMPT_PATH).replace("{{CONTRACT_TEXT}}", context)
            params_prompt = load_prompt(PARAMS_PROMPT_PATH).replace("{{CONTRACT_TEXT}}", context)
        except FileNotFoundError as e:
            task.update(status="error", detail=str(e))
            await _send_error_callback(task_id, str(e))
            return

        risks_result, params_result = await asyncio.gather(
            call_api(risks_prompt, task_id, "РИСКИ"),
            call_api(params_prompt, task_id, "ПАРАМЕТРЫ"),
            return_exceptions=True,
        )

        # Обработка ошибок
        warnings = []
        if isinstance(risks_result, Exception) or risks_result is None:
            warnings.append("Не удалось проанализировать риски")
            risks_result = []
        if isinstance(params_result, Exception) or params_result is None:
            warnings.append("Не удалось извлечь параметры")
            params_result = []

        if not risks_result and not params_result:
            detail = "Модель не вернула ни одного ответа. Попробуйте ещё раз."
            task.update(status="error", detail=detail)
            await _send_error_callback(task_id, detail)
            return

        # -- Шаг 5: Формирование отчёта --
        task.update(step="parsing", detail="Формирование отчёта...")

        total = len(risks_result) + len(params_result)
        logger.info(f"[TASK {task_id[:8]}] Итого: рисков={len(risks_result)}, параметров={len(params_result)}")

        meta = {
            "files": [{"name": d["name"], "type": d["type"], "chars": d["char_count"]} for d in docs],
            "skipped": skipped,
            "context_chars": context_chars,
        }
        text_report = format_report(risks_result, params_result, warnings, meta)
        flat_results = risks_result + params_result

        task.update(
            status="done",
            results=flat_results,
            risks=risks_result,
            parameters=params_result,
            warnings=warnings,
            meta=meta,
            text=text_report,
            context=context,
        )

        # Логирование usage
        uid = task.get("user_id", 0)
        if uid:
            tokens_out = sum(len(json.dumps(r, ensure_ascii=False)) // 4 for r in flat_results)
            auth_module.log_usage(uid, "full", task.get("files_count", 0),
                                 context_chars // 4, tokens_out, "done")

        # -- Callback --
        callback_url = task.get("callback_url")
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as cb:
                    await cb.post(callback_url, json={
                        "task_id": task_id,
                        "external_id": task.get("external_id"),
                        "status": "done",
                        "risks": risks_result,
                        "parameters": params_result,
                        "results": flat_results,
                        "warnings": warnings,
                        "text": text_report,
                        "meta": meta,
                    })
                logger.info(f"[TASK {task_id[:8]}] Callback отправлен")
            except Exception as cb_err:
                logger.error(f"[TASK {task_id[:8]}] Ошибка callback: {cb_err}")

    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] Ошибка: {e}", exc_info=True)
        tasks[task_id].update(status="error", detail=str(e)[:200])

        callback_url = tasks[task_id].get("callback_url")
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as cb:
                    await cb.post(callback_url, json={
                        "task_id": task_id,
                        "external_id": tasks[task_id].get("external_id"),
                        "status": "error", "detail": str(e)[:200],
                    })
            except Exception:
                pass

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- Legacy pipeline (кастомный промпт) ------------------------------------

async def process_task_short(task_id: str, saved_paths: list[Path], tmp_dir: Path):
    """Pipeline short: один запрос, 7 вопросов, обычный отчёт."""
    try:
        task = tasks[task_id]

        task.update(status="processing", step="extracting", detail="Извлечение текста...")
        extracted = await asyncio.to_thread(extract_files, saved_paths)
        valid = [f for f in extracted if not f.skipped and f.text.strip()]

        if not valid:
            task.update(status="error", detail="Нет пригодных файлов.")
            return

        task.update(step="classifying", detail="Обработка документов...")
        docs = []
        for f in valid:
            doc_type = classify(f.text, f.name)
            sections = split_into_sections(f.text)
            docs.append({"name": f.name, "type": doc_type,
                         "sections": sections, "char_count": len(f.text)})
            logger.info(f"[TASK {task_id[:8]}] {f.name}: {doc_type}, {len(f.text)} симв.")

        task.update(step="context", detail="Сборка контекста...")
        context, context_chars = build_full_context(docs, MAX_CONTEXT_CHARS)
        if not context:
            task.update(status="error", detail="Пустой контекст")
            return

        logger.info(f"[TASK {task_id[:8]}] Контекст: {context_chars} симв.")

        task.update(step="api", detail="Запрос к модели...")
        short_prompt = load_prompt(SHORT_PROMPT_PATH).replace("{{CONTRACT_TEXT}}", context)
        parsed = await call_api(short_prompt, task_id, "SHORT")

        if not parsed:
            task.update(status="error", detail="Модель не ответила. Попробуйте ещё раз.")
            return

        task.update(step="parsing", detail="Формирование отчёта...")

        meta = {"files": [{"name": d["name"], "type": d["type"], "chars": d["char_count"]} for d in docs],
                "skipped": [{"name": f.name, "reason": f.skip_reason} for f in extracted if f.skipped],
                "context_chars": context_chars}

        text_report = format_report([], parsed, [], meta)
        task.update(status="done", results=parsed, parameters=parsed, risks=[],
                    warnings=[], meta=meta, text=text_report, context=context)

        # Логирование usage
        uid = task.get("user_id", 0)
        if uid:
            tokens_out = sum(len(json.dumps(r, ensure_ascii=False)) // 4 for r in parsed)
            auth_module.log_usage(uid, "short", task.get("files_count", 0),
                                 context_chars // 4, tokens_out, "done")

        callback_url = task.get("callback_url")
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as cb:
                    await cb.post(callback_url, json={
                        "task_id": task_id,
                        "external_id": task.get("external_id"),
                        "status": "done",
                        "results": parsed, "text": text_report, "meta": meta,
                    })
                logger.info(f"[TASK {task_id[:8]}] Callback отправлен")
            except Exception as cb_err:
                logger.error(f"[TASK {task_id[:8]}] Ошибка callback: {cb_err}")

    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] Ошибка: {e}", exc_info=True)
        tasks[task_id].update(status="error", detail=str(e)[:200])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def process_task_legacy(task_id: str, saved_paths: list[Path], prompt_template: str, tmp_dir: Path):
    """Старый pipeline: один запрос с кастомным промптом."""
    try:
        task = tasks[task_id]
        task.update(status="processing", step="extracting", detail="Извлечение текста...")

        extracted = await asyncio.to_thread(extract_files, saved_paths)
        valid = [f for f in extracted if not f.skipped and f.text.strip()]

        if not valid:
            task.update(status="error", detail="Нет пригодных файлов.")
            return

        task.update(step="context", detail="Сборка контекста...")
        docs = []
        for f in valid:
            docs.append({"name": f.name, "type": classify(f.text, f.name),
                         "sections": split_into_sections(f.text), "char_count": len(f.text)})

        context, context_chars = build_full_context(docs, MAX_CONTEXT_CHARS)
        if not context:
            task.update(status="error", detail="Пустой контекст")
            return

        task.update(step="api", detail="Запрос к модели...")
        final_prompt = prompt_template.replace("{{CONTRACT_TEXT}}", context)
        parsed = await call_api(final_prompt, task_id, "LEGACY")

        if not parsed:
            task.update(status="error", detail="Модель не ответила. Попробуйте ещё раз.")
            return

        meta = {"files": [{"name": d["name"], "type": d["type"], "chars": d["char_count"]} for d in docs],
                "context_chars": context_chars}

        lines = ["АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ", ""]
        for i, item in enumerate(parsed, 1):
            lines.append(f'{i}. {item.get("title", "")}')
            lines.append(f'   {item.get("answer", "")}')
            for src in item.get("sources", []):
                lines.append(f'   [{src.get("doc","")}, {src.get("ref","")} — «{src.get("citation","")}»]')
            if not item.get("sources") and item.get("citation", "").strip() not in ("", "—"):
                lines.append(f'   [{item["citation"]}]')
            lines.append("")

        task.update(status="done", results=parsed, meta=meta, text="\n".join(lines).strip())

    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] Ошибка: {e}", exc_info=True)
        tasks[task_id].update(status="error", detail=str(e)[:200])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- Pydantic модель для API -----------------------------------------------

class ApiAnalyzeRequest(BaseModel):
    files: list[str]
    callback_url: str | None = None
    prompt: str | None = None
    mode: str = "full"  # "short" или "full"
    external_id: str | None = None  # ID сделки Битрикс или другой внешний ID


# -- Эндпоинты ------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_path = Path("static/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html не найден")
    html = html_path.read_text(encoding="utf-8")
    prompt = load_default_prompt().replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = html.replace("__DEFAULT_PROMPT__", prompt)

    # Роль пользователя — для показа/скрытия кнопки "Админка"
    user = getattr(request.state, "user", None)
    role = user["role"] if user else "user"
    html = html.replace("__USER_ROLE__", role)

    return HTMLResponse(content=html)


@app.head("/")
async def index_head():
    return HTMLResponse(content="", status_code=200)


@app.get("/prompt")
async def get_default_prompt():
    return {"prompt": load_default_prompt()}


@app.post("/analyze")
async def analyze(
    request: Request,
    files: list[UploadFile] = File(...),
    prompt: str = Form(None),
    mode: str = Form("full"),
):
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не переданы")

    cleanup_old_tasks()

    # Проверка лимита
    user = getattr(request.state, "user", None)
    user_id = user["id"] if user else 0
    if user_id:
        allowed, used, limit = auth_module.check_daily_limit(user_id)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"Дневной лимит исчерпан ({used}/{limit} токенов)")

    # Кастомный промпт?
    custom_prompt = None
    default = load_default_prompt()
    if prompt and prompt.strip():
        p = prompt.strip()
        if "{{CONTRACT_TEXT}}" not in p:
            raise HTTPException(status_code=400, detail="В промпте нет метки {{CONTRACT_TEXT}}")
        if p.replace("\r\n", "\n").strip() != default.replace("\r\n", "\n").strip():
            custom_prompt = p

    # Сохраняем файлы
    tmp_dir = Path(tempfile.mkdtemp())
    saved_paths = []
    for upload in files:
        dest = tmp_dir / upload.filename
        content = await upload.read()
        dest.write_bytes(content)
        saved_paths.append(dest)
        logger.info(f"Получен: {upload.filename} ({len(content)} байт)")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing", "step": "uploading",
        "detail": "Файлы загружены...", "created": time.time(), "tmp_dir": str(tmp_dir),
        "user_id": user_id, "mode": mode, "files_count": len(saved_paths),
    }

    if custom_prompt:
        asyncio.create_task(process_task_legacy(task_id, saved_paths, custom_prompt, tmp_dir))
        logger.info(f"[TASK {task_id[:8]}] LEGACY, файлов: {len(saved_paths)}")
    elif mode == "short":
        asyncio.create_task(process_task_short(task_id, saved_paths, tmp_dir))
        logger.info(f"[TASK {task_id[:8]}] SHORT, файлов: {len(saved_paths)}")
    else:
        asyncio.create_task(process_task_v3(task_id, saved_paths, tmp_dir))
        logger.info(f"[TASK {task_id[:8]}] FULL (2-PARALLEL), файлов: {len(saved_paths)}")

    return JSONResponse({"task_id": task_id})


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    if task["status"] == "done":
        return JSONResponse({
            "status": "done",
            "risks": task.get("risks", []),
            "parameters": task.get("parameters", []),
            "results": task.get("results"),
            "warnings": task.get("warnings", []),
            "text": task.get("text"),
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
async def api_analyze(req: ApiAnalyzeRequest, request: Request):
    if not req.files:
        raise HTTPException(status_code=400, detail="Список файлов пуст")

    cleanup_old_tasks()

    user = getattr(request.state, "user", None)
    user_id = user["id"] if user else 0
    if user_id:
        allowed, used, limit = auth_module.check_daily_limit(user_id)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"Дневной лимит исчерпан ({used}/{limit} токенов)")

    custom_prompt = None
    default = load_default_prompt()
    if req.prompt and req.prompt.strip():
        p = req.prompt.strip()
        if "{{CONTRACT_TEXT}}" not in p:
            raise HTTPException(status_code=400, detail="В промпте нет метки {{CONTRACT_TEXT}}")
        if p.replace("\r\n", "\n").strip() != default.replace("\r\n", "\n").strip():
            custom_prompt = p

    tmp_dir = Path(tempfile.mkdtemp())
    saved_paths = []

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as dl:
            for url in req.files:
                try:
                    resp = await dl.get(url)
                    resp.raise_for_status()
                except Exception as e:
                    logger.error(f"[API] Скачивание {url}: {e}")
                    continue
                filename = url.split("/")[-1].split("?")[0] or f"file_{len(saved_paths)}"
                dest = tmp_dir / filename
                dest.write_bytes(resp.content)
                saved_paths.append(dest)
                logger.info(f"[API] Скачан: {filename} ({len(resp.content)} байт)")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])

    if not saved_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Не удалось скачать файлы")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing", "step": "uploading",
        "detail": "Файлы скачаны...", "created": time.time(),
        "tmp_dir": str(tmp_dir), "callback_url": req.callback_url,
        "user_id": user_id, "mode": req.mode, "files_count": len(saved_paths),
        "external_id": req.external_id,
    }

    if custom_prompt:
        asyncio.create_task(process_task_legacy(task_id, saved_paths, custom_prompt, tmp_dir))
    elif req.mode == "short":
        asyncio.create_task(process_task_short(task_id, saved_paths, tmp_dir))
    else:
        asyncio.create_task(process_task_v3(task_id, saved_paths, tmp_dir))

    logger.info(f"[API TASK {task_id[:8]}] mode={req.mode}, файлов: {len(saved_paths)}, callback: {req.callback_url or 'нет'}")

    return JSONResponse({
        "task_id": task_id, "status": "processing",
        "message": f"Принято {len(saved_paths)} файлов. " +
                   ("Результат на callback." if req.callback_url else f"GET /status/{task_id}"),
    })


# -- Страницы авторизации --------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(Path("static/login.html").read_text(encoding="utf-8"))


@app.get("/profile", response_class=HTMLResponse)
async def profile_page():
    return HTMLResponse(Path("static/profile.html").read_text(encoding="utf-8"))


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        return RedirectResponse("/", status_code=302)
    return HTMLResponse(Path("static/admin.html").read_text(encoding="utf-8"))


# -- Auth API --------------------------------------------------------------

class LoginRequest(BaseModel):
    login: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


# Rate limiting: IP → [timestamps]
login_attempts: dict[str, list[float]] = {}
LOGIN_RATE_LIMIT = 5  # попыток
LOGIN_RATE_WINDOW = 60  # секунд


@app.post("/auth/login")
async def auth_login(req: LoginRequest, request: Request):
    ip = request.client.host if request.client else "unknown"

    # Rate limiting
    now = time.time()
    attempts = login_attempts.get(ip, [])
    attempts = [t for t in attempts if now - t < LOGIN_RATE_WINDOW]
    if len(attempts) >= LOGIN_RATE_LIMIT:
        logger.warning(f"[AUTH] Rate limit: {ip} ({req.login})")
        return JSONResponse({"ok": False, "detail": "Слишком много попыток. Подождите минуту."}, status_code=429)

    user = auth_module.authenticate(req.login, req.password)
    if not user:
        attempts.append(now)
        login_attempts[ip] = attempts
        logger.warning(f"[AUTH] Неудачный вход: login={req.login}, ip={ip}")
        return JSONResponse({"ok": False, "detail": "Неверный логин или пароль"}, status_code=401)

    # Успешный вход — сбрасываем счётчик
    login_attempts.pop(ip, None)
    logger.info(f"[AUTH] Вход: login={req.login}, ip={ip}")

    session_token = str(uuid.uuid4())
    sessions[session_token] = user["id"]

    response = JSONResponse({"ok": True})
    response.set_cookie("session", session_token, httponly=True, max_age=86400 * 7)
    return response


@app.get("/auth/logout")
async def auth_logout(request: Request):
    session_token = request.cookies.get("session")
    if session_token and session_token in sessions:
        del sessions[session_token]
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session")
    return response


@app.post("/auth/change_password")
async def auth_change_password(req: ChangePasswordRequest, request: Request):
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"ok": False, "detail": "Не авторизован"}, status_code=401)

    # Проверяем старый пароль
    check = auth_module.authenticate(user["login"], req.old_password)
    if not check:
        return JSONResponse({"ok": False, "detail": "Неверный текущий пароль"})

    auth_module.change_password(user["id"], req.new_password)
    return JSONResponse({"ok": True})


# -- Admin API -------------------------------------------------------------

class AddUserRequest(BaseModel):
    login: str
    password: str
    role: str = "user"
    daily_token_limit: int = 0


class ResetPasswordRequest(BaseModel):
    password: str


@app.get("/admin/api/users")
async def admin_get_users(request: Request):
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    return JSONResponse(auth_module.get_user_stats())


@app.get("/admin/api/user_log/{user_id}")
async def admin_get_user_log(user_id: int, request: Request):
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    return JSONResponse(auth_module.get_user_log(user_id))


@app.post("/admin/api/add_user")
async def admin_add_user(req: AddUserRequest, request: Request):
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    try:
        new_user = auth_module.create_user(login=req.login, password=req.password,
                                           role=req.role, daily_limit=req.daily_token_limit)
        return JSONResponse({"ok": True, "api_key": new_user["api_key"]})
    except Exception as e:
        return JSONResponse({"ok": False, "detail": str(e)}, status_code=400)


@app.post("/admin/api/toggle_user/{user_id}")
async def admin_toggle_user(user_id: int, request: Request):
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    target = auth_module.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404)
    auth_module.update_user(user_id, active=0 if target["active"] else 1)
    return JSONResponse({"ok": True})


@app.post("/admin/api/reset_password/{user_id}")
async def admin_reset_password(user_id: int, req: ResetPasswordRequest, request: Request):
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    auth_module.change_password(user_id, req.password)
    return JSONResponse({"ok": True})


# -- Refine: уточнение ответа по одному вопросу ---------------------------

class RefineRequest(BaseModel):
    task_id: str
    question_title: str  # "МЕСТО ПОСТАВКИ"
    question_text: str | None = None  # полный текст вопроса (опционально)


@app.post("/refine")
async def refine(req: RefineRequest, request: Request):
    """Переспросить модель по одному вопросу, используя сохранённый контекст."""
    user = getattr(request.state, "user", None)
    user_id = user["id"] if user else 0

    if user_id:
        allowed, used, limit = auth_module.check_daily_limit(user_id)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"Дневной лимит исчерпан ({used}/{limit} токенов)")

    task = tasks.get(req.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена или устарела (TTL 30 мин)")

    context = task.get("context")
    if not context:
        raise HTTPException(status_code=400, detail="Контекст задачи недоступен")

    question = req.question_text or req.question_title

    try:
        prompt = load_prompt(REFINE_PROMPT_PATH) \
            .replace("{{QUESTION}}", question) \
            .replace("{{CONTRACT_TEXT}}", context)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"[REFINE] {req.task_id[:8]}: {req.question_title}")
    parsed = await call_api(prompt, req.task_id, f"REFINE/{req.question_title[:20]}")

    if not parsed or not isinstance(parsed, list) or len(parsed) == 0:
        raise HTTPException(status_code=502, detail="Модель не вернула ответ. Попробуйте ещё раз.")

    # Логирование usage
    if user_id:
        tokens_out = sum(len(json.dumps(r, ensure_ascii=False)) // 4 for r in parsed)
        auth_module.log_usage(user_id, "refine", 0, len(context) // 4, tokens_out, "done")

    return JSONResponse({"item": parsed[0]})


# Статика ПОСЛЕ роутов
app.mount("/static", StaticFiles(directory="static"), name="static")
