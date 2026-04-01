"""
main.py — FastAPI-сервер анализатора тендерной документации.

Архитектура v2: 4 параллельных запроса к DeepSeek по типам документов.

Эндпоинты:
    GET  /              — HTML-интерфейс
    POST /analyze       — загрузка файлов (браузер), возврат task_id + polling
    GET  /status/{id}   — текущий статус задачи (polling)
    POST /api/analyze   — JSON с URL файлов (Битрикс и др.), callback по готовности
    GET  /prompt        — текущий промпт по умолчанию
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
from pipeline.router import (
    build_context, build_context_for_type, extract_nacreg_paragraphs,
    CONTRACT_KEYWORDS, NOTICE_KEYWORDS, TECHSPEC_KEYWORDS,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# -- Конфиг ---------------------------------------------------------------
API_KEY                = os.getenv("API_KEY", "")
API_URL                = os.getenv("API_URL", "https://litellm.tokengate.ru/v1/chat/completions")
MODEL                  = os.getenv("MODEL", "deepseek/deepseek-chat")
MAX_TOKENS             = int(os.getenv("MAX_TOKENS", "2000"))
MAX_TOKENS_PER_CALL    = int(os.getenv("MAX_TOKENS_PER_CALL", "4000"))
MAX_CONTEXT_CHARS      = int(os.getenv("MAX_CONTEXT_CHARS", "80000"))
MAX_CONTEXT_PER_CALL   = int(os.getenv("MAX_CONTEXT_CHARS_PER_CALL", "60000"))

DEFAULT_PROMPT_PATH = Path("prompts/default.txt")

# Промпты для 4 параллельных запросов
TYPED_PROMPTS = {
    "contract_risks":  Path("prompts/contract_risks.txt"),
    "contract_params": Path("prompts/contract_params.txt"),
    "notice":          Path("prompts/notice.txt"),
    "techspec":        Path("prompts/techspec.txt"),
}

# -- Хранилище задач (in-memory) ------------------------------------------
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
app = FastAPI(title="Анализатор тендерной документации v2")


# -- Утилиты ---------------------------------------------------------------

def _try_parse_json(content: str, task_id: str) -> list | None:
    """Пытается распарсить JSON-массив. При ошибке — ремонт и повтор."""
    # Попытка 1: как есть
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Попытка 2: обрезаем до последнего полного объекта
    try:
        last_brace = content.rfind("}")
        if last_brace > 0:
            repaired = content[:last_brace + 1].rstrip().rstrip(",") + "\n]"
            start = repaired.find("[")
            if start >= 0:
                repaired = repaired[start:]
                parsed = json.loads(repaired)
                if isinstance(parsed, list):
                    logger.warning(f"[TASK {task_id[:8]}] JSON отремонтирован (обрезан)")
                    return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Попытка 3: regex
    try:
        objects = re.findall(r'\{[^{}]*\}', content)
        if objects:
            results = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    if "title" in obj and "answer" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
            if results:
                logger.warning(f"[TASK {task_id[:8]}] JSON отремонтирован (regex, {len(results)} объектов)")
                return results
    except Exception:
        pass

    logger.error(f"[TASK {task_id[:8]}] Не удалось распарсить JSON:\n{content[:500]}")
    return None


def load_default_prompt() -> str:
    if DEFAULT_PROMPT_PATH.exists():
        return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    return "Проанализируй документацию и ответь на вопросы.\n\n{{CONTRACT_TEXT}}"


def load_typed_prompt(prompt_type: str) -> str:
    """Загружает промпт по типу (contract_risks, contract_params, notice, techspec)."""
    path = TYPED_PROMPTS.get(prompt_type)
    if path and path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Промпт не найден: {prompt_type} ({path})")


# -- Вызов DeepSeek API ---------------------------------------------------

async def call_deepseek(prompt_text: str, task_id: str, label: str) -> list | None:
    """Один вызов DeepSeek API. Возвращает распарсенный JSON-массив или None."""
    logger.info(f"[TASK {task_id[:8]}] [{label}] Отправка в API ({len(prompt_text)} симв.)")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.1,
                    "max_tokens": MAX_TOKENS_PER_CALL,
                },
            )

        if response.status_code != 200:
            err_body = response.text[:300]
            logger.error(f"[TASK {task_id[:8]}] [{label}] API ошибка {response.status_code}: {err_body}")
            return None

        data = response.json()
        raw_content = data["choices"][0]["message"]["content"].strip()
        logger.info(f"[TASK {task_id[:8]}] [{label}] Ответ получен ({len(raw_content)} симв.)")

        # Чистим markdown-обёртки
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_content)
        clean_content = json_match.group(1).strip() if json_match else raw_content

        parsed = _try_parse_json(clean_content, task_id)
        if parsed:
            logger.info(f"[TASK {task_id[:8]}] [{label}] Распарсено {len(parsed)} объектов")
        return parsed

    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] [{label}] Ошибка: {e}")
        return None


# -- Мерж результатов ------------------------------------------------------

def merge_results(api_results: dict[str, list | None]) -> tuple[list, list]:
    """
    Объединяет результаты 4 запросов в единый список.
    Добавляет поле source к каждому элементу.

    Returns:
        (all_results, warnings)
    """
    all_results = []
    warnings = []

    source_labels = {
        "contract_risks":  "КОНТРАКТ (риски)",
        "contract_params": "КОНТРАКТ (параметры)",
        "notice":          "ИЗВЕЩЕНИЕ",
        "techspec":        "ТЗ",
    }

    for key, label in source_labels.items():
        items = api_results.get(key)
        if items is None:
            warnings.append(f"Не удалось проанализировать: {label}")
            continue
        for item in items:
            item["source"] = label
            all_results.append(item)

    return all_results, warnings


def format_checklist_report(results: list, warnings: list, meta: dict) -> str:
    """Форматирует результаты в текстовый отчёт с секциями по источникам."""
    lines = ["АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ", ""]

    files = meta.get("files", [])
    if files:
        names = ", ".join(f'{f["name"]} ({f["type"]})' for f in files)
        lines.append(f"Файлы: {names}")
        lines.append("")

    if warnings:
        for w in warnings:
            lines.append(f"!!! {w}")
        lines.append("")

    # Группируем по source
    current_source = None
    num = 0
    for item in results:
        source = item.get("source", "")
        if source != current_source:
            current_source = source
            lines.append(f"=== {source} ===")
            lines.append("")

        num += 1
        title = item.get("title", "")
        answer = item.get("answer", "")
        citation = item.get("citation", "")

        lines.append(f"{num}. {title}")
        lines.append(f"   {answer}")
        if citation and citation != "---":
            lines.append(f"   [{citation}]")
        lines.append("")

    return "\n".join(lines).strip()


# -- Основной pipeline (4 параллельных запроса) ----------------------------

async def process_task_parallel(task_id: str, saved_paths: list[Path], tmp_dir: Path):
    """Pipeline v2: 4 параллельных запроса к DeepSeek."""
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
            doc_type = classify(f.text, f.name)
            sections = split_into_sections(f.text)
            docs.append({
                "name": f.name,
                "type": doc_type,
                "sections": sections,
                "char_count": len(f.text),
            })
            logger.info(f"[TASK {task_id[:8]}] {f.name}: тип={doc_type}, разделов={len(sections)}")

        # -- Шаг 4: Сборка контекстов по типам --
        task.update(step="context", detail="Сборка контекстов по типам документов...")

        contract_ctx, _, contract_chars = build_context_for_type(
            docs, ["КОНТРАКТ"], MAX_CONTEXT_PER_CALL, CONTRACT_KEYWORDS
        )
        notice_ctx, _, notice_chars = build_context_for_type(
            docs, ["ИЗВЕЩЕНИЕ", "ТРЕБОВАНИЯ"], MAX_CONTEXT_PER_CALL, NOTICE_KEYWORDS
        )
        techspec_ctx, _, techspec_chars = build_context_for_type(
            docs, ["ТЗ"], MAX_CONTEXT_PER_CALL, TECHSPEC_KEYWORDS
        )

        logger.info(f"[TASK {task_id[:8]}] Контексты: "
                    f"контракт={contract_chars}, извещение={notice_chars}, ТЗ={techspec_chars}")

        if not contract_ctx and not notice_ctx and not techspec_ctx:
            task.update(status="error", detail="Не удалось сформировать контекст из документов")
            return

        # -- Шаг 5: 4 параллельных запроса к DeepSeek --
        task.update(step="api", detail="Запросы к DeepSeek (4 параллельных)...")

        coros = {}

        if contract_ctx:
            try:
                p1 = load_typed_prompt("contract_risks").replace("{{CONTRACT_TEXT}}", contract_ctx)
                coros["contract_risks"] = call_deepseek(p1, task_id, "КОНТРАКТ-РИСКИ")
            except FileNotFoundError:
                logger.warning(f"[TASK {task_id[:8]}] Промпт contract_risks не найден")

            try:
                p2 = load_typed_prompt("contract_params").replace("{{CONTRACT_TEXT}}", contract_ctx)
                coros["contract_params"] = call_deepseek(p2, task_id, "КОНТРАКТ-ПАРАМЕТРЫ")
            except FileNotFoundError:
                logger.warning(f"[TASK {task_id[:8]}] Промпт contract_params не найден")

        if notice_ctx:
            try:
                p3 = load_typed_prompt("notice").replace("{{CONTRACT_TEXT}}", notice_ctx)
                coros["notice"] = call_deepseek(p3, task_id, "ИЗВЕЩЕНИЕ")
            except FileNotFoundError:
                logger.warning(f"[TASK {task_id[:8]}] Промпт notice не найден")

        if techspec_ctx:
            try:
                p4 = load_typed_prompt("techspec").replace("{{CONTRACT_TEXT}}", techspec_ctx)
                coros["techspec"] = call_deepseek(p4, task_id, "ТЗ")
            except FileNotFoundError:
                logger.warning(f"[TASK {task_id[:8]}] Промпт techspec не найден")

        if not coros:
            task.update(status="error", detail="Не удалось подготовить ни одного запроса к модели")
            return

        # Параллельный запуск
        keys = list(coros.keys())
        results_list = await asyncio.gather(*coros.values(), return_exceptions=True)

        api_results = {}
        for key, result in zip(keys, results_list):
            if isinstance(result, Exception):
                logger.error(f"[TASK {task_id[:8]}] [{key}] Ошибка: {result}")
                api_results[key] = None
            else:
                api_results[key] = result

        # -- Шаг 6: Мерж результатов --
        task.update(step="parsing", detail="Формирование отчёта...")

        all_results, warnings = merge_results(api_results)

        if not all_results:
            task.update(status="error", detail="Модель не вернула ни одного ответа. Попробуйте ещё раз.")
            return

        logger.info(f"[TASK {task_id[:8]}] Итого: {len(all_results)} ответов, {len(warnings)} предупреждений")

        # -- Готово --
        meta = {
            "files": [{"name": d["name"], "type": d["type"], "chars": d["char_count"]} for d in docs],
            "skipped": skipped,
            "context_chars": contract_chars + notice_chars + techspec_chars,
            "truncated": any(c > 0 for c in [contract_chars, notice_chars, techspec_chars]),
        }
        text_report = format_checklist_report(all_results, warnings, meta)
        task.update(
            status="done",
            results=all_results,
            warnings=warnings,
            meta=meta,
            text=text_report,
        )

        # -- Callback --
        callback_url = task.get("callback_url")
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as cb_client:
                    await cb_client.post(callback_url, json={
                        "task_id": task_id,
                        "status": "done",
                        "results": all_results,
                        "warnings": warnings,
                        "text": text_report,
                        "meta": meta,
                    })
                logger.info(f"[TASK {task_id[:8]}] Callback отправлен на {callback_url}")
            except Exception as cb_err:
                logger.error(f"[TASK {task_id[:8]}] Ошибка callback: {cb_err}")

    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] Необработанная ошибка: {e}", exc_info=True)
        tasks[task_id].update(status="error", detail=f"Внутренняя ошибка сервера: {str(e)[:200]}")

        callback_url = tasks[task_id].get("callback_url")
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as cb_client:
                    await cb_client.post(callback_url, json={
                        "task_id": task_id, "status": "error", "detail": str(e)[:200],
                    })
            except Exception:
                pass

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- Legacy pipeline (один запрос, для кастомных промптов) -----------------

async def process_task_legacy(task_id: str, saved_paths: list[Path], prompt_template: str, tmp_dir: Path):
    """Старый pipeline: один запрос с кастомным промптом."""
    try:
        task = tasks[task_id]

        task.update(status="processing", step="extracting", detail="Извлечение текста из документов...")
        extracted = await asyncio.to_thread(extract_files, saved_paths)

        if not extracted:
            task.update(status="error", detail="Не удалось извлечь текст ни из одного файла")
            return

        skipped = [{"name": f.name, "reason": f.skip_reason} for f in extracted if f.skipped]
        valid = [f for f in extracted if not f.skipped and f.text.strip()]

        if not valid:
            task.update(status="error", detail="Нет пригодных для анализа файлов.")
            return

        task.update(step="classifying", detail="Классификация и нарезка на разделы...")
        docs = []
        for f in valid:
            doc_type = classify(f.text, f.name)
            sections = split_into_sections(f.text)
            docs.append({"name": f.name, "type": doc_type, "sections": sections, "char_count": len(f.text)})

        task.update(step="context", detail="Сборка контекста для модели...")
        context, truncated, context_chars = build_context(docs, max_chars=MAX_CONTEXT_CHARS)

        nacreg_extra = extract_nacreg_paragraphs(docs)
        if nacreg_extra:
            context = context + nacreg_extra
            context_chars = len(context)

        if not context:
            task.update(status="error", detail="Не удалось сформировать контекст")
            return

        task.update(step="api", detail="Запрос к DeepSeek...")
        final_prompt = prompt_template.replace("{{CONTRACT_TEXT}}", context)

        parsed = await call_deepseek(final_prompt, task_id, "LEGACY")

        if not parsed:
            task.update(status="error", detail="Модель вернула некорректный ответ. Попробуйте ещё раз.")
            return

        task.update(step="parsing", detail="Формирование отчёта...")

        meta = {
            "files": [{"name": d["name"], "type": d["type"], "chars": d["char_count"]} for d in docs],
            "skipped": skipped,
            "context_chars": context_chars,
            "truncated": truncated,
        }

        lines = ["АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ", ""]
        files_str = ", ".join(f["name"] for f in meta["files"])
        if files_str:
            lines.append(f"Файлы: {files_str}")
            lines.append("")
        for i, item in enumerate(parsed, 1):
            lines.append(f'{i}. {item.get("title", "")}')
            lines.append(f'   {item.get("answer", "")}')
            cit = item.get("citation", "")
            if cit and cit != "---":
                lines.append(f"   [{cit}]")
            lines.append("")
        text_report = "\n".join(lines).strip()

        task.update(status="done", results=parsed, meta=meta, text=text_report)

        callback_url = task.get("callback_url")
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as cb_client:
                    await cb_client.post(callback_url, json={
                        "task_id": task_id, "status": "done",
                        "results": parsed, "text": text_report, "meta": meta,
                    })
            except Exception as cb_err:
                logger.error(f"[TASK {task_id[:8]}] Ошибка callback: {cb_err}")

    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] Ошибка: {e}", exc_info=True)
        tasks[task_id].update(status="error", detail=str(e)[:200])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- Модель запроса для API ------------------------------------------------

class ApiAnalyzeRequest(BaseModel):
    files: list[str]
    callback_url: str | None = None
    prompt: str | None = None


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
    """Принимает файлы, запускает обработку, возвращает task_id."""
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не переданы")

    cleanup_old_tasks()

    # Определяем: кастомный промпт или стандартный pipeline
    custom_prompt = None
    if prompt and prompt.strip():
        custom_prompt = prompt.strip()
        if "{{CONTRACT_TEXT}}" not in custom_prompt:
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

    # Выбираем pipeline
    if custom_prompt:
        asyncio.create_task(process_task_legacy(task_id, saved_paths, custom_prompt, tmp_dir))
        logger.info(f"[TASK {task_id[:8]}] Задача создана (LEGACY), файлов: {len(saved_paths)}")
    else:
        asyncio.create_task(process_task_parallel(task_id, saved_paths, tmp_dir))
        logger.info(f"[TASK {task_id[:8]}] Задача создана (4-PARALLEL), файлов: {len(saved_paths)}")

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
async def api_analyze(req: ApiAnalyzeRequest):
    """API для Битрикса: JSON с URL файлов, скачивание, обработка, callback."""
    if not req.files:
        raise HTTPException(status_code=400, detail="Список файлов пуст")

    cleanup_old_tasks()

    custom_prompt = None
    if req.prompt and req.prompt.strip():
        custom_prompt = req.prompt.strip()
        if "{{CONTRACT_TEXT}}" not in custom_prompt:
            raise HTTPException(status_code=400, detail="В промпте отсутствует метка {{CONTRACT_TEXT}}")

    # Скачиваем файлы
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

                filename = url.split("/")[-1].split("?")[0]
                if not filename:
                    filename = f"file_{len(saved_paths)}"
                dest = tmp_dir / filename
                dest.write_bytes(resp.content)
                saved_paths.append(dest)
                logger.info(f"[API] Скачан: {filename} ({len(resp.content)} байт)")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Ошибка скачивания: {str(e)[:200]}")

    if not saved_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Не удалось скачать ни одного файла")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "step": "uploading",
        "detail": "Файлы скачаны, начинаем обработку...",
        "created": time.time(),
        "tmp_dir": str(tmp_dir),
        "callback_url": req.callback_url,
    }

    if custom_prompt:
        asyncio.create_task(process_task_legacy(task_id, saved_paths, custom_prompt, tmp_dir))
    else:
        asyncio.create_task(process_task_parallel(task_id, saved_paths, tmp_dir))

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
