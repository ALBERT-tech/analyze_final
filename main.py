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
import socket
import ipaddress
import os.path
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

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
import cache as cache_module

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

# Retry основного провайдера. agentplatform балансирует запросы между upstream-
# серверами, часть которых периодически нездорова (напр. протухший SSL на
# aiproxy.agentplatform.ru → мгновенный 500/0). Повтор обычно попадает на
# здоровый сервер. Ошибка приходит за ~35мс, так что повторы дешёвые.
MAIN_MAX_ATTEMPTS = int(os.getenv("MAIN_MAX_ATTEMPTS", "3"))
RETRY_DELAY       = float(os.getenv("RETRY_DELAY", "1.0"))
# Статусы, при которых имеет смысл повторить/уйти в резерв (временные сбои).
# 404 — провайдер снял модель из эндпоинтов; 500 — SSL/connect к upstream.
# 401/400 сюда НЕ входят (это наша ошибка — ключ/запрос — повтор не поможет).
RETRY_STATUSES = (0, 404, 429, 500, 502, 503, 504)

# Лимиты на скачивание/распаковку (anti-DoS, см. security-аудит)
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(60 * 1024 * 1024)))   # 60 МБ на файл

# Базовый публичный URL для ссылок на отчёт (/r/{token})
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://robotender.ru").rstrip("/")
PURCHASE_NUMBER_RE = re.compile(r"^[A-Za-zА-Яа-яЁё0-9._\-/]{1,40}$")


def is_safe_outbound_url(url: str) -> tuple[bool, str]:
    """Защита от SSRF: разрешены только http/https и хосты, НЕ резолвящиеся в
    приватные/loopback/link-local/reserved адреса. Возвращает (ok, причина_отказа)."""
    try:
        parts = urlsplit((url or "").strip())
    except Exception:
        return False, "не удалось разобрать URL"
    if parts.scheme not in ("http", "https"):
        return False, f"схема '{parts.scheme}' запрещена (только http/https)"
    host = parts.hostname
    if not host:
        return False, "пустой хост"
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception as e:
        return False, f"не удалось зарезолвить хост ({e})"
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"некорректный IP {ip_str}"
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False, f"хост резолвится во внутренний адрес {ip_str}"
    return True, ""


def safe_filename(name: str, fallback: str = "file") -> str:
    """Возвращает безопасное имя файла без компонентов пути (защита от path traversal)."""
    base = os.path.basename((name or "").replace("\\", "/").strip())
    if not base or base in (".", "..") or base.startswith("/"):
        return fallback
    return base


def sanitize_purchase_number(raw: str | None) -> str | None:
    """Валидирует номер закупки от клиента (защита от хранимого XSS в админке).
    Разрешены буквы/цифры/.-_/ длиной до 40. Иначе — None (номер не используется)."""
    v = (raw or "").strip()
    if not v:
        return None
    return v if PURCHASE_NUMBER_RE.match(v) else None

DEFAULT_PROMPT_PATH = Path("prompts/default.txt")
SESSION_SECRET      = os.getenv("SESSION_SECRET", "tender-analyzer-secret-key-change-me")

# -- Сессии (cookie → user_id) ---------------------------------------------
sessions: dict[str, int] = {}  # session_token → user_id
RISKS_PROMPT_PATH         = Path("prompts/risks.txt")
PARAMS_PROMPT_PATH        = Path("prompts/params.txt")
SHORT_PROMPT_PATH         = Path("prompts/short.txt")
REFINE_PROMPT_PATH        = Path("prompts/refine.txt")
EXTRACT_NUMBER_PROMPT_PATH = Path("prompts/extract_number.txt")

# Максимум символов извещения, отправляемых в LLM для извлечения номера
EXTRACT_NUMBER_MAX_CHARS = 30000

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
# Swagger/OpenAPI выключены в проде (раскрывали карту API без авторизации).
# Включить для отладки: ENABLE_DOCS=1 в .env.
_docs_on = os.getenv("ENABLE_DOCS", "") == "1"
app = FastAPI(
    title="Анализатор тендерной документации v3",
    docs_url="/docs" if _docs_on else None,
    redoc_url="/redoc" if _docs_on else None,
    openapi_url="/openapi.json" if _docs_on else None,
)

# Инициализация БД
auth_module.init_db()
cache_module.init_db()

# Пути без авторизации
PUBLIC_PATHS = {"/auth/login", "/health", "/favicon.ico"}
if _docs_on:
    PUBLIC_PATHS |= {"/openapi.json", "/docs", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Публичные пути
        if path in PUBLIC_PATHS or path.startswith("/static/"):
            return await call_next(request)

        # Публичная ссылка на отчёт по неугадываемому токену (/r/{token}) — без авторизации.
        # Доступ ограничен валидным share_token внутри обработчика.
        if path.startswith("/r/"):
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


def extract_purchase_number(results: list) -> str | None:
    """Извлекает номер закупки из results модели.

    Стратегия (3 уровня):
      1. Контекстный поиск: «номер извещения: N», «закупка № N» — самый надёжный,
         работает для коротких номеров (Торги РФ: 4360) и длинных (44-ФЗ: 19 цифр).
      2. Длинные числа: 19/11/7-8 цифр (44-ФЗ / 223-ФЗ / РТС-РосТендер).
      3. Любое 4+ значное число, кроме годов (19XX/20XX).

    Возвращает строку с номером или None.
    """
    if not results:
        return None
    for item in results:
        if not isinstance(item, dict):
            continue
        title_upper = (item.get("title") or "").upper()
        if not ("НОМЕР" in title_upper and ("ЗАКУПК" in title_upper or "ИЗВЕЩ" in title_upper)):
            continue
        answer = (item.get("answer") or "")
        if "не найден" in answer.lower():
            return None

        # Стратегия 1: контекстный поиск (4-25 цифр)
        context_patterns = [
            r"номер\s+извещени[яеи][:\s]+№?\s*(\d{3,25})",
            r"извещени[яеи]\s*№\s*(\d{3,25})",
            r"закупк[аи]\s*№\s*(\d{3,25})",
            r"реестровый\s+номер[:\s]+(\d{3,25})",
            r"номер\s+закупк[аи][:\s]+№?\s*(\d{3,25})",
            r"№\s*(\d{4,25})",  # после № — обычно номер, не год
        ]
        for pattern in context_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                num = match.group(1)
                if _is_valid_purchase_number(num):
                    return num

        # Стратегия 2: типичные длины (с возможным буквенным префиксом).
        # 19-25 цифр — 44-ФЗ и региональные (Тульская обл — 20 цифр).
        # 11 — 223-ФЗ. 7-8 — РТС/РосТендер. Префикс ≤5 букв слитно.
        match = re.search(
            r"(?<![A-Za-zА-Яа-яЁё0-9])([A-Za-zА-Яа-яЁё]{1,5})?(\d{19,25}|\d{11}|\d{7,8})(?!\d)",
            answer,
        )
        if match:
            return (match.group(1) or "") + match.group(2)

        # Стратегия 3: расширенный fallback — 4+ цифр (с префиксом), кроме годов
        for match in re.finditer(
            r"(?<![A-Za-zА-Яа-яЁё0-9])([A-Za-zА-Яа-яЁё]{1,5})?(\d{4,25})(?!\d)",
            answer,
        ):
            num = match.group(2)
            if _is_valid_purchase_number(num):
                return (match.group(1) or "") + num
    return None


def _is_valid_purchase_number(num: str) -> bool:
    """Отсекает явные ложные срабатывания (годы, телефоны).

    Длина 4-25 — покрывает все известные форматы:
      19-значные 44-ФЗ, 20-значные региональные (Тульская обл и подобные),
      11-значные 223-ФЗ, 7-8-значные РТС/РосТендер, 4-6-значные Торги РФ.
    """
    if not (4 <= len(num) <= 25):
        return False
    # 4-значные годы (19XX, 20XX, 21XX) — не номер
    if len(num) == 4 and num[:2] in ("19", "20", "21"):
        return False
    return True


def build_files_meta(docs: list[dict], extracted: list) -> list[dict]:
    """Собирает meta-инфу по всем файлам: успешные + пропущенные.

    docs       — валидные документы (из valid после классификации)
    extracted  — всё что вернул extract_files (включая skipped)

    Возвращает список словарей с полями:
      name, type, size_bytes, chars, tokens, status, skip_reason
    """
    meta_files: list[dict] = []
    # Успешные документы
    for d in docs:
        chars = d.get("char_count", 0)
        meta_files.append({
            "name": d["name"],
            "type": d["type"],
            "size_bytes": d.get("size_bytes", 0),
            "chars": chars,
            "tokens": chars // 4,
            "status": "ok" if chars >= 50 else "warn_short",
            "skip_reason": "" if chars >= 50 else "текст короткий (возможно скан без OCR)",
        })
    # Пропущенные
    for f in extracted:
        if f.skipped:
            meta_files.append({
                "name": f.name,
                "type": "—",
                "size_bytes": getattr(f, "size_bytes", 0),
                "chars": 0,
                "tokens": 0,
                "status": "skipped",
                "skip_reason": f.skip_reason,
            })
    return meta_files


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
    """Вызов LLM API с retry основного провайдера и fallback на резервный.

    1) До MAIN_MAX_ATTEMPTS попыток к основному провайдеру (лечит «через раз»
       из-за больного upstream-сервера у agentplatform).
    2) Если все попытки исчерпаны и статус временный — уходим на резерв.
    """
    logger.info(f"[TASK {task_id[:8]}] [{label}] Отправка ({len(prompt_text)} симв.)")

    parsed, status = None, 0
    for attempt in range(1, MAIN_MAX_ATTEMPTS + 1):
        parsed, status = await _try_api(API_URL, API_KEY, MODEL, prompt_text, task_id, label, "MAIN")
        if parsed is not None:
            return parsed
        # status=200 с parsed=None — модель вернула нераспарсиваемый ответ, повтор не поможет
        if status not in RETRY_STATUSES:
            break
        if attempt < MAIN_MAX_ATTEMPTS:
            logger.warning(f"[TASK {task_id[:8]}] [{label}] MAIN попытка {attempt}/{MAIN_MAX_ATTEMPTS} неудачна ({status}), повтор через {RETRY_DELAY}с")
            await asyncio.sleep(RETRY_DELAY)

    # Fallback на резервный провайдер
    if parsed is None and status in RETRY_STATUSES and FALLBACK_API_KEY:
        logger.warning(f"[TASK {task_id[:8]}] [{label}] MAIN недоступен после {MAIN_MAX_ATTEMPTS} попыток ({status}), переключаюсь на резерв")
        parsed, status = await _try_api(FALLBACK_API_URL, FALLBACK_API_KEY, FALLBACK_MODEL,
                                        prompt_text, task_id, label, "FALLBACK")
        if parsed and task_id in tasks:
            tasks[task_id]["fallback_used"] = True

    return parsed


# -- Извлечение номера закупки через LLM (для кэша) ------------------------

async def _try_api_raw_text(url: str, key: str, model: str, prompt_text: str,
                             task_id: str, label: str, provider: str) -> tuple[str | None, int]:
    """Вызов API, возвращает сырой текст ответа (без JSON-парсинга). Для коротких запросов."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.0,
                    "max_tokens": 100,
                },
            )
        if response.status_code != 200:
            logger.error(f"[TASK {task_id[:8]}] [{label}] [{provider}] API ошибка {response.status_code}: {response.text[:200]}")
            return None, response.status_code
        raw = response.json()["choices"][0]["message"]["content"].strip()
        logger.info(f"[TASK {task_id[:8]}] [{label}] [{provider}] Ответ: {raw[:80]}")
        return raw, 200
    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] [{label}] [{provider}] Ошибка: {e}")
        return None, 0


async def extract_number_via_llm(notice_text: str, task_id: str) -> str | None:
    """Отдельный быстрый LLM-вызов: «найди номер закупки» по тексту извещения.

    Использует основную модель, при 429 — резервную. На вход уходит только текст
    извещений (обрезанный до EXTRACT_NUMBER_MAX_CHARS), на выход — одна строка.
    """
    if not notice_text or not notice_text.strip():
        return None
    try:
        prompt = load_prompt(EXTRACT_NUMBER_PROMPT_PATH).replace("{{NOTICE_TEXT}}", notice_text)
    except FileNotFoundError:
        logger.warning(f"[TASK {task_id[:8]}] [EXTRACT_NUMBER] Промпт extract_number.txt не найден")
        return None

    logger.info(f"[TASK {task_id[:8]}] [EXTRACT_NUMBER] Отправка ({len(prompt)} симв.)")

    raw, status = None, 0
    for attempt in range(1, MAIN_MAX_ATTEMPTS + 1):
        raw, status = await _try_api_raw_text(API_URL, API_KEY, MODEL, prompt, task_id, "EXTRACT_NUMBER", "MAIN")
        if raw is not None:
            break
        if status not in RETRY_STATUSES:
            break
        if attempt < MAIN_MAX_ATTEMPTS:
            logger.warning(f"[TASK {task_id[:8]}] [EXTRACT_NUMBER] MAIN попытка {attempt}/{MAIN_MAX_ATTEMPTS} неудачна ({status}), повтор через {RETRY_DELAY}с")
            await asyncio.sleep(RETRY_DELAY)

    if raw is None and status in RETRY_STATUSES and FALLBACK_API_KEY:
        logger.warning(f"[TASK {task_id[:8]}] [EXTRACT_NUMBER] MAIN недоступен после {MAIN_MAX_ATTEMPTS} попыток ({status}) — fallback")
        raw, status = await _try_api_raw_text(FALLBACK_API_URL, FALLBACK_API_KEY, FALLBACK_MODEL,
                                              prompt, task_id, "EXTRACT_NUMBER", "FALLBACK")

    if not raw:
        return None

    text = raw.strip().strip('"').strip("'").strip("`").strip()
    if text.upper() == "NONE" or "не найден" in text.lower():
        return None

    # Поддерживаем буквенно-цифровые номера (ЗП605694 на ТЭК-Торге, Lot12345).
    # Группа 1 — необязательный префикс из 1-5 букв (кириллица или латиница).
    # Группа 2 — цифры (3-25, чтобы покрыть 20-значные региональные номера
    # типа Тульской «Малые закупки» 03662000157202600045).
    match = re.search(r"(?<![A-Za-zА-Яа-яЁё0-9])([A-Za-zА-Яа-яЁё]{1,5})?(\d{3,25})(?!\d)", text)
    if match and _is_valid_purchase_number(match.group(2)):
        prefix = match.group(1) or ""
        return prefix + match.group(2)
    return None


def _restore_task_from_cache(task: dict, cached: dict, user_login: str = "") -> None:
    """Восстанавливает task в состояние 'done' из записи кэша. Инкрементит hit_count.
    Используется и в /decide (use_cache=true), и в авто-кэше API-потока."""
    result_data = json.loads(cached["result_json"])
    files_meta = json.loads(cached["files_meta_json"])
    meta = {
        "files": files_meta,
        "skipped": [{"name": f["name"], "reason": f.get("skip_reason", "")}
                    for f in files_meta if f.get("status") == "skipped"],
        "context_chars": len(cached.get("extracted_text", "")),
        "total_size_bytes": cached.get("total_size_bytes", 0),
    }
    task.update(
        status="done",
        results=result_data.get("results", []),
        risks=result_data.get("risks", []),
        parameters=result_data.get("parameters", []),
        warnings=result_data.get("warnings", []),
        meta=meta,
        text=cached.get("text_report", ""),
        context=cached.get("extracted_text", ""),
        purchase_number=cached["purchase_number"],
        completed_at=cached.get("created_at"),  # дата = когда сделан оригинал
        from_cache=True,
        awaiting_decision=False,
        share_token=cached.get("share_token"),
    )
    cache_module.increment_hit(cached["purchase_number"], user_login)


async def _send_done_callback(task_id: str) -> None:
    """Отправляет done-callback с готовым результатом (для API-потока)."""
    task = tasks.get(task_id, {})
    callback_url = task.get("callback_url")
    if not callback_url:
        return
    try:
        async with httpx.AsyncClient(timeout=30.0) as cb:
            await cb.post(callback_url, json={
                "task_id": task_id,
                "external_id": task.get("external_id"),
                "status": "done",
                "risks": task.get("risks", []),
                "parameters": task.get("parameters", []),
                "results": task.get("results", []),
                "warnings": task.get("warnings", []),
                "text": task.get("text", ""),
                "meta": task.get("meta"),
                "purchase_number": task.get("purchase_number"),
                "from_cache": task.get("from_cache", False),
                "share_url": (f"{PUBLIC_BASE_URL}/r/{task['share_token']}"
                              if task.get("share_token") else None),
            })
        logger.info(f"[TASK {task_id[:8]}] Callback отправлен (from_cache={task.get('from_cache')})")
    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] Ошибка callback: {e}")


async def _try_cache_match(task_id: str, extracted: list, mode: str) -> bool:
    """Pre-step: определяет номер закупки и проверяет кэш.

    Номер берётся из `purchase_number_prefetched` (передан клиентом API напрямую)
    или, если его нет, извлекается моделью из текста извещения.

    Поведение при совпадении в кэше:
      - API-поток (is_api): авто-решение по размеру файлов — совпал → отдаём
        из кэша в callback (без анализа); отличается → свежий анализ.
      - Веб-поток: статус cached_exists, ждём решения пользователя через /decide.

    Возвращает True, если pipeline должен выйти (кэш отдан / ждём решение),
    False — если нужен полный анализ (номер сохранён в purchase_number_prefetched).
    """
    task = tasks[task_id]
    is_api = task.get("is_api", False)

    # --- 1. Определяем номер закупки ---
    number = task.get("purchase_number_prefetched")
    if number:
        logger.info(f"[TASK {task_id[:8]}] [CACHE] Номер из запроса (API): {number}")
    else:
        valid = [f for f in extracted if not f.skipped and f.text.strip()]
        notice_files = [f for f in valid if classify(f.text, f.name) == "ИЗВЕЩЕНИЕ"]
        if not notice_files:
            logger.info(f"[TASK {task_id[:8]}] [CACHE] Извещение не найдено — кэш не проверяем")
            return False

        notice_text = "\n\n".join(f.text for f in notice_files)[:EXTRACT_NUMBER_MAX_CHARS]
        task.update(step="cache_check", detail="Извлечение номера закупки...")
        number = await extract_number_via_llm(notice_text, task_id)

        if not number:
            logger.info(f"[TASK {task_id[:8]}] [CACHE] Номер не извлечён — кэш не проверяем")
            return False

        logger.info(f"[TASK {task_id[:8]}] [CACHE] Номер из извещения: {number}")
        task["purchase_number_prefetched"] = number

        uid = task.get("user_id", 0)
        if uid:
            auth_module.log_usage(uid, "extract_number", 0, len(notice_text) // 4, 10, "done",
                                 purchase_number=number)

    # --- 2. Проверяем кэш ---
    cached = cache_module.get_cache(number)
    if not cached:
        logger.info(f"[TASK {task_id[:8]}] [CACHE] В кэше нет — будет полный анализ")
        return False

    # Размер для diff — то, что юзер реально загрузил (не распакованные внутренности)
    new_size = task.get("upload_total_size", 0)
    cached_size = cached.get("total_size_bytes", 0)
    size_match = (new_size == cached_size)

    # --- 3. API-поток: авто-решение по размеру (нет UI-диалога) ---
    if is_api:
        if size_match:
            user_login = ""
            uid = task.get("user_id", 0)
            if uid:
                u = auth_module.get_user_by_id(uid)
                if u:
                    user_login = u["login"]
            _restore_task_from_cache(task, cached, user_login)
            await _send_done_callback(task_id)
            logger.info(f"[TASK {task_id[:8]}] [CACHE] API: размер совпал — отдан из кэша №{number}")
            return True
        logger.info(f"[TASK {task_id[:8]}] [CACHE] API: размер отличается ({cached_size}→{new_size}) — свежий анализ")
        return False

    # --- 4. Веб-поток: диалог cached_exists ---
    task.update(
        status="cached_exists",
        step="awaiting_decision",
        detail="Тендер уже анализировался ранее — ждём решения пользователя",
        purchase_number=number,
        awaiting_decision=True,
        cached_info={
            "purchase_number": number,
            "created_at": cached["created_at"],
            "expires_at": cached["expires_at"],
            "user_login": cached.get("user_login", ""),
            "mode": cached.get("mode", ""),
            "current_mode": mode,
            "total_size_bytes": cached_size,
            "current_size_bytes": new_size,
            "size_changed": not size_match,
            "hit_count": cached.get("hit_count", 0),
        },
    )
    task["_cached_data"] = cached
    logger.info(
        f"[TASK {task_id[:8]}] [CACHE] CACHED_EXISTS: №{number}, "
        f"размер {'совпадает' if size_match else 'отличается'}"
    )
    return True


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

    # Обработанные файлы (исключая skipped)
    files = meta.get("files", [])
    ok_files = [f for f in files if f.get("status") != "skipped"]
    if ok_files:
        lines.append("Обработанные файлы:")
        for f in ok_files:
            tokens = f.get("tokens") or (f.get("chars", 0) // 4)
            lines.append(f'  + {f["name"]} ({f["type"]}, {f["chars"]} симв., ~{tokens} токенов)')
        lines.append("")

    # Пропущенные файлы — берём из meta.files со status=skipped (предпочтительно),
    # фолбэк на старое meta.skipped для совместимости.
    skipped_from_files = [{"name": f["name"], "reason": f.get("skip_reason", "")}
                          for f in files if f.get("status") == "skipped"]
    skipped = skipped_from_files or meta.get("skipped", [])
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

        # -- Шаг 1.5: Проверка кэша по номеру (skip если force_refresh) --
        if not task.get("force_refresh"):
            if await _try_cache_match(task_id, extracted, "full"):
                # cached_exists — pipeline останавливается, ждём решение через /decide.
                # tmp_dir НЕ удаляем (флаг awaiting_decision проверится в finally).
                return

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
                "size_bytes": getattr(f, "size_bytes", 0),
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

        files_meta = build_files_meta(docs, extracted)
        # total_size_bytes — то, что прислал юзер (для diff), не сумма распакованных
        upload_size = task.get("upload_total_size") or sum(fm["size_bytes"] for fm in files_meta)
        meta = {
            "files": files_meta,
            "skipped": skipped,  # для обратной совместимости со старым UI/Битрикс
            "context_chars": context_chars,
            "total_size_bytes": upload_size,
        }
        flat_results = risks_result + params_result
        # Предпочитаем номер, извлечённый отдельным LLM-запросом на pre-step.
        # Если его нет (нет извещения / NONE) — пытаемся вытащить из самого ответа.
        purchase_number = task.get("purchase_number_prefetched") or extract_purchase_number(flat_results)
        if purchase_number:
            logger.info(f"[TASK {task_id[:8]}] Номер закупки: {purchase_number}")
        # Псевдо-ключ для записей БЕЗ номера — чтобы отчёт всё равно был
        # доступен из админки (юзер кликает «нет номера» и видит этот отчёт).
        cache_key = purchase_number or f"no-{task_id[:8]}"
        text_report = format_report(risks_result, params_result, warnings, meta)

        task.update(
            status="done",
            results=flat_results,
            risks=risks_result,
            parameters=params_result,
            warnings=warnings,
            meta=meta,
            text=text_report,
            context=context,
            purchase_number=purchase_number,
            completed_at=time.time(),
        )

        # Запись в кэш — ВСЕГДА. Если номер не определён, используем псевдо-ключ
        # no-{task_id[:8]} чтобы отчёт всё равно был доступен через админку.
        try:
            user_login = ""
            uid = task.get("user_id", 0)
            if uid:
                u = auth_module.get_user_by_id(uid)
                if u:
                    user_login = u["login"]
            share_token = cache_module.save_cache(
                purchase_number=cache_key, mode="full",
                total_size_bytes=meta["total_size_bytes"],
                result={"results": flat_results, "risks": risks_result,
                        "parameters": params_result, "warnings": warnings},
                files_meta=files_meta, extracted_text=context,
                text_report=text_report, user_login=user_login,
            )
            task["share_token"] = share_token
        except Exception as ce:
            logger.error(f"[TASK {task_id[:8]}] [CACHE] Ошибка записи: {ce}")

        # Логирование usage — пишем cache_key (реальный номер ИЛИ no-XXX),
        # чтобы клик из логов всегда находил отчёт.
        uid = task.get("user_id", 0)
        if uid:
            tokens_out = sum(len(json.dumps(r, ensure_ascii=False)) // 4 for r in flat_results)
            auth_module.log_usage(uid, "full", task.get("files_count", 0),
                                 context_chars // 4, tokens_out, "done",
                                 purchase_number=cache_key)

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
                        "purchase_number": purchase_number,
                        "share_url": (f"{PUBLIC_BASE_URL}/r/{task['share_token']}"
                                      if task.get("share_token") else None),
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
        # Не удаляем tmp_dir пока ждём решения пользователя по кэшу —
        # файлы могут понадобиться для перезапуска через /decide.
        if not tasks.get(task_id, {}).get("awaiting_decision"):
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

        # -- Шаг 1.5: Проверка кэша по номеру (skip если force_refresh) --
        if not task.get("force_refresh"):
            if await _try_cache_match(task_id, extracted, "short"):
                return  # cached_exists — ждём /decide

        task.update(step="classifying", detail="Обработка документов...")
        docs = []
        for f in valid:
            doc_type = classify(f.text, f.name)
            sections = split_into_sections(f.text)
            docs.append({"name": f.name, "type": doc_type,
                         "sections": sections, "char_count": len(f.text),
                         "size_bytes": getattr(f, "size_bytes", 0)})
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

        files_meta = build_files_meta(docs, extracted)
        upload_size = task.get("upload_total_size") or sum(fm["size_bytes"] for fm in files_meta)
        meta = {"files": files_meta,
                "skipped": [{"name": f.name, "reason": f.skip_reason} for f in extracted if f.skipped],
                "context_chars": context_chars,
                "total_size_bytes": upload_size}

        purchase_number = task.get("purchase_number_prefetched") or extract_purchase_number(parsed)
        if purchase_number:
            logger.info(f"[TASK {task_id[:8]}] Номер закупки: {purchase_number}")
        cache_key = purchase_number or f"no-{task_id[:8]}"

        text_report = format_report([], parsed, [], meta)
        task.update(status="done", results=parsed, parameters=parsed, risks=[],
                    warnings=[], meta=meta, text=text_report, context=context,
                    purchase_number=purchase_number, completed_at=time.time())

        # Запись в кэш — ВСЕГДА (даже без номера, с псевдо-ключом)
        try:
            user_login = ""
            uid = task.get("user_id", 0)
            if uid:
                u = auth_module.get_user_by_id(uid)
                if u:
                    user_login = u["login"]
            share_token = cache_module.save_cache(
                purchase_number=cache_key, mode="short",
                total_size_bytes=meta["total_size_bytes"],
                result={"results": parsed, "risks": [], "parameters": parsed, "warnings": []},
                files_meta=files_meta, extracted_text=context,
                text_report=text_report, user_login=user_login,
            )
            task["share_token"] = share_token
        except Exception as ce:
            logger.error(f"[TASK {task_id[:8]}] [CACHE] Ошибка записи: {ce}")

        # Логирование usage — пишем cache_key чтобы из админки клик находил отчёт
        uid = task.get("user_id", 0)
        if uid:
            tokens_out = sum(len(json.dumps(r, ensure_ascii=False)) // 4 for r in parsed)
            auth_module.log_usage(uid, "short", task.get("files_count", 0),
                                 context_chars // 4, tokens_out, "done",
                                 purchase_number=cache_key)

        callback_url = task.get("callback_url")
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as cb:
                    await cb.post(callback_url, json={
                        "task_id": task_id,
                        "external_id": task.get("external_id"),
                        "status": "done",
                        "results": parsed, "text": text_report, "meta": meta,
                        "purchase_number": purchase_number,
                        "share_url": (f"{PUBLIC_BASE_URL}/r/{task['share_token']}"
                                      if task.get("share_token") else None),
                    })
                logger.info(f"[TASK {task_id[:8]}] Callback отправлен")
            except Exception as cb_err:
                logger.error(f"[TASK {task_id[:8]}] Ошибка callback: {cb_err}")

    except Exception as e:
        logger.error(f"[TASK {task_id[:8]}] Ошибка: {e}", exc_info=True)
        tasks[task_id].update(status="error", detail=str(e)[:200])
    finally:
        if not tasks.get(task_id, {}).get("awaiting_decision"):
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

        files_meta_legacy = build_files_meta(docs, extracted)
        upload_size = task.get("upload_total_size") or sum(fm["size_bytes"] for fm in files_meta_legacy)
        meta = {"files": files_meta_legacy,
                "context_chars": context_chars,
                "total_size_bytes": upload_size}

        purchase_number = extract_purchase_number(parsed)
        if purchase_number:
            logger.info(f"[TASK {task_id[:8]}] Номер закупки: {purchase_number}")

        lines = ["АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ", ""]
        for i, item in enumerate(parsed, 1):
            lines.append(f'{i}. {item.get("title", "")}')
            lines.append(f'   {item.get("answer", "")}')
            for src in item.get("sources", []):
                lines.append(f'   [{src.get("doc","")}, {src.get("ref","")} — «{src.get("citation","")}»]')
            if not item.get("sources") and item.get("citation", "").strip() not in ("", "—"):
                lines.append(f'   [{item["citation"]}]')
            lines.append("")

        task.update(status="done", results=parsed, meta=meta, text="\n".join(lines).strip(),
                    purchase_number=purchase_number, completed_at=time.time())

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
    purchase_number: str | None = None  # номер закупки от клиента (Saby/Битрикс знают точно)


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
    purchase_number: str = Form(None),
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
    upload_total_size = 0  # сумма ровно того, что прислал юзер (а не распакованных файлов)
    for idx, upload in enumerate(files):
        # safe_filename — защита от path traversal через имя файла (../../app/main.py)
        fname = safe_filename(upload.filename, f"file_{idx}")
        dest = tmp_dir / fname
        content = await upload.read()
        dest.write_bytes(content)
        saved_paths.append(dest)
        upload_total_size += len(content)
        logger.info(f"Получен: {fname} ({len(content)} байт)")

    # Номер закупки от программного клиента (Saby-интеграция качает файлы у себя
    # с сессией и заливает multipart'ом, т.к. trade.sbis.ru требует авторизацию).
    # Если номер передан — это headless-вызов: модель для извлечения номера НЕ
    # нужна, и кэш решается автоматически по размеру (is_api=True), без UI-диалога.
    prefetched_number = sanitize_purchase_number(purchase_number)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing", "step": "uploading",
        "detail": "Файлы загружены...", "created": time.time(), "tmp_dir": str(tmp_dir),
        "user_id": user_id, "mode": mode, "files_count": len(saved_paths),
        "upload_total_size": upload_total_size,
        "purchase_number_prefetched": prefetched_number,
        "is_api": prefetched_number is not None,
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


def _check_task_owner(task: dict, request: Request) -> bool:
    """True, если текущий пользователь — владелец задачи или админ.
    Защита от IDOR: чужие отчёты по task_id (UUID) читать нельзя."""
    user = getattr(request.state, "user", None)
    if not user:
        return False
    if user.get("role") == "admin":
        return True
    owner = task.get("user_id")
    # owner=0/None — задача без владельца (легаси): доступ только админу (выше)
    return bool(owner) and owner == user.get("id")


@app.get("/status/{task_id}")
async def get_status(task_id: str, request: Request):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if not _check_task_owner(task, request):
        # 404 (не 403) — не раскрываем существование чужой задачи
        raise HTTPException(status_code=404, detail="Задача не найдена")

    if task["status"] == "done":
        completed_at_ts = task.get("completed_at")
        return JSONResponse({
            "status": "done",
            "risks": task.get("risks", []),
            "parameters": task.get("parameters", []),
            "results": task.get("results"),
            "warnings": task.get("warnings", []),
            "text": task.get("text"),
            "meta": task.get("meta"),
            "purchase_number": task.get("purchase_number"),
            "completed_at": completed_at_ts,
            "from_cache": task.get("from_cache", False),
            "share_url": (f"{PUBLIC_BASE_URL}/r/{task['share_token']}"
                          if task.get("share_token") else None),
        })
    elif task["status"] == "cached_exists":
        return JSONResponse({
            "status": "cached_exists",
            "purchase_number": task.get("purchase_number"),
            "cached_info": task.get("cached_info"),
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


class DecideRequest(BaseModel):
    use_cache: bool


@app.post("/analyze/{task_id}/decide")
async def decide(task_id: str, req: DecideRequest, request: Request):
    """Решение пользователя по cached_exists: использовать кэш или повторить анализ."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if not _check_task_owner(task, request):
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if task.get("status") != "cached_exists":
        raise HTTPException(status_code=400, detail="Задача не в состоянии cached_exists")

    cached = task.get("_cached_data")
    if not cached:
        raise HTTPException(status_code=500, detail="Кэш-данные потеряны")

    user = getattr(request.state, "user", None)
    user_login = user["login"] if user else ""

    if req.use_cache:
        try:
            _restore_task_from_cache(task, cached, user_login)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Битый кэш: {e}")
        # Удаляем tmp_dir — он больше не нужен
        tmp_dir_str = task.get("tmp_dir")
        if tmp_dir_str:
            shutil.rmtree(tmp_dir_str, ignore_errors=True)
        logger.info(f"[TASK {task_id[:8]}] [CACHE] Отдан из кэша: №{cached['purchase_number']}")
        return JSONResponse({"status": "done", "from_cache": True})

    # use_cache=false → перезапуск полного анализа
    tmp_dir_str = task.get("tmp_dir")
    if not tmp_dir_str or not Path(tmp_dir_str).exists():
        raise HTTPException(status_code=500, detail="Файлы потеряны, загрузите заново")

    tmp_dir = Path(tmp_dir_str)
    saved_paths = [p for p in tmp_dir.iterdir() if p.is_file()]
    if not saved_paths:
        raise HTTPException(status_code=500, detail="Файлы не найдены, загрузите заново")

    # Сбрасываем cached_exists, ставим флаг force_refresh — pipeline пропустит кэш-проверку
    task.update(
        status="processing", step="extracting",
        detail="Повторный анализ запущен...",
        awaiting_decision=False, force_refresh=True,
    )
    task.pop("_cached_data", None)
    task.pop("cached_info", None)

    mode = task.get("mode", "full")
    if mode == "short":
        asyncio.create_task(process_task_short(task_id, saved_paths, tmp_dir))
    else:
        asyncio.create_task(process_task_v3(task_id, saved_paths, tmp_dir))
    logger.info(f"[TASK {task_id[:8]}] [CACHE] Перезапуск (force_refresh), mode={mode}")
    return JSONResponse({"status": "processing", "from_cache": False})


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
    upload_total_size = 0

    try:
        # follow_redirects=False — иначе внешний 302 обходит SSRF-проверку исходного URL
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as dl:
            for idx, url in enumerate(req.files):
                ok, reason = is_safe_outbound_url(url)
                if not ok:
                    logger.warning(f"[API] URL отклонён (SSRF-защита): {url[:80]} — {reason}")
                    continue
                try:
                    resp = await dl.get(url)
                    if resp.is_redirect:
                        logger.warning(f"[API] Редирект отклонён (SSRF-защита): {url[:80]}")
                        continue
                    resp.raise_for_status()
                except Exception as e:
                    logger.error(f"[API] Скачивание {url[:80]}: {e}")
                    continue
                content = resp.content
                if len(content) > MAX_DOWNLOAD_BYTES:
                    logger.warning(f"[API] Файл превышает лимит {MAX_DOWNLOAD_BYTES} байт: {url[:80]}")
                    continue
                filename = safe_filename(url.split("?")[0].split("/")[-1], f"file_{idx}")
                dest = tmp_dir / filename
                dest.write_bytes(content)
                saved_paths.append(dest)
                upload_total_size += len(content)
                logger.info(f"[API] Скачан: {filename} ({len(content)} байт)")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Ошибка скачивания файлов")

    if not saved_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Не удалось скачать ни одного файла (проверьте URL; внутренние адреса запрещены)")

    # callback_url — сервер будет POST-ить на него результат. Защита от SSRF:
    # запрещаем внутренние адреса (иначе атакующий бьёт по соседним контейнерам/127.0.0.1).
    if req.callback_url:
        ok, reason = is_safe_outbound_url(req.callback_url)
        if not ok:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"callback_url отклонён: {reason}")

    # Номер закупки от клиента (Saby/Битрикс знают точно) — валидируем (анти-XSS),
    # если задан — модель для извлечения номера НЕ вызывается.
    prefetched_number = sanitize_purchase_number(req.purchase_number)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing", "step": "uploading",
        "detail": "Файлы скачаны...", "created": time.time(),
        "tmp_dir": str(tmp_dir), "callback_url": req.callback_url,
        "user_id": user_id, "mode": req.mode, "files_count": len(saved_paths),
        "external_id": req.external_id,
        "upload_total_size": upload_total_size,
        "is_api": True,
        "purchase_number_prefetched": prefetched_number,
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
    # Secure — сайт только по HTTPS (nginx редиректит 80→443); SameSite=Lax — анти-CSRF
    response.set_cookie("session", session_token, httponly=True, secure=True,
                        samesite="lax", max_age=86400 * 7)
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


@app.delete("/admin/api/delete_user/{user_id}")
async def admin_delete_user(user_id: int, request: Request):
    """Удалить пользователя. Защита: нельзя удалить главного admin (id=1)
    и нельзя удалить самого себя."""
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    if user["id"] == user_id:
        raise HTTPException(status_code=400, detail="Нельзя удалить самого себя")
    ok, msg = auth_module.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    # Инвалидируем активные cookie-сессии удалённого пользователя
    expired_tokens = [tok for tok, uid in sessions.items() if uid == user_id]
    for tok in expired_tokens:
        del sessions[tok]
    if expired_tokens:
        logger.info(f"[ADMIN] Инвалидировано {len(expired_tokens)} активных сессий удалённого user_id={user_id}")
    logger.info(f"[ADMIN] {user['login']} удалил пользователя: {msg}")
    return JSONResponse({"ok": True, "login": msg})


# -- Admin API: кэш тендеров ---------------------------------------------

@app.get("/admin/api/cache_list")
async def admin_cache_list(request: Request):
    """Список записей кэша (без тяжёлого extracted_text)."""
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    return JSONResponse(cache_module.get_all_cached())


@app.get("/admin/api/cache_view/{purchase_number}")
async def admin_cache_view(purchase_number: str, request: Request):
    """Полный отчёт из кэша (text + files_meta) для просмотра."""
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    cached = cache_module.get_cache(purchase_number)
    if not cached:
        raise HTTPException(status_code=404, detail="Запись не найдена или истекла")
    try:
        result_data = json.loads(cached["result_json"])
        files_meta = json.loads(cached["files_meta_json"])
    except Exception:
        result_data, files_meta = {}, []
    return JSONResponse({
        "purchase_number": cached["purchase_number"],
        "mode": cached["mode"],
        "created_at": cached["created_at"],
        "expires_at": cached["expires_at"],
        "total_size_bytes": cached["total_size_bytes"],
        "user_login": cached.get("user_login"),
        "user_login_last": cached.get("user_login_last"),
        "hit_count": cached.get("hit_count", 0),
        "last_accessed": cached.get("last_accessed"),
        "text_report": cached.get("text_report", ""),
        "files_meta": files_meta,
        "results": result_data.get("results", []),
        "risks": result_data.get("risks", []),
        "parameters": result_data.get("parameters", []),
    })


@app.delete("/admin/api/cache_delete/{purchase_number}")
async def admin_cache_delete(purchase_number: str, request: Request):
    """Принудительная инвалидация записи кэша."""
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    deleted = cache_module.delete_cache(purchase_number)
    logger.info(f"[ADMIN] {user['login']} удалил кэш №{purchase_number}: existed={deleted}")
    return JSONResponse({"ok": True, "deleted": deleted})


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
    if not _check_task_owner(task, request):
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
        auth_module.log_usage(user_id, "refine", 0, len(context) // 4, tokens_out, "done",
                             purchase_number=task.get("purchase_number"))

    return JSONResponse({"item": parsed[0]})


# -- Публичная ссылка на отчёт (без авторизации, по неугадываемому токену) --

@app.get("/r/{token}", response_class=HTMLResponse)
async def public_report(token: str):
    """Публичная read-only страница отчёта по share_token из кэша.
    Токен неугадываем (token_urlsafe) → ссылка не перечислима. Контент —
    анализ ПУБЛИЧНОЙ тендерной документации, отдельная авторизация не нужна."""
    cached = cache_module.get_by_token(token)
    if not cached:
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8><title>Отчёт не найден</title>"
            "<body style='max-width:700px;margin:48px auto;font:15px/1.5 system-ui;color:#444'>"
            "<h2>Отчёт не найден</h2><p>Ссылка недействительна или срок хранения отчёта истёк "
            f"(отчёты хранятся {cache_module.CACHE_TTL_DAYS} дней).</p>",
            status_code=404,
        )
    num = cached.get("purchase_number", "")
    title = f"Отчёт по закупке № {escape(str(num))}" if not str(num).startswith("no-") else "Отчёт по закупке"
    report = escape(cached.get("text_report", "") or "")
    html = (
        "<!doctype html><meta charset=utf-8>"
        f"<title>{title}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<body style='max-width:900px;margin:24px auto;padding:0 16px;"
        "font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a1a'>"
        f"<h2 style='color:#208C8D'>{title}</h2>"
        "<pre style='white-space:pre-wrap;word-wrap:break-word;font:inherit'>"
        f"{report}</pre>"
        "<hr style='margin-top:32px;border:none;border-top:1px solid #e0e0e0'>"
        "<p style='color:#888;font-size:13px'>Анализ тендерной документации · robotender.ru</p>"
    )
    # Засчитываем просмотр (как обращение к кэшу)
    try:
        cache_module.increment_hit(cached["purchase_number"], "public-link")
    except Exception:
        pass
    return HTMLResponse(html)


# Статика ПОСЛЕ роутов
app.mount("/static", StaticFiles(directory="static"), name="static")
