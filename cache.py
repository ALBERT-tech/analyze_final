"""
cache.py — Кэш анализов тендеров по номеру закупки.

Хранится в общей БД (data/app.db, рядом с users).
TTL по умолчанию 30 дней. Просроченные записи удаляются автоматически
при каждом get_cache() / get_all_cached().

Структура tender_cache:
    purchase_number   — PK, строка-номер закупки на площадке (4-19 цифр)
    mode              — 'short' | 'full'
    created_at        — unix timestamp создания
    expires_at        — unix timestamp истечения
    total_size_bytes  — сумма размеров загруженных файлов (для diff)
    result_json       — полный JSON results модели
    files_meta_json   — список файлов с категориями/символами/токенами
    extracted_text    — собранный context (для refine на cached)
    text_report       — готовый текстовый отчёт
    user_login        — кто первый запустил
    user_login_last   — кто последний обращался
    hit_count         — сколько раз отдавалось из кэша
    last_accessed     — unix timestamp последнего обращения
"""

import sqlite3
import json
import logging
import time
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/app.db")
CACHE_TTL_DAYS = 30
CACHE_TTL_SECONDS = CACHE_TTL_DAYS * 86400


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Создаёт таблицу tender_cache при первом запуске + миграции."""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tender_cache (
            purchase_number   TEXT PRIMARY KEY,
            mode              TEXT,
            created_at        REAL NOT NULL,
            expires_at        REAL NOT NULL,
            total_size_bytes  INTEGER DEFAULT 0,
            result_json       TEXT NOT NULL,
            files_meta_json   TEXT NOT NULL,
            extracted_text    TEXT NOT NULL,
            text_report       TEXT NOT NULL,
            user_login        TEXT,
            user_login_last   TEXT,
            hit_count         INTEGER DEFAULT 0,
            last_accessed     REAL,
            share_token       TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_cache_expires ON tender_cache(expires_at);
    """)

    # Миграция: share_token для публичной ссылки на отчёт (/r/{token})
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tender_cache)").fetchall()]
    if "share_token" not in cols:
        conn.execute("ALTER TABLE tender_cache ADD COLUMN share_token TEXT")
        logger.info("[CACHE] Миграция: добавлен столбец share_token")

    # Backfill: каждой старой строке без токена — свой случайный токен
    missing = conn.execute(
        "SELECT purchase_number FROM tender_cache WHERE share_token IS NULL OR share_token = ''"
    ).fetchall()
    for r in missing:
        conn.execute("UPDATE tender_cache SET share_token = ? WHERE purchase_number = ?",
                     (secrets.token_urlsafe(12), r[0]))
    if missing:
        logger.info(f"[CACHE] Backfill share_token: {len(missing)} записей")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_token ON tender_cache(share_token)")
    conn.commit()
    conn.close()


def cleanup_expired() -> int:
    """Удаляет записи с истёкшим TTL. Возвращает число удалённых."""
    conn = _get_db()
    cur = conn.execute("DELETE FROM tender_cache WHERE expires_at < ?", (time.time(),))
    n = cur.rowcount
    conn.commit()
    conn.close()
    if n > 0:
        logger.info(f"[CACHE] Удалено {n} истёкших записей")
    return n


def get_cache(purchase_number: str) -> dict | None:
    """Возвращает запись по номеру или None если нет/истекла."""
    cleanup_expired()
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM tender_cache WHERE purchase_number = ? AND expires_at > ?",
        (purchase_number, time.time()),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_cache(
    purchase_number: str,
    mode: str,
    total_size_bytes: int,
    result: list,
    files_meta: list,
    extracted_text: str,
    text_report: str,
    user_login: str = "",
) -> str:
    """Сохраняет или перезаписывает запись (UPSERT). hit_count и share_token
    сохраняются при перезаписи. Возвращает share_token (для публичной ссылки)."""
    now = time.time()
    expires = now + CACHE_TTL_SECONDS
    new_token = secrets.token_urlsafe(12)  # используется только если строка новая
    conn = _get_db()
    conn.execute("""
        INSERT INTO tender_cache (
            purchase_number, mode, created_at, expires_at, total_size_bytes,
            result_json, files_meta_json, extracted_text, text_report,
            user_login, user_login_last, hit_count, last_accessed, share_token
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        ON CONFLICT(purchase_number) DO UPDATE SET
            mode = excluded.mode,
            created_at = excluded.created_at,
            expires_at = excluded.expires_at,
            total_size_bytes = excluded.total_size_bytes,
            result_json = excluded.result_json,
            files_meta_json = excluded.files_meta_json,
            extracted_text = excluded.extracted_text,
            text_report = excluded.text_report,
            user_login_last = excluded.user_login_last,
            last_accessed = excluded.last_accessed
    """, (
        purchase_number, mode, now, expires, total_size_bytes,
        json.dumps(result, ensure_ascii=False), json.dumps(files_meta, ensure_ascii=False),
        extracted_text, text_report, user_login, user_login, now, new_token,
    ))
    # На случай старой строки без токена — проставим (UPSERT его не трогает)
    conn.execute(
        "UPDATE tender_cache SET share_token = ? WHERE purchase_number = ? "
        "AND (share_token IS NULL OR share_token = '')",
        (new_token, purchase_number),
    )
    row = conn.execute(
        "SELECT share_token FROM tender_cache WHERE purchase_number = ?", (purchase_number,)
    ).fetchone()
    conn.commit()
    conn.close()
    token = row["share_token"] if row else new_token
    logger.info(f"[CACHE] Сохранено: №{purchase_number}, mode={mode}, TTL {CACHE_TTL_DAYS} дней")
    return token


def get_by_token(token: str) -> dict | None:
    """Возвращает запись по share_token (для публичного /r/{token}) или None
    если токен не найден / запись истекла."""
    if not token:
        return None
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM tender_cache WHERE share_token = ? AND expires_at > ?",
        (token, time.time()),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def increment_hit(purchase_number: str, user_login: str = "") -> None:
    """Увеличивает hit_count и обновляет user_login_last/last_accessed."""
    conn = _get_db()
    conn.execute("""
        UPDATE tender_cache
        SET hit_count = hit_count + 1, user_login_last = ?, last_accessed = ?
        WHERE purchase_number = ?
    """, (user_login, time.time(), purchase_number))
    conn.commit()
    conn.close()


def delete_cache(purchase_number: str) -> bool:
    """Принудительная инвалидация. Возвращает True если запись была."""
    conn = _get_db()
    cur = conn.execute("DELETE FROM tender_cache WHERE purchase_number = ?", (purchase_number,))
    n = cur.rowcount
    conn.commit()
    conn.close()
    return n > 0


def get_all_cached() -> list[dict]:
    """Список всех записей (без extracted_text/result_json — для админки/листинга)."""
    cleanup_expired()
    conn = _get_db()
    rows = conn.execute(
        "SELECT purchase_number, mode, created_at, expires_at, total_size_bytes, "
        "user_login, user_login_last, hit_count, last_accessed, "
        "LENGTH(extracted_text) AS ctx_chars "
        "FROM tender_cache ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
