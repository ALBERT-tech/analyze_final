"""
auth.py — Авторизация, пользователи, лимиты, логирование.

SQLite-хранилище: data/app.db
"""

import sqlite3
import uuid
import hashlib
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

DB_PATH = Path("data/app.db")
MSK_TZ = timezone(timedelta(hours=3))


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _hash_password(password: str) -> str:
    """Хэш пароля через SHA-256 + salt (простой вариант без bcrypt для минимизации зависимостей)."""
    salt = os.urandom(16).hex()
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{h}"


def _check_password(password: str, password_hash: str) -> bool:
    salt, h = password_hash.split(":", 1)
    return hashlib.sha256((salt + password).encode()).hexdigest() == h


def init_db():
    """Создаёт таблицы и admin-пользователя при первом запуске."""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            login TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            active INTEGER NOT NULL DEFAULT 1,
            daily_token_limit INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            mode TEXT,
            files_count INTEGER DEFAULT 0,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            status TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Миграция: добавить purchase_number в usage_log если его нет
    cols = [r[1] for r in conn.execute("PRAGMA table_info(usage_log)").fetchall()]
    if "purchase_number" not in cols:
        conn.execute("ALTER TABLE usage_log ADD COLUMN purchase_number TEXT")
        logger.info("[AUTH] Миграция: добавлен столбец usage_log.purchase_number")

    # Создаём admin если нет
    existing = conn.execute("SELECT id FROM users WHERE login = ?", ("Admin",)).fetchone()
    if not existing:
        create_user(conn, "Admin", "AdmPass12=", role="admin")
        logger.info("[AUTH] Создан admin-пользователь: Admin")

    conn.close()


def create_user(conn: sqlite3.Connection | None = None, login: str = "", password: str = "",
                role: str = "user", daily_limit: int = 0) -> dict:
    """Создаёт пользователя. Возвращает dict с данными."""
    own_conn = conn is None
    if own_conn:
        conn = _get_db()

    api_key = str(uuid.uuid4())
    password_hash = _hash_password(password)
    now = datetime.now(MSK_TZ).isoformat()

    conn.execute(
        "INSERT INTO users (login, password_hash, api_key, role, active, daily_token_limit, created_at) "
        "VALUES (?, ?, ?, ?, 1, ?, ?)",
        (login, password_hash, api_key, role, daily_limit, now),
    )
    conn.commit()

    user = conn.execute("SELECT * FROM users WHERE login = ?", (login,)).fetchone()
    result = dict(user)

    if own_conn:
        conn.close()

    logger.info(f"[AUTH] Создан пользователь: {login} ({role}), api_key={api_key[:8]}...")
    return result


def authenticate(login: str, password: str) -> dict | None:
    """Проверяет логин/пароль. Возвращает user dict или None."""
    conn = _get_db()
    user = conn.execute("SELECT * FROM users WHERE login = ?", (login,)).fetchone()
    conn.close()

    if not user:
        return None
    if not user["active"]:
        return None
    if not _check_password(password, user["password_hash"]):
        return None

    return dict(user)


def get_user_by_api_key(api_key: str) -> dict | None:
    """Ищет пользователя по API-ключу."""
    conn = _get_db()
    user = conn.execute("SELECT * FROM users WHERE api_key = ? AND active = 1", (api_key,)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_user_by_id(user_id: int) -> dict | None:
    conn = _get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_all_users() -> list[dict]:
    conn = _get_db()
    users = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
    conn.close()
    return [dict(u) for u in users]


def update_user(user_id: int, **kwargs):
    """Обновляет поля пользователя. Поддерживает: active, daily_token_limit, role."""
    conn = _get_db()
    for key, value in kwargs.items():
        if key in ("active", "daily_token_limit", "role"):
            conn.execute(f"UPDATE users SET {key} = ? WHERE id = ?", (value, user_id))
    conn.commit()
    conn.close()


def change_password(user_id: int, new_password: str):
    conn = _get_db()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                 (_hash_password(new_password), user_id))
    conn.commit()
    conn.close()


def reset_password(login: str, new_password: str) -> bool:
    """Сброс пароля по логину (для консоли сервера)."""
    conn = _get_db()
    user = conn.execute("SELECT id FROM users WHERE login = ?", (login,)).fetchone()
    if not user:
        conn.close()
        return False
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                 (_hash_password(new_password), user["id"]))
    conn.commit()
    conn.close()
    return True


# -- Лимиты ----------------------------------------------------------------

def check_daily_limit(user_id: int) -> tuple[bool, int, int]:
    """
    Проверяет дневной лимит.
    Returns: (allowed, used_today, limit)
    """
    conn = _get_db()
    user = conn.execute("SELECT daily_token_limit FROM users WHERE id = ?", (user_id,)).fetchone()
    limit = user["daily_token_limit"] if user else 0

    # Сегодня по Москве
    today_start = datetime.now(MSK_TZ).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    row = conn.execute(
        "SELECT COALESCE(SUM(tokens_in + tokens_out), 0) as total "
        "FROM usage_log WHERE user_id = ? AND timestamp >= ?",
        (user_id, today_start),
    ).fetchone()
    conn.close()

    used = row["total"]

    if limit == 0:  # безлимит
        return True, used, 0

    return used < limit, used, limit


# -- Логирование -----------------------------------------------------------

def log_usage(user_id: int, mode: str, files_count: int,
              tokens_in: int, tokens_out: int, status: str,
              purchase_number: str | None = None):
    """Записывает использование. purchase_number — номер закупки, если известен."""
    conn = _get_db()
    conn.execute(
        "INSERT INTO usage_log (user_id, timestamp, mode, files_count, tokens_in, tokens_out, status, purchase_number) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, datetime.now(MSK_TZ).isoformat(), mode, files_count, tokens_in, tokens_out, status, purchase_number),
    )
    conn.commit()
    conn.close()


def get_user_stats(user_id: int | None = None) -> list[dict]:
    """Статистика по пользователям. Если user_id=None — все."""
    conn = _get_db()

    today_start = datetime.now(MSK_TZ).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    week_start = (datetime.now(MSK_TZ) - timedelta(days=7)).isoformat()

    query = """
        SELECT
            u.id, u.login, u.role, u.active, u.daily_token_limit, u.api_key,
            COALESCE(SUM(CASE WHEN l.timestamp >= ? THEN l.tokens_in + l.tokens_out ELSE 0 END), 0) as today_tokens,
            COALESCE(SUM(CASE WHEN l.timestamp >= ? THEN l.tokens_in + l.tokens_out ELSE 0 END), 0) as week_tokens,
            COALESCE(SUM(l.tokens_in + l.tokens_out), 0) as total_tokens,
            COUNT(l.id) as total_requests
        FROM users u
        LEFT JOIN usage_log l ON u.id = l.user_id
    """

    if user_id:
        query += " WHERE u.id = ?"
        rows = conn.execute(query + " GROUP BY u.id", (today_start, week_start, user_id)).fetchall()
    else:
        rows = conn.execute(query + " GROUP BY u.id ORDER BY u.login", (today_start, week_start)).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def get_user_log(user_id: int, limit: int = 50) -> list[dict]:
    """Детальный лог пользователя."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM usage_log WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
