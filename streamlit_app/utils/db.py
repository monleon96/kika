"""
Database helpers for the Streamlit UI.

This module centralizes access to the SQLite database used to store
authentication and user preference data. Keeping the connection logic in
one place makes it easy to swap the backend in the future (e.g. move to
PostgreSQL) while the rest of the app continues to call the same helpers.

Set `KIKA_DB_PATH` in the environment to override the default database
location when deploying or sharing a network-mounted SQLite file.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "kika.db"
_DB_PATH = Path(os.getenv("KIKA_DB_PATH", str(_DEFAULT_DB_PATH)))
if _DB_PATH.parent:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Each thread lazily opens a single SQLite connection. Streamlit can execute
# callbacks in different threads, so we avoid cross-thread usage of sqlite3
# connections by keeping them thread-local.
_thread_local = threading.local()


def _create_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def get_connection() -> sqlite3.Connection:
    """
    Return a cached SQLite connection for the current thread.
    """
    conn = getattr(_thread_local, "connection", None)
    if conn is None:
        conn = _create_connection()
        _thread_local.connection = conn
    return conn


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    """
    Context manager that yields a cursor and commits on success.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def init_db() -> None:
    """
    Ensure the required tables exist. The schema is intentionally simple:

    - users table stores basic account information.
    - user_settings table stores JSON blobs of per-user configuration.

    Both tables can be migrated easily if the backend changes later.
    """
    with db_cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                email_verified INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                settings_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        if "email_verified" not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
            # Assume legacy accounts were already trusted.
            cursor.execute("UPDATE users SET email_verified = 1 WHERE email_verified IS NULL OR email_verified = 0")
