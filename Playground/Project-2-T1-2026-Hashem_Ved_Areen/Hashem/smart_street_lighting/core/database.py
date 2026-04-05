"""
PostgreSQL database module for chat history persistence.

Manages conversations and messages using a connection pool.
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
from uuid import UUID

import psycopg2
import psycopg2.pool
import psycopg2.extras
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

# Register UUID adapter
psycopg2.extras.register_uuid()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://localhost:5432/smart_lighting"
)

_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None


def get_pool() -> psycopg2.pool.SimpleConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        _pool = psycopg2.pool.SimpleConnectionPool(1, 10, DATABASE_URL)
    return _pool


@contextmanager
def get_conn():
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ============================================================
# Conversations
# ============================================================

def create_conversation(title: str = "New Conversation") -> dict:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO conversations (title) VALUES (%s) RETURNING *",
                (title,),
            )
            return dict(cur.fetchone())


def list_conversations(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT %s",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]


def get_conversation(conversation_id: UUID) -> Optional[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM conversations WHERE id = %s", (conversation_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None


def update_conversation_title(conversation_id: UUID, title: str) -> Optional[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "UPDATE conversations SET title = %s, updated_at = now() "
                "WHERE id = %s RETURNING *",
                (title, conversation_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def delete_conversation(conversation_id: UUID) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE id = %s", (conversation_id,)
            )
            return cur.rowcount > 0


def touch_conversation(conversation_id: UUID):
    """Update the updated_at timestamp."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET updated_at = now() WHERE id = %s",
                (conversation_id,),
            )


# ============================================================
# Messages
# ============================================================

def add_message(
    conversation_id: UUID,
    role: str,
    content: str,
    design_data: Optional[dict] = None,
) -> dict:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content, design_data) "
                "VALUES (%s, %s, %s, %s) RETURNING *",
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(design_data) if design_data else None,
                ),
            )
            row = dict(cur.fetchone())
            # Also bump conversation updated_at
            cur.execute(
                "UPDATE conversations SET updated_at = now() WHERE id = %s",
                (conversation_id,),
            )
            return row


def get_messages(conversation_id: UUID) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM messages WHERE conversation_id = %s "
                "ORDER BY created_at ASC",
                (conversation_id,),
            )
            return [dict(row) for row in cur.fetchall()]
