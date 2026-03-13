from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint, current_app, g, jsonify


db_api = Blueprint("db_api", __name__, url_prefix="/db")


def _dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_db() -> sqlite3.Connection:
    conn = g.get("db_conn")
    if conn is None:
        db_path = current_app.config["DATABASE_PATH"]
        conn = sqlite3.connect(db_path)
        conn.row_factory = _dict_factory
        g.db_conn = conn
    return conn


def close_db() -> None:
    conn = g.pop("db_conn", None)
    if conn is not None:
        conn.close()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS studies (
                id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                is_public INTEGER DEFAULT 0,
                original_dcm_path TEXT,
                roi_mask_path TEXT,
                roi_overlay_png_path TEXT,
                watermarked_png_path TEXT,
                watermarked_dcm_path TEXT,
                robust_payload_text TEXT,
                robust_payload_hex TEXT,
                robust_verified INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(owner_user_id) REFERENCES users(id)
            )
            """
        )
        _ensure_column(cur, "studies", "is_public", "INTEGER DEFAULT 0")
        conn.commit()
    finally:
        conn.close()


def _ensure_column(cur: sqlite3.Cursor, table_name: str, column_name: str, column_def: str) -> None:
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {r[1] for r in rows}
    if column_name not in existing:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def create_user(name: str, email: str, password_hash: str, role: str) -> dict[str, Any]:
    conn = get_db()
    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    conn.execute(
        """
        INSERT INTO users (id, name, email, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, name, email.lower(), password_hash, role, now),
    )
    conn.commit()
    return {
        "id": user_id,
        "name": name,
        "email": email.lower(),
        "role": role,
        "created_at": now,
    }


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    return row


def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    conn = get_db()
    row = conn.execute(
        "SELECT id, name, email, role, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return row


def create_study(owner_user_id: str, original_filename: str, original_dcm_path: str) -> dict[str, Any]:
    conn = get_db()
    study_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    conn.execute(
        """
        INSERT INTO studies (
            id, owner_user_id, original_filename, is_public, original_dcm_path,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (study_id, owner_user_id, original_filename, 0, original_dcm_path, now, now),
    )
    conn.commit()

    return {
        "id": study_id,
        "owner_user_id": owner_user_id,
        "original_filename": original_filename,
        "is_public": 0,
        "original_dcm_path": original_dcm_path,
        "created_at": now,
        "updated_at": now,
    }


def get_study(study_id: str) -> Optional[dict[str, Any]]:
    conn = get_db()
    return conn.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()


def update_study_fields(study_id: str, fields: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not fields:
        return get_study(study_id)

    conn = get_db()
    payload = {**fields, "updated_at": datetime.utcnow().isoformat()}

    set_clause = ", ".join(f"{key} = ?" for key in payload.keys())
    values = list(payload.values()) + [study_id]
    conn.execute(f"UPDATE studies SET {set_clause} WHERE id = ?", values)
    conn.commit()
    return get_study(study_id)


def list_studies_for_user(user_id: str, role: str, scope: str = "mine") -> list[dict[str, Any]]:
    conn = get_db()
    scope = (scope or "mine").lower()

    if role == "ADMIN" and scope == "all":
        rows = conn.execute("SELECT * FROM studies ORDER BY created_at DESC").fetchall()
    elif scope == "public":
        rows = conn.execute(
            "SELECT * FROM studies WHERE is_public = 1 ORDER BY created_at DESC"
        ).fetchall()
    elif scope == "mixed":
        rows = conn.execute(
            "SELECT * FROM studies WHERE owner_user_id = ? OR is_public = 1 ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM studies WHERE owner_user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return rows


@db_api.route("/health", methods=["GET"])
def db_health() -> tuple[Any, int]:
    try:
        conn = get_db()
        conn.execute("SELECT 1").fetchone()
        return jsonify({"status": "ok", "service": "database_api"}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
