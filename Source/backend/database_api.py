from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint, current_app, g, jsonify
from werkzeug.security import generate_password_hash


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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS study_shares (
                id TEXT PRIMARY KEY,
                study_id TEXT NOT NULL,
                recipient_user_id TEXT NOT NULL,
                shared_by_user_id TEXT NOT NULL,
                shared_at TEXT NOT NULL,
                UNIQUE(study_id, recipient_user_id),
                FOREIGN KEY(study_id) REFERENCES studies(id),
                FOREIGN KEY(recipient_user_id) REFERENCES users(id),
                FOREIGN KEY(shared_by_user_id) REFERENCES users(id)
            )
            """
        )
        _ensure_column(cur, "studies", "is_public", "INTEGER DEFAULT 0")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                actor_user_id TEXT,
                actor_email TEXT,
                actor_role TEXT,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                outcome TEXT NOT NULL DEFAULT 'SUCCESS',
                detail TEXT
            )
            """
        )
        _ensure_column(cur, "users", "username", "TEXT")
        _seed_auditor_user(conn, cur)
        conn.commit()
        
    finally:
        conn.close()


def _ensure_column(cur: sqlite3.Cursor, table_name: str, column_name: str, column_def: str) -> None:
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {r[1] for r in rows}
    if column_name not in existing:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def _seed_auditor_user(conn: sqlite3.Connection, cur: sqlite3.Cursor) -> None:
    """Ensure the default AUDITOR account exists with known credentials."""
    now = datetime.utcnow().isoformat()
    pw_hash = generate_password_hash("auditor1")

    existing = cur.execute(
        """
        SELECT id
        FROM users
        WHERE email = ? OR username = ?
        LIMIT 1
        """,
        ("auditor@system.local", "auditor"),
    ).fetchone()

    if existing:
        cur.execute(
            """
            UPDATE users
            SET name = ?,
                email = ?,
                username = ?,
                password_hash = ?,
                role = ?
            WHERE id = ?
            """,
            ("Auditor", "auditor@system.local", "auditor", pw_hash, "AUDITOR", existing[0]),
        )
        conn.commit()
        return

    user_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO users (id, name, email, username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, "Auditor", "auditor@system.local", "auditor", pw_hash, "AUDITOR", now),
    )
    conn.commit()


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


def delete_study(study_id: str) -> bool:
    conn = get_db()
    conn.execute("DELETE FROM study_shares WHERE study_id = ?", (study_id,))
    cur = conn.execute("DELETE FROM studies WHERE id = ?", (study_id,))
    conn.commit()
    return cur.rowcount > 0


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
    elif scope == "shared":
        rows = conn.execute(
            """
            SELECT s.*
            FROM studies s
            INNER JOIN study_shares ss ON ss.study_id = s.id
            WHERE ss.recipient_user_id = ?
            ORDER BY ss.shared_at DESC
            """,
            (user_id,),
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


def is_study_shared_with_user(study_id: str, user_id: str) -> bool:
    conn = get_db()
    row = conn.execute(
        """
        SELECT 1
        FROM study_shares
        WHERE study_id = ? AND recipient_user_id = ?
        LIMIT 1
        """,
        (study_id, user_id),
    ).fetchone()
    return bool(row)


def share_study_with_user(study_id: str, recipient_user_id: str, shared_by_user_id: str) -> bool:
    conn = get_db()
    share_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO study_shares (
            id, study_id, recipient_user_id, shared_by_user_id, shared_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (share_id, study_id, recipient_user_id, shared_by_user_id, now),
    )
    conn.commit()
    return cur.rowcount > 0


def get_user_by_username(username: str) -> Optional[dict[str, Any]]:
    conn = get_db()
    return conn.execute(
        "SELECT * FROM users WHERE username = ?", (username.lower(),)
    ).fetchone()


def create_audit_log_entry(
    actor_user_id: Optional[str],
    actor_email: Optional[str],
    actor_role: Optional[str],
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    outcome: str = "SUCCESS",
    detail: Optional[str] = None,
) -> str:
    conn = get_db()
    log_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO audit_logs (
            id, timestamp, actor_user_id, actor_email, actor_role,
            action, resource_type, resource_id, ip_address, user_agent,
            outcome, detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            log_id, now, actor_user_id, actor_email, actor_role,
            action, resource_type, resource_id, ip_address, user_agent,
            outcome, detail,
        ),
    )
    conn.commit()
    return log_id


def query_audit_logs(
    limit: int = 100,
    offset: int = 0,
    actor_user_id: Optional[str] = None,
    action: Optional[str] = None,
    outcome: Optional[str] = None,
    resource_id: Optional[str] = None,
    from_ts: Optional[str] = None,
    to_ts: Optional[str] = None,
) -> tuple[list[dict[str, Any]], int]:
    conn = get_db()
    conditions: list[str] = []
    params: list[Any] = []

    if actor_user_id:
        conditions.append("actor_user_id = ?")
        params.append(actor_user_id)
    if action:
        conditions.append("action = ?")
        params.append(action.upper())
    if outcome:
        conditions.append("outcome = ?")
        params.append(outcome.upper())
    if resource_id:
        conditions.append("resource_id = ?")
        params.append(resource_id)
    if from_ts:
        conditions.append("timestamp >= ?")
        params.append(from_ts)
    if to_ts:
        conditions.append("timestamp <= ?")
        params.append(to_ts)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    count_row = conn.execute(
        f"SELECT COUNT(*) AS cnt FROM audit_logs {where}", params
    ).fetchone()
    total = count_row["cnt"] if count_row else 0
    rows = conn.execute(
        f"SELECT * FROM audit_logs {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return rows, total


@db_api.route("/health", methods=["GET"])
def db_health() -> tuple[Any, int]:
    try:
        conn = get_db()
        conn.execute("SELECT 1").fetchone()
        return jsonify({"status": "ok", "service": "database_api"}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
