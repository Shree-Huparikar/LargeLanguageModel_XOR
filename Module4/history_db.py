"""SQLite message & thread history store for Module 4."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SQLITE_DB_PATH = Path(__file__).resolve().parent / "module4_support.sqlite"


def init_history_db(db_path: str | Path = DEFAULT_SQLITE_DB_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT DEFAULT '',
                draft_response TEXT DEFAULT '',
                requires_approval INTEGER DEFAULT 0,
                steps_taken_json TEXT DEFAULT '[]',
                turn_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_message(
    thread_id: str,
    role: str,
    content: str,
    *,
    category: str = "",
    draft_response: str = "",
    requires_approval: bool = False,
    steps_taken: list[str] | None = None,
    turn_count: int = 0,
    db_path: str | Path = DEFAULT_SQLITE_DB_PATH,
) -> None:
    init_history_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO workflow_messages (
                thread_id, role, content, category, draft_response,
                requires_approval, steps_taken_json, turn_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                role,
                content,
                category,
                draft_response,
                1 if requires_approval else 0,
                json.dumps(steps_taken or []),
                turn_count,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def get_thread_messages(
    thread_id: str,
    db_path: str | Path = DEFAULT_SQLITE_DB_PATH,
) -> list[dict[str, Any]]:
    init_history_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT role, content, category, draft_response, requires_approval,
                   steps_taken_json, turn_count, created_at
            FROM workflow_messages
            WHERE thread_id = ?
            ORDER BY id ASC
            """,
            (thread_id,),
        ).fetchall()

    return [
        {
            "role": r["role"],
            "content": r["content"],
            "response": r["content"] if r["role"] == "assistant" else "",
            "category": r["category"] or "",
            "draft_response": r["draft_response"] or "",
            "requires_approval": bool(r["requires_approval"]),
            "steps_taken": json.loads(r["steps_taken_json"] or "[]"),
            "turn_count": r["turn_count"] or 0,
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def list_threads(
    limit: int = 25,
    db_path: str | Path = DEFAULT_SQLITE_DB_PATH,
) -> list[dict[str, Any]]:
    init_history_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        ordered_ids = [
            r["thread_id"]
            for r in conn.execute(
                """
                SELECT thread_id
                FROM workflow_messages
                GROUP BY thread_id
                ORDER BY MIN(id) ASC
                """
            ).fetchall()
        ]
        thread_num_map = {tid: idx + 1 for idx, tid in enumerate(ordered_ids)}

        rows = conn.execute(
            """
            SELECT thread_id, COUNT(*) AS message_count, MAX(created_at) AS last_activity
            FROM workflow_messages
            GROUP BY thread_id
            ORDER BY MAX(id) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        result = []
        for r in rows:
            tid = r["thread_id"]
            last_q = conn.execute(
                "SELECT content FROM workflow_messages WHERE thread_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1",
                (tid,),
            ).fetchone()
            last_cat = conn.execute(
                "SELECT category FROM workflow_messages WHERE thread_id = ? AND category != '' ORDER BY id DESC LIMIT 1",
                (tid,),
            ).fetchone()
            num = thread_num_map.get(tid, 1)
            result.append(
                {
                    "thread_id": tid,
                    "thread_num": num,
                    "title": f"Thread {num}",
                    "message_count": r["message_count"],
                    "last_activity": r["last_activity"],
                    "last_question": last_q["content"] if last_q else "",
                    "last_category": last_cat["category"] if last_cat else "",
                }
            )
    return result