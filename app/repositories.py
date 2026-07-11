import sqlite3


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


def create_session(db: sqlite3.Connection, title: str = "New chat") -> dict:
    cursor = db.execute(
        "INSERT INTO sessions(title) VALUES(?) RETURNING *",
        (title.strip() or "New chat",),
    )
    return row_to_dict(cursor.fetchone())


def list_sessions(db: sqlite3.Connection) -> list[dict]:
    cursor = db.execute(
        """
        SELECT * FROM sessions
        ORDER BY datetime(updated_at) DESC, id DESC
        """
    )
    return [row_to_dict(row) for row in cursor.fetchall()]


def get_session(db: sqlite3.Connection, session_id: int) -> dict | None:
    cursor = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    return row_to_dict(row) if row else None


def update_session_title(db: sqlite3.Connection, session_id: int, title: str) -> None:
    db.execute(
        """
        UPDATE sessions
        SET title = ?, updated_at = datetime('now')
        WHERE id = ?
        """,
        (title[:120], session_id),
    )


def touch_session(db: sqlite3.Connection, session_id: int) -> None:
    db.execute(
        "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
        (session_id,),
    )


def add_message(
    db: sqlite3.Connection,
    session_id: int,
    role: str,
    content: str,
) -> dict:
    cursor = db.execute(
        """
        INSERT INTO messages(session_id, role, content)
        VALUES(?, ?, ?)
        RETURNING *
        """,
        (session_id, role, content),
    )
    touch_session(db, session_id)
    return row_to_dict(cursor.fetchone())


def list_messages(db: sqlite3.Connection, session_id: int) -> list[dict]:
    cursor = db.execute(
        """
        SELECT * FROM messages
        WHERE session_id = ?
        ORDER BY datetime(created_at) ASC, id ASC
        """,
        (session_id,),
    )
    return [row_to_dict(row) for row in cursor.fetchall()]


def recent_messages(
    db: sqlite3.Connection,
    session_id: int,
    limit: int,
) -> list[dict]:
    cursor = db.execute(
        """
        SELECT * FROM (
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
        )
        ORDER BY datetime(created_at) ASC, id ASC
        """,
        (session_id, limit),
    )
    return [row_to_dict(row) for row in cursor.fetchall()]


def get_setting(db: sqlite3.Connection, key: str) -> str | None:
    cursor = db.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row["value"] if row else None


def set_setting(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute(
        """
        INSERT INTO settings(key, value, updated_at)
        VALUES(?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (key, value),
    )
