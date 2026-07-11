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


def create_document(db: sqlite3.Connection, title: str, content: str) -> dict:
    cursor = db.execute(
        """
        INSERT INTO documents(title, content)
        VALUES(?, ?)
        RETURNING *
        """,
        (title.strip() or "Untitled document", content),
    )
    return row_to_dict(cursor.fetchone())


def update_document_chunk_count(
    db: sqlite3.Connection,
    document_id: int,
    chunk_count: int,
) -> None:
    db.execute(
        """
        UPDATE documents
        SET chunk_count = ?, updated_at = datetime('now')
        WHERE id = ?
        """,
        (chunk_count, document_id),
    )


def list_documents(db: sqlite3.Connection) -> list[dict]:
    cursor = db.execute(
        """
        SELECT id, title, chunk_count, created_at, updated_at
        FROM documents
        ORDER BY datetime(updated_at) DESC, id DESC
        """
    )
    return [row_to_dict(row) for row in cursor.fetchall()]


def add_document_chunk(
    db: sqlite3.Connection,
    document_id: int,
    chunk_index: int,
    content: str,
    embedding: str,
) -> dict:
    cursor = db.execute(
        """
        INSERT INTO document_chunks(document_id, chunk_index, content, embedding)
        VALUES(?, ?, ?, ?)
        RETURNING *
        """,
        (document_id, chunk_index, content, embedding),
    )
    return row_to_dict(cursor.fetchone())


def list_document_chunks(db: sqlite3.Connection) -> list[dict]:
    cursor = db.execute(
        """
        SELECT
            document_chunks.*,
            documents.title AS document_title
        FROM document_chunks
        JOIN documents ON documents.id = document_chunks.document_id
        ORDER BY document_chunks.document_id ASC, document_chunks.chunk_index ASC
        """
    )
    return [row_to_dict(row) for row in cursor.fetchall()]


def create_evaluation_run(
    db: sqlite3.Connection,
    prompt: str,
    models: str,
) -> dict:
    cursor = db.execute(
        """
        INSERT INTO evaluation_runs(prompt, models)
        VALUES(?, ?)
        RETURNING *
        """,
        (prompt, models),
    )
    return row_to_dict(cursor.fetchone())


def add_evaluation_result(
    db: sqlite3.Connection,
    run_id: int,
    result: dict,
) -> dict:
    cursor = db.execute(
        """
        INSERT INTO evaluation_results(
            run_id,
            model,
            latency_ms,
            first_token_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            quality_score,
            consistency_score,
            output,
            error
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING *
        """,
        (
            run_id,
            result["model"],
            result.get("latency_ms"),
            result.get("first_token_ms"),
            result.get("prompt_tokens", 0),
            result.get("completion_tokens", 0),
            result.get("total_tokens", 0),
            result.get("quality_score", 0),
            result.get("consistency_score", 0),
            result.get("output", ""),
            result.get("error"),
        ),
    )
    return row_to_dict(cursor.fetchone())


def list_evaluation_runs(db: sqlite3.Connection, limit: int = 10) -> list[dict]:
    cursor = db.execute(
        """
        SELECT * FROM evaluation_runs
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [row_to_dict(row) for row in cursor.fetchall()]


def list_evaluation_results(db: sqlite3.Connection, run_id: int) -> list[dict]:
    cursor = db.execute(
        """
        SELECT * FROM evaluation_results
        WHERE run_id = ?
        ORDER BY id ASC
        """,
        (run_id,),
    )
    return [row_to_dict(row) for row in cursor.fetchall()]
