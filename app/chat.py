import json
from collections.abc import AsyncIterator

from app import repositories as repo
from app.config import get_settings
from app.database import get_db
from app.llm import LLMError, OpenAICompatibleClient


def make_title(message: str) -> str:
    normalized = " ".join(message.strip().split())
    if len(normalized) <= 44:
        return normalized or "New chat"
    return f"{normalized[:44]}..."


def encode_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_model_messages(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> list[dict[str, str]]:
    model_messages = [{"role": "system", "content": system_prompt}]
    for message in history:
        if message["role"] in {"user", "assistant"}:
            model_messages.append(
                {"role": message["role"], "content": message["content"]}
            )
    model_messages.append({"role": "user", "content": user_message})
    return model_messages


async def stream_chat_response(
    session_id: int | None,
    user_message: str,
) -> AsyncIterator[str]:
    settings = get_settings()

    with get_db() as db:
        if session_id is None:
            session = repo.create_session(db, make_title(user_message))
            session_id = session["id"]
        else:
            session = repo.get_session(db, session_id)
            if session is None:
                yield encode_sse(
                    "error",
                    {"message": f"Session {session_id} does not exist."},
                )
                return

        existing_messages = repo.list_messages(db, session_id)
        if not existing_messages:
            repo.update_session_title(db, session_id, make_title(user_message))

        repo.add_message(db, session_id, "user", user_message)
        system_prompt = (
            repo.get_setting(db, "system_prompt") or settings.default_system_prompt
        )
        history = repo.recent_messages(
            db,
            session_id,
            settings.max_history_messages,
        )

    yield encode_sse("session", {"session_id": session_id})

    model_messages = build_model_messages(system_prompt, history[:-1], user_message)
    client = OpenAICompatibleClient(settings)
    assistant_text = ""

    try:
        async for token in client.stream_chat(model_messages):
            assistant_text += token
            yield encode_sse("delta", {"content": token})
    except LLMError as exc:
        yield encode_sse("error", {"message": str(exc)})
        return

    with get_db() as db:
        repo.add_message(db, session_id, "assistant", assistant_text)

    yield encode_sse(
        "done",
        {"session_id": session_id, "content": assistant_text},
    )
