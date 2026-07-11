import json
import re
from collections.abc import AsyncIterator

from app import repositories as repo
from app.database import get_db
from app.llm import LLMError, OpenAICompatibleClient
from app.rag import build_rag_system_prompt, retrieve_sources
from app.runtime_config import get_effective_settings


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


def validate_citations(assistant_text: str, sources: list[dict]) -> dict:
    available = [source["source_id"] for source in sources]
    available_set = set(available)
    cited = sorted(set(re.findall(r"\[(S\d+)\]", assistant_text)))
    supported = [source_id for source_id in cited if source_id in available_set]
    unsupported = [source_id for source_id in cited if source_id not in available_set]
    missing = [source_id for source_id in available if source_id not in cited]

    if unsupported:
        status = "unsupported"
        message = "Some citations do not match retrieved sources."
    elif available and not cited:
        status = "missing"
        message = "Retrieved sources were available, but the answer did not cite them."
    elif available and supported:
        status = "valid"
        message = "Citations match retrieved sources."
    else:
        status = "not_applicable"
        message = "No retrieved sources were available for citation checking."

    return {
        "status": status,
        "message": message,
        "available_source_ids": available,
        "cited_source_ids": cited,
        "supported_source_ids": supported,
        "unsupported_source_ids": unsupported,
        "missing_source_ids": missing,
    }


async def stream_chat_response(
    session_id: int | None,
    user_message: str,
    use_rag: bool = False,
) -> AsyncIterator[str]:
    settings = get_effective_settings()
    public_sources = []

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
        if use_rag:
            sources = retrieve_sources(db, user_message)
            public_sources = [
                source.to_public_dict(index)
                for index, source in enumerate(sources, start=1)
            ]
            system_prompt = build_rag_system_prompt(system_prompt, sources)
        history = repo.recent_messages(
            db,
            session_id,
            settings.max_history_messages,
        )

    yield encode_sse("session", {"session_id": session_id})
    if use_rag:
        yield encode_sse("sources", {"sources": public_sources})

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

    if use_rag:
        yield encode_sse(
            "citation_check",
            validate_citations(assistant_text, public_sources),
        )

    yield encode_sse(
        "done",
        {"session_id": session_id, "content": assistant_text},
    )
