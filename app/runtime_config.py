from app import repositories as repo
from app.config import Settings, get_settings
from app.database import get_db


MODEL_CONFIG_KEYS = (
    "openai_model",
    "openai_base_url",
    "temperature",
    "embedding_provider",
    "embedding_model",
)


def get_effective_settings() -> Settings:
    settings = get_settings()
    with get_db() as db:
        values = repo.get_settings(db, MODEL_CONFIG_KEYS)

    updates: dict[str, str | float] = {}
    for key in MODEL_CONFIG_KEYS:
        value = values.get(key)
        if value is None:
            continue
        updates[key] = float(value) if key == "temperature" else value

    return settings.model_copy(update=updates)


def model_config_dict(settings: Settings) -> dict:
    return {
        "openai_model": settings.openai_model,
        "openai_base_url": settings.openai_base_url,
        "temperature": settings.temperature,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "api_key_configured": bool(settings.openai_api_key),
    }


def save_model_config(db, payload) -> None:
    for key in MODEL_CONFIG_KEYS:
        repo.set_setting(db, key, str(getattr(payload, key)))
