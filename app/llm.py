from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

from app.config import Settings


class LLMError(RuntimeError):
    pass


@dataclass
class ChatMetrics:
    content: str
    latency_ms: float
    first_token_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(message.get("content", "")) for message in messages)


class OpenAICompatibleClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key or "missing-api-key",
            base_url=settings.openai_base_url,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> AsyncIterator[str]:
        if not self.settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not configured.")

        try:
            stream = await self.client.chat.completions.create(
                model=model or self.settings.openai_model,
                messages=messages,
                temperature=self.settings.temperature,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except APIStatusError as exc:
            message = exc.response.text or str(exc)
            raise LLMError(f"Model provider returned HTTP {exc.status_code}: {message}") from exc
        except APIConnectionError as exc:
            raise LLMError(f"Could not connect to model provider: {exc}") from exc
        except OpenAIError as exc:
            raise LLMError(f"Model provider error: {exc}") from exc

    async def stream_chat_with_metrics(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> ChatMetrics:
        if not self.settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not configured.")

        started_at = perf_counter()
        first_token_at: float | None = None
        content = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        try:
            try:
                stream = await self.client.chat.completions.create(
                    model=model or self.settings.openai_model,
                    messages=messages,
                    temperature=self.settings.temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except APIStatusError as exc:
                message = exc.response.text or str(exc)
                if exc.status_code not in {400, 422} or "stream_options" not in message:
                    raise
                stream = await self.client.chat.completions.create(
                    model=model or self.settings.openai_model,
                    messages=messages,
                    temperature=self.settings.temperature,
                    stream=True,
                )

            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage:
                    prompt_tokens = usage.prompt_tokens or 0
                    completion_tokens = usage.completion_tokens or 0
                    total_tokens = usage.total_tokens or 0

                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    if first_token_at is None:
                        first_token_at = perf_counter()
                    content += delta
        except APIStatusError as exc:
            message = exc.response.text or str(exc)
            raise LLMError(f"Model provider returned HTTP {exc.status_code}: {message}") from exc
        except APIConnectionError as exc:
            raise LLMError(f"Could not connect to model provider: {exc}") from exc
        except OpenAIError as exc:
            raise LLMError(f"Model provider error: {exc}") from exc

        latency_ms = (perf_counter() - started_at) * 1000
        if total_tokens == 0:
            prompt_tokens = estimate_message_tokens(messages)
            completion_tokens = estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

        return ChatMetrics(
            content=content,
            latency_ms=latency_ms,
            first_token_ms=(first_token_at - started_at) * 1000
            if first_token_at
            else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
