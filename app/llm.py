from collections.abc import AsyncIterator

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

from app.config import Settings


class LLMError(RuntimeError):
    pass


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
    ) -> AsyncIterator[str]:
        if not self.settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not configured.")

        try:
            stream = await self.client.chat.completions.create(
                model=self.settings.openai_model,
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
