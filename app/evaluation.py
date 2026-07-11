import json
import re
from dataclasses import asdict

from app import repositories as repo
from app.config import Settings
from app.database import get_db
from app.llm import LLMError, OpenAICompatibleClient


DEFAULT_EVALUATION_PROMPT = (
    "A student is applying to an AI graduate program and wants to build a useful "
    "AI assistant project. Propose a concise, practical plan with tradeoffs."
)


def normalize_model_list(models: list[str], fallback: str) -> list[str]:
    cleaned = []
    for model in models:
        value = " ".join(model.strip().split())
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned or [fallback]


def score_output_quality(prompt: str, output: str) -> float:
    if not output.strip():
        return 0.0

    prompt_terms = {
        term
        for term in re.findall(r"[a-zA-Z]{4,}", prompt.lower())
        if term not in {"that", "with", "from", "this", "will"}
    }
    output_lower = output.lower()
    overlap = sum(1 for term in prompt_terms if term in output_lower)
    coverage = overlap / max(len(prompt_terms), 1)

    structure = min(output.count("\n") / 4, 1.0)
    length_fit = min(len(output) / 900, 1.0)
    refusal_penalty = 0.25 if "i can't" in output_lower or "cannot" in output_lower else 0

    score = (coverage * 0.45) + (structure * 0.25) + (length_fit * 0.30)
    return round(max(0.0, min(1.0, score - refusal_penalty)), 3)


def score_consistency(output: str) -> float:
    normalized = output.lower()
    has_time = "2" in normalized or "two" in normalized
    has_week = "week" in normalized or "weekly" in normalized
    has_constraint = "constraint" in normalized or "limit" in normalized
    return round((has_time + has_week + has_constraint) / 3, 3)


class EvaluationRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAICompatibleClient(settings)

    async def run(self, models: list[str], prompt: str) -> dict:
        selected_models = normalize_model_list(models, self.settings.openai_model)
        with get_db() as db:
            run = repo.create_evaluation_run(
                db,
                prompt,
                json.dumps(selected_models),
            )

        results = []
        for model in selected_models:
            result = await self.evaluate_model(model, prompt)
            with get_db() as db:
                saved = repo.add_evaluation_result(db, run["id"], result)
            results.append(saved)

        run["results"] = results
        return run

    async def evaluate_model(self, model: str, prompt: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": "Answer clearly, concretely, and with practical tradeoffs.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            metrics = await self.client.stream_chat_with_metrics(messages, model=model)
            consistency = await self.evaluate_consistency(model)
            return {
                "model": model,
                **asdict(metrics),
                "quality_score": score_output_quality(prompt, metrics.content),
                "consistency_score": consistency,
                "output": metrics.content,
                "error": None,
            }
        except LLMError as exc:
            return {
                "model": model,
                "latency_ms": None,
                "first_token_ms": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "quality_score": 0,
                "consistency_score": 0,
                "output": "",
                "error": str(exc),
            }

    async def evaluate_consistency(self, model: str) -> float:
        messages = [
            {
                "role": "system",
                "content": "Track user constraints accurately across turns.",
            },
            {
                "role": "user",
                "content": "My only constraint is that I can study at most 2 hours per week.",
            },
            {
                "role": "assistant",
                "content": "Understood. I will keep the plan within 2 hours per week.",
            },
            {
                "role": "user",
                "content": "What constraint should the plan respect?",
            },
        ]
        metrics = await self.client.stream_chat_with_metrics(messages, model=model)
        return score_consistency(metrics.content)
