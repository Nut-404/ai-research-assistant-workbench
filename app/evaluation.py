import json
import re
from dataclasses import dataclass

from app import repositories as repo
from app.config import Settings
from app.database import get_db
from app.llm import LLMError, OpenAICompatibleClient
from app.rag import EmbeddingError, TextCandidate, rank_text_candidates


DEFAULT_EVALUATION_PROMPT = (
    "Evaluate how well the assistant answers grounded AI research-workbench "
    "questions using retrieved evidence and explicit citations."
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str
    expected_source_ids: tuple[str, ...]
    reference_terms: tuple[str, ...]


BENCHMARK_SOURCES = [
    TextCandidate(
        id="rag-grounding",
        title="RAG grounding note",
        content=(
            "Retrieval-augmented generation should retrieve relevant document "
            "chunks, expose cited source excerpts, and tell the user when the "
            "retrieved context is insufficient instead of inventing facts."
        ),
    ),
    TextCandidate(
        id="evaluation-metrics",
        title="Model evaluation note",
        content=(
            "A useful model evaluation should compare latency, first-token time, "
            "token usage, citation accuracy, answer faithfulness, retrieval hit "
            "rate, and multi-turn consistency across the same benchmark questions."
        ),
    ),
    TextCandidate(
        id="research-value",
        title="AI system research note",
        content=(
            "A useful AI systems project is stronger when it explains the research "
            "question, experimental design, measurable results, failure cases, "
            "limitations, and future research extensions."
        ),
    ),
]


BENCHMARK_CASES = [
    BenchmarkCase(
        case_id="rag-grounding",
        question=(
            "How should this workbench reduce hallucination when answering from "
            "uploaded documents?"
        ),
        expected_source_ids=("rag-grounding",),
        reference_terms=("retrieve", "chunks", "citations", "insufficient"),
    ),
    BenchmarkCase(
        case_id="model-evaluation",
        question=(
            "Which metrics should be tracked when comparing models in this "
            "research assistant?"
        ),
        expected_source_ids=("evaluation-metrics",),
        reference_terms=(
            "latency",
            "first-token",
            "token",
            "faithfulness",
            "retrieval",
            "consistency",
        ),
    ),
    BenchmarkCase(
        case_id="research-value",
        question=(
            "Why is this project useful for studying AI systems beyond a simple "
            "chat demo?"
        ),
        expected_source_ids=("research-value",),
        reference_terms=("research", "experimental", "results", "limitations"),
    ),
]


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


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def score_retrieval_hit_rate(
    retrieved_source_ids: list[str],
    expected_source_ids: tuple[str, ...],
) -> float:
    if not expected_source_ids:
        return 0.0
    hits = sum(1 for source_id in expected_source_ids if source_id in retrieved_source_ids)
    return round(hits / len(expected_source_ids), 3)


def score_citation_accuracy(
    output: str,
    expected_citation_labels: list[str],
    available_citation_labels: list[str],
) -> float:
    if not expected_citation_labels:
        return 0.0
    cited_labels = set(re.findall(r"\[(S\d+)\]", output))
    expected_hits = sum(1 for label in expected_citation_labels if label in cited_labels)
    unsupported = cited_labels.difference(available_citation_labels)
    score = expected_hits / len(expected_citation_labels)
    if unsupported:
        score -= 0.25
    return round(max(0.0, min(1.0, score)), 3)


def score_faithfulness(output: str, reference_terms: tuple[str, ...]) -> float:
    if not output.strip() or not reference_terms:
        return 0.0
    normalized = output.lower()
    hits = sum(1 for term in reference_terms if term.lower() in normalized)
    return round(hits / len(reference_terms), 3)


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
        benchmark_prompt = prompt.strip() or DEFAULT_EVALUATION_PROMPT
        case_outputs = []
        latencies = []
        first_token_times = []
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        quality_scores = []
        retrieval_scores = []
        citation_scores = []
        faithfulness_scores = []
        try:
            for case in BENCHMARK_CASES:
                retrieved_sources = rank_text_candidates(
                    case.question,
                    BENCHMARK_SOURCES,
                    limit=2,
                )
                retrieved_ids = [source.id for source in retrieved_sources]
                retrieval_scores.append(
                    score_retrieval_hit_rate(retrieved_ids, case.expected_source_ids)
                )
                source_labels = {
                    source.id: f"S{index}"
                    for index, source in enumerate(retrieved_sources, start=1)
                }
                source_block = "\n\n".join(
                    "\n".join(
                        [
                            f"[{source_labels[source.id]}] {source.title}",
                            source.content,
                        ]
                    )
                    for source in retrieved_sources
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are being evaluated as a grounded AI research "
                            "assistant. Answer only from the retrieved excerpts, "
                            "cite sources like [S1], and state what is missing if "
                            "the excerpts are insufficient."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Evaluation goal: {benchmark_prompt}\n\n"
                            f"Question: {case.question}\n\n"
                            f"Retrieved excerpts:\n{source_block}"
                        ),
                    },
                ]
                metrics = await self.client.stream_chat_with_metrics(
                    messages,
                    model=model,
                )
                expected_labels = [
                    source_labels[source_id]
                    for source_id in case.expected_source_ids
                    if source_id in source_labels
                ]
                available_labels = list(source_labels.values())
                quality_scores.append(score_output_quality(case.question, metrics.content))
                citation_scores.append(
                    score_citation_accuracy(
                        metrics.content,
                        expected_labels,
                        available_labels,
                    )
                )
                faithfulness_scores.append(
                    score_faithfulness(metrics.content, case.reference_terms)
                )
                latencies.append(metrics.latency_ms)
                if metrics.first_token_ms is not None:
                    first_token_times.append(metrics.first_token_ms)
                prompt_tokens += metrics.prompt_tokens
                completion_tokens += metrics.completion_tokens
                total_tokens += metrics.total_tokens
                case_outputs.append(f"{case.case_id}: {metrics.content}")

            consistency = await self.evaluate_consistency(model)
            return {
                "model": model,
                "latency_ms": sum(latencies),
                "first_token_ms": average(first_token_times)
                if first_token_times
                else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "quality_score": average(quality_scores),
                "consistency_score": consistency,
                "retrieval_hit_rate": average(retrieval_scores),
                "citation_accuracy": average(citation_scores),
                "faithfulness_score": average(faithfulness_scores),
                "benchmark_case_count": len(BENCHMARK_CASES),
                "output": "\n\n".join(case_outputs),
                "error": None,
            }
        except (EmbeddingError, LLMError) as exc:
            return {
                "model": model,
                "latency_ms": None,
                "first_token_ms": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "quality_score": 0,
                "consistency_score": 0,
                "retrieval_hit_rate": 0,
                "citation_accuracy": 0,
                "faithfulness_score": 0,
                "benchmark_case_count": len(BENCHMARK_CASES),
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
