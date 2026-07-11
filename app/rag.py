import hashlib
import json
import math
import re
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError

from app import repositories as repo
from app.config import Settings, get_settings


EMBEDDING_DIMENSIONS = 256
LOCAL_EMBEDDING_PROVIDER = "local-hashing"
LOCAL_EMBEDDING_MODEL = "hashing-256"


class EmbeddingError(RuntimeError):
    pass


@dataclass
class SourceChunk:
    id: int
    document_id: int
    document_title: str
    chunk_index: int
    content: str
    score: float

    def to_public_dict(self, index: int) -> dict:
        return {
            "source_id": f"S{index}",
            "document_id": self.document_id,
            "document_title": self.document_title,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "excerpt": self.content[:420],
        }


@dataclass
class TextCandidate:
    id: str
    title: str
    content: str


@dataclass
class RankedTextCandidate:
    id: str
    title: str
    content: str
    score: float


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    return re.findall(r"[\w\u4e00-\u9fff]+", normalized)


def embed_text(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
        vector[bucket] += 1.0

    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def serialize_embedding(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"))


def deserialize_embedding(value: str) -> list[float]:
    return [float(item) for item in json.loads(value)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def resolve_embedding_backend(settings: Settings) -> tuple[str, str]:
    provider = settings.embedding_provider.strip().lower()
    if provider in {"", "auto"}:
        if settings.openai_api_key:
            return "openai", settings.embedding_model
        return LOCAL_EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL
    if provider in {"local", "hashing", LOCAL_EMBEDDING_PROVIDER}:
        return LOCAL_EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL
    return provider, settings.embedding_model


class EmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider, self.model = resolve_embedding_backend(self.settings)
        self._client: OpenAI | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_with_backend(texts, self.provider, self.model)

    def embed_query_for(self, text: str, provider: str, model: str) -> list[float]:
        return self._embed_with_backend([text], provider, model)[0]

    def _embed_with_backend(
        self,
        texts: list[str],
        provider: str,
        model: str,
    ) -> list[list[float]]:
        if provider == LOCAL_EMBEDDING_PROVIDER:
            return [embed_text(text) for text in texts]
        if provider == "openai":
            return self._embed_with_openai(texts, model)
        raise EmbeddingError(f"Unsupported embedding provider: {provider}")

    def _embed_with_openai(self, texts: list[str], model: str) -> list[list[float]]:
        if not self.settings.openai_api_key:
            raise EmbeddingError("OPENAI_API_KEY is required for semantic embeddings.")
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        try:
            response = self._client.embeddings.create(model=model, input=texts)
        except APIStatusError as exc:
            message = exc.response.text or str(exc)
            raise EmbeddingError(
                f"Embedding provider returned HTTP {exc.status_code}: {message}"
            ) from exc
        except APIConnectionError as exc:
            raise EmbeddingError(f"Could not connect to embedding provider: {exc}") from exc
        except OpenAIError as exc:
            raise EmbeddingError(f"Embedding provider error: {exc}") from exc
        return [normalize_vector(item.embedding) for item in response.data]


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines())
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    if not paragraphs:
        paragraphs = [cleaned.strip()] if cleaned.strip() else []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + max_chars].strip())
                start += max_chars - overlap
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def store_document(db, title: str, content: str) -> dict:
    chunks = chunk_text(content)
    document = repo.create_document(db, title, content)
    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_documents(chunks) if chunks else []
    for index, chunk in enumerate(chunks):
        repo.add_document_chunk(
            db,
            document["id"],
            index,
            chunk,
            serialize_embedding(embeddings[index]),
            embedding_service.provider,
            embedding_service.model,
        )
    repo.update_document_chunk_count(db, document["id"], len(chunks))
    document["chunk_count"] = len(chunks)
    return document


def retrieve_sources(db, query: str, limit: int = 4) -> list[SourceChunk]:
    embedding_service = EmbeddingService()
    query_embeddings: dict[tuple[str, str], list[float]] = {}
    ranked: list[SourceChunk] = []
    for chunk in repo.list_document_chunks(db):
        provider = chunk["embedding_provider"]
        model = chunk["embedding_model"]
        key = (provider, model)
        if key not in query_embeddings:
            try:
                query_embeddings[key] = embedding_service.embed_query_for(
                    query,
                    provider,
                    model,
                )
            except EmbeddingError:
                continue

        score = cosine_similarity(
            query_embeddings[key],
            deserialize_embedding(chunk["embedding"]),
        )
        if score <= 0:
            continue
        ranked.append(
            SourceChunk(
                id=chunk["id"],
                document_id=chunk["document_id"],
                document_title=chunk["document_title"],
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                score=score,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:limit]


def rank_text_candidates(
    query: str,
    candidates: list[TextCandidate],
    limit: int = 4,
) -> list[RankedTextCandidate]:
    embedding_service = EmbeddingService()
    query_embedding = embedding_service.embed_query_for(
        query,
        embedding_service.provider,
        embedding_service.model,
    )
    candidate_embeddings = embedding_service.embed_documents(
        [candidate.content for candidate in candidates]
    )
    ranked = []
    for candidate, embedding in zip(candidates, candidate_embeddings, strict=True):
        score = cosine_similarity(query_embedding, embedding)
        if score <= 0:
            continue
        ranked.append(
            RankedTextCandidate(
                id=candidate.id,
                title=candidate.title,
                content=candidate.content,
                score=score,
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:limit]


def build_context_block(sources: list[SourceChunk]) -> str:
    if not sources:
        return "No relevant knowledge base excerpts were retrieved."

    blocks = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[S{index}] {source.document_title} chunk {source.chunk_index}",
                    source.content,
                ]
            )
        )
    return "\n\n".join(blocks)


def build_rag_system_prompt(base_prompt: str, sources: list[SourceChunk]) -> str:
    context = build_context_block(sources)
    return (
        f"{base_prompt}\n\n"
        "Use the retrieved knowledge base context when it is relevant. "
        "Cite supporting excerpts with source ids such as [S1]. "
        "If the context is insufficient, say what is missing instead of inventing facts.\n\n"
        f"Retrieved context:\n{context}"
    )
