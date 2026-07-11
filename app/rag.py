import hashlib
import json
import math
import re
from dataclasses import dataclass

from app import repositories as repo


EMBEDDING_DIMENSIONS = 256


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
    for index, chunk in enumerate(chunks):
        repo.add_document_chunk(
            db,
            document["id"],
            index,
            chunk,
            serialize_embedding(embed_text(chunk)),
        )
    repo.update_document_chunk_count(db, document["id"], len(chunks))
    document["chunk_count"] = len(chunks)
    return document


def retrieve_sources(db, query: str, limit: int = 4) -> list[SourceChunk]:
    query_embedding = embed_text(query)
    ranked: list[SourceChunk] = []
    for chunk in repo.list_document_chunks(db):
        score = cosine_similarity(query_embedding, deserialize_embedding(chunk["embedding"]))
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
