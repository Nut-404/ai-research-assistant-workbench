from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class SessionOut(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    created_at: str


class ChatRequest(BaseModel):
    session_id: int | None = None
    message: str = Field(min_length=1, max_length=12000)
    use_rag: bool = False


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=500_000)


class DocumentOut(BaseModel):
    id: int
    title: str
    chunk_count: int
    created_at: str
    updated_at: str


class RetrievedSourceOut(BaseModel):
    source_id: str
    document_id: int
    document_title: str
    chunk_index: int
    score: float
    excerpt: str


class RetrievalPreviewRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=4, ge=1, le=10)


class RetrievalPreviewOut(BaseModel):
    sources: list[RetrievedSourceOut]


class EvaluationRequest(BaseModel):
    models: list[str] = Field(default_factory=list, max_length=6)
    prompt: str = Field(default="", max_length=4000)


class ModelConfigUpdate(BaseModel):
    openai_model: str = Field(min_length=1, max_length=160)
    openai_base_url: str = Field(min_length=1, max_length=500)
    temperature: float = Field(ge=0, le=2)
    embedding_provider: str = Field(min_length=1, max_length=80)
    embedding_model: str = Field(min_length=1, max_length=160)


class ModelConfigOut(ModelConfigUpdate):
    api_key_configured: bool


class EvaluationResultOut(BaseModel):
    id: int
    run_id: int
    model: str
    latency_ms: float | None
    first_token_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    quality_score: float
    consistency_score: float
    retrieval_hit_rate: float
    citation_accuracy: float
    faithfulness_score: float
    benchmark_case_count: int
    output: str
    error: str | None
    created_at: str


class EvaluationRunOut(BaseModel):
    id: int
    prompt: str
    models: str
    created_at: str
    results: list[EvaluationResultOut] = Field(default_factory=list)


class PromptUpdate(BaseModel):
    system_prompt: str = Field(min_length=1, max_length=12000)


class PromptOut(BaseModel):
    system_prompt: str


class HealthOut(BaseModel):
    status: str
    model: str
    base_url: str
    api_key_configured: bool
    embedding_provider: str
    embedding_model: str
