from contextlib import asynccontextmanager
from collections import defaultdict, deque
from pathlib import Path
from time import monotonic

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import repositories as repo
from app.chat import stream_chat_response
from app.config import get_settings
from app.database import get_db, init_db
from app.evaluation import DEFAULT_EVALUATION_PROMPT, EvaluationRunner
from app.rag import EmbeddingError, resolve_embedding_backend, retrieve_sources, store_document
from app.runtime_config import (
    get_effective_settings,
    model_config_dict,
    save_model_config,
)
from app.schemas import (
    ChatRequest,
    DocumentCreate,
    DocumentOut,
    EvaluationRequest,
    EvaluationRunOut,
    HealthOut,
    ModelConfigOut,
    ModelConfigUpdate,
    MessageOut,
    PromptOut,
    PromptUpdate,
    RetrievalPreviewOut,
    RetrievalPreviewRequest,
    SessionCreate,
    SessionOut,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def parse_allowed_origins(value: str) -> list[str]:
    origins = [origin.strip() for origin in value.split(",") if origin.strip()]
    return origins or ["http://127.0.0.1:8000", "http://localhost:8000"]


def is_authorized(request: Request) -> bool:
    token = get_settings().api_token
    if not token or request.url.path in {"/api/health"}:
        return True
    auth_header = request.headers.get("authorization", "")
    api_key = request.headers.get("x-api-key", "")
    return api_key == token or auth_header == f"Bearer {token}"


def is_rate_limited(request: Request) -> bool:
    settings = get_settings()
    if (
        settings.rate_limit_per_minute <= 0
        or not request.url.path.startswith("/api/")
        or request.url.path == "/api/health"
    ):
        return False

    forwarded_for = request.headers.get("x-forwarded-for", "")
    client_host = forwarded_for.split(",")[0].strip()
    if not client_host and request.client:
        client_host = request.client.host
    key = f"{client_host or 'unknown'}:{request.url.path}"
    now = monotonic()
    window_started = now - 60
    bucket = RATE_LIMIT_BUCKETS[key]
    while bucket and bucket[0] < window_started:
        bucket.popleft()
    if len(bucket) >= settings.rate_limit_per_minute:
        return True
    bucket.append(now)
    return False


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(get_settings().allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        if not is_authorized(request):
            return JSONResponse(
                {"detail": "A valid API token is required."},
                status_code=401,
            )
        if is_rate_limited(request):
            return JSONResponse(
                {"detail": "Rate limit exceeded. Please wait before retrying."},
                status_code=429,
            )
    return await call_next(request)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health", response_model=HealthOut)
def health() -> HealthOut:
    settings = get_effective_settings()
    api_key_configured = bool(settings.openai_api_key)
    embedding_provider, embedding_model = resolve_embedding_backend(settings)
    return HealthOut(
        status="ok" if api_key_configured else "degraded",
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        api_key_configured=api_key_configured,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )


@app.get("/api/sessions", response_model=list[SessionOut])
def sessions() -> list[dict]:
    with get_db() as db:
        return repo.list_sessions(db)


@app.post("/api/sessions", response_model=SessionOut)
def create_session(payload: SessionCreate) -> dict:
    with get_db() as db:
        return repo.create_session(db, payload.title or "New chat")


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageOut])
def messages(session_id: int) -> list[dict]:
    with get_db() as db:
        if repo.get_session(db, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return repo.list_messages(db, session_id)


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_chat_response(payload.session_id, payload.message, payload.use_rag),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/prompt", response_model=PromptOut)
def get_prompt() -> PromptOut:
    settings = get_settings()
    with get_db() as db:
        system_prompt = (
            repo.get_setting(db, "system_prompt") or settings.default_system_prompt
        )
    return PromptOut(system_prompt=system_prompt)


@app.put("/api/prompt", response_model=PromptOut)
def update_prompt(payload: PromptUpdate) -> PromptOut:
    with get_db() as db:
        repo.set_setting(db, "system_prompt", payload.system_prompt)
    return PromptOut(system_prompt=payload.system_prompt)


@app.get("/api/documents", response_model=list[DocumentOut])
def documents() -> list[dict]:
    with get_db() as db:
        return repo.list_documents(db)


@app.post("/api/documents", response_model=DocumentOut)
def create_document(payload: DocumentCreate) -> dict:
    with get_db() as db:
        try:
            return store_document(db, payload.title, payload.content)
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: int) -> Response:
    with get_db() as db:
        if not repo.delete_document(db, document_id):
            raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=204)


@app.post("/api/retrieval/preview", response_model=RetrievalPreviewOut)
def retrieval_preview(payload: RetrievalPreviewRequest) -> RetrievalPreviewOut:
    with get_db() as db:
        sources = retrieve_sources(db, payload.query, payload.limit)
    return RetrievalPreviewOut(
        sources=[
            source.to_public_dict(index)
            for index, source in enumerate(sources, start=1)
        ]
    )


@app.get("/api/evaluations", response_model=list[EvaluationRunOut])
def evaluation_runs() -> list[dict]:
    with get_db() as db:
        runs = repo.list_evaluation_runs(db)
        for run in runs:
            run["results"] = repo.list_evaluation_results(db, run["id"])
        return runs


@app.get("/api/model-config", response_model=ModelConfigOut)
def get_model_config() -> dict:
    return model_config_dict(get_effective_settings())


@app.put("/api/model-config", response_model=ModelConfigOut)
def update_model_config(payload: ModelConfigUpdate) -> dict:
    with get_db() as db:
        save_model_config(db, payload)
    return model_config_dict(get_effective_settings())


@app.post("/api/evaluations", response_model=EvaluationRunOut)
async def create_evaluation(payload: EvaluationRequest) -> dict:
    settings = get_effective_settings()
    prompt = payload.prompt.strip() or DEFAULT_EVALUATION_PROMPT
    runner = EvaluationRunner(settings)
    return await runner.run(payload.models, prompt)
