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


class PromptUpdate(BaseModel):
    system_prompt: str = Field(min_length=1, max_length=12000)


class PromptOut(BaseModel):
    system_prompt: str


class HealthOut(BaseModel):
    status: str
    model: str
    base_url: str
