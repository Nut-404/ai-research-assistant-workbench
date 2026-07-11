# AI Chat Assistant

A lightweight LLM-powered web application built with FastAPI, SQLite, and an OpenAI-compatible streaming API.

![AI Chat Assistant screenshot](screenshots/app.png)

## Features

- FastAPI backend with a small modular service layer.
- Browser chat interface with session switching and persisted history.
- SQLite storage for sessions, messages, and prompt settings.
- Streaming assistant replies over Server-Sent Events.
- Editable system prompt from the web UI.
- OpenAI-compatible model configuration through environment variables.

## Project Summary

- Developed a lightweight LLM-powered web application using FastAPI.
- Implemented streamed responses, conversation history management, prompt configuration, and SQLite persistence.
- Designed a modular backend supporting multiple OpenAI-compatible language models.

## Tech Stack

- Backend: FastAPI, Pydantic, OpenAI Python SDK
- Database: SQLite
- Frontend: HTML, CSS, JavaScript
- Streaming: Server-Sent Events over `StreamingResponse`

## Project Structure

```text
app/
  main.py              # FastAPI app and API routes
  config.py            # Environment-based settings
  database.py          # SQLite connection and schema
  repositories.py      # Data access helpers
  schemas.py           # Request and response models
  llm.py               # OpenAI-compatible streaming client
  chat.py              # Conversation orchestration
  static/
    index.html
    styles.css
    app.js
data/                  # Runtime SQLite database location
screenshots/
requirements.txt
.env.example
```

## Local Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your local environment file:

```bash
cp .env.example .env
```

4. Edit `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

5. Start the app:

```bash
uvicorn app.main:app --reload
```

6. Open the browser:

```text
http://127.0.0.1:8000
```

## OpenAI-Compatible Providers

The backend uses the OpenAI Python SDK with configurable `base_url`, so it can work with providers that expose an OpenAI-compatible Chat Completions API.

Common configuration values:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.7
MAX_HISTORY_MESSAGES=20
```

This follows the same provider-replacement idea used by projects such as Pydantic AI: keep model-specific configuration outside the application flow, and route model calls through one dedicated client module.

## API Overview

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Chat UI |
| `GET` | `/api/health` | Runtime model/provider status |
| `GET` | `/api/sessions` | List chat sessions |
| `POST` | `/api/sessions` | Create a new session |
| `GET` | `/api/sessions/{session_id}/messages` | Read session history |
| `POST` | `/api/chat/stream` | Stream assistant response |
| `GET` | `/api/prompt` | Read system prompt |
| `PUT` | `/api/prompt` | Update system prompt |

## Streaming Protocol

`POST /api/chat/stream` returns `text/event-stream` events:

```text
event: session
data: {"session_id": 1}

event: delta
data: {"content": "Hello"}

event: done
data: {"session_id": 1, "content": "Hello!"}
```

If the provider fails or the API key is missing, the stream returns:

```text
event: error
data: {"message": "OPENAI_API_KEY is not configured."}
```

## Notes

- SQLite files are created under `data/` at runtime and ignored by Git.
- The app stores the full assistant message after streaming completes.
- The model context is limited by `MAX_HISTORY_MESSAGES` to keep requests practical.
- This is intentionally lightweight: it does not include authentication, RAG, file upload, or multi-user permissions.
