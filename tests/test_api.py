import json
import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.config import get_settings
from app.database import init_db
from app.main import app


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            if line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if event and data is not None:
            events.append((event, data))
    return events


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(
            os.environ,
            {
                "DATABASE_PATH": os.path.join(
                    self.temp_dir.name,
                    "ai_chat_test.sqlite3",
                ),
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        self.env_patch.start()
        get_settings.cache_clear()
        init_db()
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        get_settings.cache_clear()
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_health_reports_missing_api_key(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "degraded",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key_configured": False,
            },
        )

    def test_prompt_can_be_read_and_updated(self):
        self.assertEqual(
            self.client.get("/api/prompt").json(),
            {"system_prompt": "You are a helpful, concise AI assistant."},
        )

        response = self.client.put(
            "/api/prompt",
            json={"system_prompt": "Answer in short, practical steps."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"system_prompt": "Answer in short, practical steps."},
        )
        self.assertEqual(self.client.get("/api/prompt").json(), response.json())

    def test_sessions_and_messages_flow(self):
        session_response = self.client.post("/api/sessions", json={"title": "Roadmap"})

        self.assertEqual(session_response.status_code, 200)
        session = session_response.json()
        self.assertEqual(session["title"], "Roadmap")

        sessions = self.client.get("/api/sessions").json()
        self.assertEqual(sessions[0]["id"], session["id"])

        messages = self.client.get(f"/api/sessions/{session['id']}/messages")
        self.assertEqual(messages.status_code, 200)
        self.assertEqual(messages.json(), [])

    def test_missing_session_messages_return_404(self):
        response = self.client.get("/api/sessions/999/messages")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Session not found"})

    def test_stream_returns_clear_error_when_api_key_is_missing(self):
        with self.client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "Hello"},
        ) as response:
            text = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            parse_sse(text),
            [
                ("session", {"session_id": 1}),
                ("error", {"message": "OPENAI_API_KEY is not configured."}),
            ],
        )

        messages = self.client.get("/api/sessions/1/messages").json()
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages],
            [("user", "Hello")],
        )

    def test_documents_are_chunked_and_retrievable(self):
        response = self.client.post(
            "/api/documents",
            json={
                "title": "Admissions project",
                "content": (
                    "Retrieval augmented generation helps a chat assistant answer "
                    "questions with citations from uploaded project documents."
                ),
            },
        )

        self.assertEqual(response.status_code, 200)
        document = response.json()
        self.assertEqual(document["title"], "Admissions project")
        self.assertGreaterEqual(document["chunk_count"], 1)

        documents = self.client.get("/api/documents").json()
        self.assertEqual(documents[0]["id"], document["id"])

        preview = self.client.post(
            "/api/retrieval/preview",
            json={"query": "How does RAG use citations?", "limit": 2},
        )
        self.assertEqual(preview.status_code, 200)
        sources = preview.json()["sources"]
        self.assertGreaterEqual(len(sources), 1)
        self.assertEqual(sources[0]["document_title"], "Admissions project")

    def test_rag_stream_emits_sources_before_model_error(self):
        self.client.post(
            "/api/documents",
            json={
                "title": "Memory design",
                "content": "Long-term memory stores document chunks for retrieval.",
            },
        )

        with self.client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "What stores document chunks?", "use_rag": True},
        ) as response:
            text = "".join(response.iter_text())

        events = parse_sse(text)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[0], ("session", {"session_id": 1}))
        self.assertEqual(events[1][0], "sources")
        self.assertEqual(events[1][1]["sources"][0]["document_title"], "Memory design")
        self.assertEqual(
            events[2],
            ("error", {"message": "OPENAI_API_KEY is not configured."}),
        )

    def test_evaluation_records_model_configuration_errors(self):
        response = self.client.post(
            "/api/evaluations",
            json={"models": ["gpt-test-a", "gpt-test-b"], "prompt": "Compare RAG."},
        )

        self.assertEqual(response.status_code, 200)
        run = response.json()
        self.assertEqual(run["prompt"], "Compare RAG.")
        self.assertEqual(len(run["results"]), 2)
        self.assertEqual(run["results"][0]["model"], "gpt-test-a")
        self.assertIn("OPENAI_API_KEY", run["results"][0]["error"])

    def test_stream_success_persists_user_and_assistant_messages(self):
        from app import chat

        test_case = self

        class FakeLLMClient:
            def __init__(self, settings):
                self.settings = settings

            async def stream_chat(self, messages):
                test_case.assertEqual(
                    messages[-1],
                    {"role": "user", "content": "Hello"},
                )
                yield "Hi"
                yield " there"

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            get_settings.cache_clear()
            with mock.patch.object(chat, "OpenAICompatibleClient", FakeLLMClient):
                with self.client.stream(
                    "POST",
                    "/api/chat/stream",
                    json={"message": "Hello"},
                ) as response:
                    text = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            parse_sse(text),
            [
                ("session", {"session_id": 1}),
                ("delta", {"content": "Hi"}),
                ("delta", {"content": " there"}),
                ("done", {"session_id": 1, "content": "Hi there"}),
            ],
        )

        messages = self.client.get("/api/sessions/1/messages").json()
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages],
            [
                ("user", "Hello"),
                ("assistant", "Hi there"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
