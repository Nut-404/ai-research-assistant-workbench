const state = {
  sessions: [],
  activeSessionId: null,
  isStreaming: false,
};

const els = {
  sessions: document.querySelector("#sessions"),
  newSessionButton: document.querySelector("#new-session-button"),
  messages: document.querySelector("#messages"),
  form: document.querySelector("#chat-form"),
  input: document.querySelector("#message-input"),
  sendButton: document.querySelector("#send-button"),
  status: document.querySelector("#status"),
  activeTitle: document.querySelector("#active-title"),
  modelName: document.querySelector("#model-name"),
  baseUrl: document.querySelector("#base-url"),
  promptInput: document.querySelector("#prompt-input"),
  savePromptButton: document.querySelector("#save-prompt-button"),
  promptSaveState: document.querySelector("#prompt-save-state"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed: ${response.status}`);
  }

  return response.json();
}

function setStatus(text, isError = false) {
  els.status.textContent = text;
  els.status.classList.toggle("error", isError);
}

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(`${value}Z`));
}

function renderSessions() {
  els.sessions.innerHTML = "";
  state.sessions.forEach((session) => {
    const button = document.createElement("button");
    button.className = "session-button";
    button.type = "button";
    button.classList.toggle("active", session.id === state.activeSessionId);
    button.innerHTML = `
      <strong>${escapeHtml(session.title)}</strong>
      <span>${formatDate(session.updated_at)}</span>
    `;
    button.addEventListener("click", () => selectSession(session.id));
    els.sessions.appendChild(button);
  });
}

function renderMessages(messages) {
  els.messages.innerHTML = "";
  if (!messages.length) {
    els.messages.innerHTML = `
      <div class="empty-state">
        <h3>Start a practical assistant session</h3>
        <p>Messages stream in real time and are saved to SQLite.</p>
      </div>
    `;
    return;
  }

  messages.forEach((message) => appendMessage(message.role, message.content));
  scrollToBottom();
}

function appendMessage(role, content = "") {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  wrapper.appendChild(bubble);
  els.messages.appendChild(wrapper);
  scrollToBottom();
  return bubble;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

async function loadHealth() {
  const health = await api("/api/health");
  els.modelName.textContent = health.model;
  els.baseUrl.textContent = health.base_url;
}

async function loadPrompt() {
  const prompt = await api("/api/prompt");
  els.promptInput.value = prompt.system_prompt;
}

async function savePrompt() {
  els.promptSaveState.textContent = "Saving";
  await api("/api/prompt", {
    method: "PUT",
    body: JSON.stringify({ system_prompt: els.promptInput.value }),
  });
  els.promptSaveState.textContent = "Saved";
}

async function loadSessions() {
  state.sessions = await api("/api/sessions");
  renderSessions();
}

async function createSession() {
  const session = await api("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ title: "New chat" }),
  });
  state.sessions.unshift(session);
  state.activeSessionId = session.id;
  els.activeTitle.textContent = session.title;
  renderSessions();
  renderMessages([]);
  setStatus("Ready");
}

async function selectSession(sessionId) {
  state.activeSessionId = sessionId;
  const session = state.sessions.find((item) => item.id === sessionId);
  els.activeTitle.textContent = session?.title || "Conversation";
  renderSessions();
  const messages = await api(`/api/sessions/${sessionId}/messages`);
  renderMessages(messages);
  setStatus("Ready");
}

function parseSseEvents(buffer) {
  const events = [];
  const blocks = buffer.split("\n\n");
  const remainder = blocks.pop() || "";

  for (const block of blocks) {
    const lines = block.split("\n");
    const eventLine = lines.find((line) => line.startsWith("event:"));
    const dataLine = lines.find((line) => line.startsWith("data:"));
    if (!eventLine || !dataLine) continue;
    events.push({
      event: eventLine.slice(6).trim(),
      data: JSON.parse(dataLine.slice(5).trim()),
    });
  }

  return { events, remainder };
}

async function sendMessage(event) {
  event.preventDefault();
  if (state.isStreaming) return;

  const message = els.input.value.trim();
  if (!message) return;

  state.isStreaming = true;
  els.sendButton.disabled = true;
  els.input.value = "";
  els.input.style.height = "auto";
  setStatus("Streaming");

  if (els.messages.querySelector(".empty-state")) {
    els.messages.innerHTML = "";
  }

  appendMessage("user", message);
  const assistantBubble = appendMessage("assistant", "");

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.activeSessionId,
        message,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Streaming failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parsed = parseSseEvents(buffer);
      buffer = parsed.remainder;

      for (const item of parsed.events) {
        if (item.event === "session") {
          state.activeSessionId = item.data.session_id;
        }

        if (item.event === "delta") {
          assistantBubble.textContent += item.data.content;
          scrollToBottom();
        }

        if (item.event === "error") {
          assistantBubble.textContent = item.data.message;
          setStatus("Error", true);
        }

        if (item.event === "done") {
          setStatus("Saved");
        }
      }
    }

    await loadSessions();
    if (state.activeSessionId) {
      await selectSession(state.activeSessionId);
    }
  } catch (error) {
    assistantBubble.textContent = error.message;
    setStatus("Error", true);
  } finally {
    state.isStreaming = false;
    els.sendButton.disabled = false;
    els.input.focus();
  }
}

function autoGrowInput() {
  els.input.style.height = "auto";
  els.input.style.height = `${els.input.scrollHeight}px`;
}

async function init() {
  try {
    await Promise.all([loadHealth(), loadPrompt(), loadSessions()]);
    if (state.sessions.length) {
      await selectSession(state.sessions[0].id);
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

els.newSessionButton.addEventListener("click", createSession);
els.savePromptButton.addEventListener("click", savePrompt);
els.form.addEventListener("submit", sendMessage);
els.input.addEventListener("input", autoGrowInput);
els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

init();
