/**
 * Borus — Frontend Chat & Ingestion Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chatForm");
  const queryInput = document.getElementById("queryInput");
  const messagesContainer = document.getElementById("messagesContainer");
  const btnSend = document.getElementById("btnSend");
  const btnClearChat = document.getElementById("btnClearChat");
  const btnIngest = document.getElementById("btnIngest");
  const btnUploadTrigger = document.getElementById("btnUploadTrigger");
  const fileInput = document.getElementById("fileInput");
  const toast = document.getElementById("toast");

  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const vectorCount = document.getElementById("vectorCount");
  const modelName = document.getElementById("modelName");
  const suggestionChips = document.querySelectorAll(".suggestion-chip");

  // State
  let chatHistory = [];
  let isSubmitting = false;

  // Initialize marked options for safe rendering and code highlighting
  marked.setOptions({
    breaks: true,
    highlight: function (code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(code, { language: lang }).value;
        } catch (e) {
          console.error(e);
        }
      }
      return hljs.highlightAuto(code).value;
    },
  });

  // Show Toast notification
  function showToast(message, duration = 3500) {
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => {
      toast.classList.remove("show");
    }, duration);
  }

  // Fetch API health and stats
  async function fetchHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();

      statusDot.className = "status-dot healthy";
      statusText.textContent = "Online";
      vectorCount.textContent = data.total_vectors;
      if (data.groq_model) {
        modelName.textContent = data.groq_model.split("/").pop();
      }

      if (!data.groq_configured) {
        statusDot.className = "status-dot warning";
        statusText.textContent = "Sem Groq Key";
      }
    } catch (err) {
      statusDot.className = "status-dot error";
      statusText.textContent = "Offline";
      console.warn("Could not connect to API:", err);
    }
  }

  // Auto-resize textarea
  queryInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 160) + "px";
  });

  // Keydown handler for Shift+Enter vs Enter to submit
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Render a message in the chat
  function appendMessage(role, text, sources = [], chunksCount = 0) {
    const wrapper = document.createElement("div");
    wrapper.className = `message-wrapper ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";

    if (role === "assistant") {
      avatar.innerHTML = `<img src="/static/borus_avatar.jpg" alt="Borus" class="msg-avatar-img">`;
    } else {
      avatar.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
          <circle cx="12" cy="7" r="4"/>
        </svg>`;
    }

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (role === "assistant") {
      bubble.innerHTML = marked.parse(text);

      // Append Sources citations if available
      if (sources && sources.length > 0) {
        const sourcesContainer = document.createElement("div");
        sourcesContainer.className = "sources-container";
        sourcesContainer.innerHTML = `
          <div class="sources-title">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
            </svg>
            Fontes Consultadas (${sources.length}):
          </div>
          <div class="sources-list">
            ${sources
              .map(
                (s) => `
                <div class="source-chip" title="${(s.snippet || '').replace(/"/g, "&quot;")}">
                  📄 ${s.source}${s.page ? ` (Pág. ${s.page})` : ""}
                  <span class="source-score">${Math.round(s.similarity_score * 100)}%</span>
                </div>`
              )
              .join("")}
          </div>
        `;
        bubble.appendChild(sourcesContainer);
      }
    } else {
      bubble.textContent = text;
    }

    contentDiv.appendChild(bubble);
    wrapper.appendChild(avatar);
    wrapper.appendChild(contentDiv);
    messagesContainer.appendChild(wrapper);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return wrapper;
  }

  // Show typing indicator
  function showTypingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.className = "message-wrapper assistant typing-wrapper";
    wrapper.id = "typingIndicator";

    wrapper.innerHTML = `
      <div class="avatar">
        <img src="/static/borus_avatar.jpg" alt="Borus" class="msg-avatar-img">
      </div>
      <div class="message-content">
        <div class="bubble">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    `;
    messagesContainer.appendChild(wrapper);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function removeTypingIndicator() {
    const indicator = document.getElementById("typingIndicator");
    if (indicator) indicator.remove();
  }

  // Handle Form Submit (Chat Query)
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query || isSubmitting) return;

    // Reset input
    queryInput.value = "";
    queryInput.style.height = "auto";
    isSubmitting = true;
    btnSend.disabled = true;

    // Append user message
    appendMessage("user", query);
    chatHistory.push({ role: "user", content: query });

    showTypingIndicator();

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          history: chatHistory.slice(-6),
        }),
      });

      removeTypingIndicator();

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Erro HTTP ${response.status}`);
      }

      const data = await response.json();
      appendMessage("assistant", data.answer, data.sources, data.chunks_retrieved);
      chatHistory.push({ role: "assistant", content: data.answer });
    } catch (err) {
      removeTypingIndicator();
      appendMessage(
        "assistant",
        `❌ **Erro ao processar sua pergunta:** ${err.message}\n\n*Verifique se a API está online e se as variáveis de ambiente foram configuradas.*`
      );
    } finally {
      isSubmitting = false;
      btnSend.disabled = false;
      queryInput.focus();
    }
  });

  // Suggestion chips
  suggestionChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt) {
        queryInput.value = prompt;
        queryInput.style.height = "auto";
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

  // Clear chat
  btnClearChat.addEventListener("click", () => {
    chatHistory = [];
    messagesContainer.innerHTML = `
      <div class="message-wrapper assistant">
        <div class="avatar">
          <img src="/static/borus_avatar.jpg" alt="Borus" class="msg-avatar-img">
        </div>
        <div class="message-content">
          <div class="bubble">
            <p>Conversa reiniciada. Em que posso te ajudar hoje com a documentação do backend?</p>
          </div>
        </div>
      </div>
    `;
  });

  // Re-ingest docs_source button
  btnIngest.addEventListener("click", async () => {
    btnIngest.disabled = true;
    btnIngest.innerHTML = `<span>Indexando...</span>`;

    try {
      const res = await fetch("/ingest", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Falha na ingestão");

      showToast(`✅ ${data.message}`);
      fetchHealth();
    } catch (err) {
      showToast(`❌ Erro na ingestão: ${err.message}`);
    } finally {
      btnIngest.disabled = false;
      btnIngest.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
        </svg>
        Reindexar docs_source/ (.pdf)
      `;
    }
  });

  // Upload file button
  btnUploadTrigger.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", async () => {
    if (!fileInput.files || fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    btnUploadTrigger.disabled = true;
    btnUploadTrigger.textContent = "Processando PDF...";

    try {
      const res = await fetch("/ingest/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Falha no upload");

      showToast(`✅ ${data.message}`);
      fetchHealth();
    } catch (err) {
      showToast(`❌ ${err.message}`);
    } finally {
      fileInput.value = "";
      btnUploadTrigger.disabled = false;
      btnUploadTrigger.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
        </svg>
        Upload de Arquivo PDF (.pdf)
      `;
    }
  });

  // Initial health check
  fetchHealth();
});
