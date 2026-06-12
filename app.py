"""
Backend API for FAQ Agent with context visualization.
Exposes chat endpoint + context metrics.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DOCUMENT = "policies.md"
MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 512
WINDOW_SIZE = 1   # keep last N user+assistant pairs
# ──────────────────────────────────────────────────────────────────────────────

client = OpenAI()
enc = tiktoken.encoding_for_model(MODEL)
conversation_history = []


def trim_history(history: list, window_size: int) -> list:
    keep = window_size * 2
    return history[-keep:] if len(history) > keep else history


def load_document(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "# No policy document found\n\nPlease create a policies.md file."


def build_system_prompt(document: str) -> str:
    return f"""You are a helpful HR assistant.

You should:
- Answer policy questions using the policy document.
- Remember information the user tells you during the conversation.
- Use both the conversation history and the policy document when answering.

Policy information takes precedence if there is a conflict.

--- POLICY DOCUMENT ---
{document}
--- END DOCUMENT ---"""


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def get_context_info(last_user_message=None):
    """Build context info for visualization."""
    document = load_document(POLICY_DOCUMENT)
    system_prompt = build_system_prompt(document)

    windowed_history = trim_history(conversation_history, WINDOW_SIZE)

    system_tokens = count_tokens(system_prompt)
    windowed_tokens = sum(count_tokens(msg["content"]) for msg in windowed_history)
    total_tokens = system_tokens + windowed_tokens

    def truncate(text, max_len=200):
        return text[:max_len] + "..." if len(text) > max_len else text

    # Full payload = exactly what was sent to LLM (windowed, not full history)
    payload_messages = [{"role": "system", "content": system_prompt}] + windowed_history
    if last_user_message:
        payload_messages = payload_messages + [{"role": "user", "content": last_user_message}]

    full_payload_tokens = sum(count_tokens(m["content"]) for m in payload_messages)

    return {
        "system_prompt": truncate(system_prompt, 300),
        "system_prompt_full": system_prompt,
        "system_tokens": system_tokens,
        "conversation": [
            {
                "role": msg["role"],
                "content": truncate(msg["content"]),
                "tokens": count_tokens(msg["content"])
            }
            for msg in conversation_history
        ],
        "windowed_conversation": [
            {
                "role": msg["role"],
                "content": truncate(msg["content"]),
                "tokens": count_tokens(msg["content"])
            }
            for msg in windowed_history
        ],
        "history_tokens": windowed_tokens,
        "total_history_messages": len(conversation_history),
        "windowed_history_messages": len(windowed_history),
        "total_tokens": total_tokens,
        "max_tokens": MAX_TOKENS,
        "message_count": len(conversation_history),
        "window_size": WINDOW_SIZE,
        "full_payload": [
            {
                "role": m["role"],
                "content": m["content"],
                "tokens": count_tokens(m["content"])
            }
            for m in payload_messages
        ],
        "full_payload_tokens": full_payload_tokens,
    }


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """Process user message and return response + context."""
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    document = load_document(POLICY_DOCUMENT)
    system_prompt = build_system_prompt(document)

    # Build messages: system + windowed history + new message
    windowed = trim_history(conversation_history, WINDOW_SIZE)
    messages = [{"role": "system", "content": system_prompt}] + windowed + [
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )
        assistant_reply = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Capture context BEFORE appending (payload = what was actually sent)
    context = get_context_info(last_user_message=user_input)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    # Update message count to reflect post-append state
    context["message_count"] = len(conversation_history)

    return jsonify({
        "reply": assistant_reply,
        "context": context,
    })


@app.route("/api/context", methods=["GET"])
def context_endpoint():
    """Get current context info."""
    return jsonify(get_context_info())


@app.route("/api/clear", methods=["POST"])
def clear_endpoint():
    """Clear conversation history."""
    global conversation_history
    conversation_history = []
    return jsonify({"status": "cleared"})


@app.route("/", methods=["GET"])
def index():
    """Serve frontend."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FAQ Agent — Context Visualization</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background: #fafafa;
                height: 100vh;
                overflow: hidden;
            }

            .container {
                display: flex;
                height: 100vh;
            }

            .sidebar {
                width: 350px;
                background: #f5f5f5;
                border-right: 1px solid #e0e0e0;
                display: flex;
                flex-direction: column;
                overflow-y: auto;
            }

            .sidebar-header {
                padding: 16px;
                border-bottom: 1px solid #e0e0e0;
                background: white;
                font-weight: 600;
                color: #333;
                position: sticky;
                top: 0;
                z-index: 10;
            }

            .sidebar-section {
                padding: 12px;
                border-bottom: 1px solid #e0e0e0;
            }

            .section-title {
                font-size: 12px;
                font-weight: 700;
                color: #666;
                text-transform: uppercase;
                margin-bottom: 8px;
                letter-spacing: 0.5px;
            }

            .metric {
                display: flex;
                justify-content: space-between;
                margin: 6px 0;
                font-size: 13px;
            }

            .metric-label {
                color: #555;
            }

            .metric-value {
                font-weight: 600;
                color: #333;
                font-family: "Monaco", monospace;
            }

            .progress-bar {
                width: 100%;
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
                margin-top: 4px;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #4CAF50, #FFC107);
                transition: width 0.3s ease;
            }

            .message-item {
                background: white;
                padding: 8px;
                margin: 4px 0;
                border-radius: 4px;
                border-left: 3px solid #ddd;
                font-size: 12px;
            }

            .message-item.user {
                border-left-color: #2196F3;
            }

            .message-item.assistant {
                border-left-color: #4CAF50;
            }

            .message-role {
                font-weight: 600;
                font-size: 11px;
                text-transform: uppercase;
                color: #666;
                margin-bottom: 4px;
            }

            .message-content {
                color: #333;
                line-height: 1.3;
                word-break: break-word;
                font-family: monospace;
                font-size: 11px;
            }

            .message-tokens {
                font-size: 10px;
                color: #999;
                margin-top: 4px;
            }

            .chat-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: white;
            }

            .chat-header {
                padding: 16px;
                border-bottom: 1px solid #e0e0e0;
                background: white;
            }

            .chat-header h1 {
                font-size: 18px;
                color: #333;
            }

            .chat-body {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .chat-message {
                display: flex;
                gap: 12px;
                animation: fadeIn 0.3s ease;
            }

            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .chat-message.user {
                justify-content: flex-end;
            }

            .message-bubble {
                max-width: 60%;
                padding: 12px 16px;
                border-radius: 8px;
                word-wrap: break-word;
                line-height: 1.4;
            }

            .message-bubble.user {
                background: #2196F3;
                color: white;
                border-radius: 8px 2px 8px 8px;
            }

            .message-bubble.assistant {
                background: #f0f0f0;
                color: #333;
                border-radius: 2px 8px 8px 8px;
            }

            .chat-input-area {
                padding: 16px;
                border-top: 1px solid #e0e0e0;
                background: white;
                display: flex;
                gap: 8px;
            }

            .chat-input {
                flex: 1;
                padding: 10px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                font-family: inherit;
            }

            .chat-input:focus {
                outline: none;
                border-color: #2196F3;
                box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1);
            }

            button {
                padding: 10px 20px;
                background: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 600;
                font-size: 14px;
                transition: background 0.2s;
            }

            button:hover {
                background: #1976D2;
            }

            button:active {
                transform: scale(0.98);
            }

            button.secondary {
                background: #666;
                padding: 6px 12px;
                font-size: 12px;
            }

            button.secondary:hover {
                background: #555;
            }

            .loading {
                opacity: 0.6;
                pointer-events: none;
            }

            .spinner {
                display: inline-block;
                width: 12px;
                height: 12px;
                border: 2px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 0.6s linear infinite;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .system-prompt {
                background: white;
                padding: 8px;
                border-radius: 4px;
                border-left: 3px solid #FF9800;
                font-family: monospace;
                font-size: 10px;
                color: #555;
                line-height: 1.3;
                max-height: 120px;
                overflow-y: auto;
            }

            .sidebar::-webkit-scrollbar,
            .chat-body::-webkit-scrollbar,
            .system-prompt::-webkit-scrollbar {
                width: 6px;
            }

            .sidebar::-webkit-scrollbar-track,
            .chat-body::-webkit-scrollbar-track,
            .system-prompt::-webkit-scrollbar-track {
                background: transparent;
            }

            .sidebar::-webkit-scrollbar-thumb,
            .chat-body::-webkit-scrollbar-thumb,
            .system-prompt::-webkit-scrollbar-thumb {
                background: #ccc;
                border-radius: 3px;
            }

            .sidebar::-webkit-scrollbar-thumb:hover,
            .chat-body::-webkit-scrollbar-thumb:hover,
            .system-prompt::-webkit-scrollbar-thumb:hover {
                background: #999;
            }

            .section-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 8px;
            }

            .expand-btn {
                background: none;
                border: none;
                color: #666;
                cursor: pointer;
                padding: 0;
                width: 16px;
                height: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                line-height: 1;
                transition: color 0.2s;
            }

            .expand-btn:hover {
                color: #333;
            }

            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
                z-index: 1000;
                align-items: center;
                justify-content: center;
            }

            .modal.active {
                display: flex;
            }

            .modal-content {
                background: white;
                border-radius: 8px;
                width: 90%;
                max-width: 600px;
                max-height: 80vh;
                display: flex;
                flex-direction: column;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            }

            .modal-header {
                padding: 16px;
                border-bottom: 1px solid #e0e0e0;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: 600;
                color: #333;
            }

            .modal-body {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
                font-family: monospace;
                font-size: 12px;
                color: #555;
                line-height: 1.5;
                white-space: pre-wrap;
                word-wrap: break-word;
            }

            .modal-close {
                background: none;
                border: none;
                color: #999;
                cursor: pointer;
                font-size: 20px;
                padding: 0;
                width: 32px;
                height: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: color 0.2s;
            }

            .modal-close:hover {
                color: #333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="sidebar" id="sidebar">
                <div class="sidebar-header">Context Info</div>
                <div class="sidebar-section">
                    <div class="section-title">Token Usage</div>
                    <div class="metric">
                        <span class="metric-label">System Prompt:</span>
                        <span class="metric-value" id="systemTokens">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Windowed History:</span>
                        <span class="metric-value" id="historyTokens">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Payload Total:</span>
                        <span class="metric-value" id="payloadTokens">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Max per response:</span>
                        <span class="metric-value" id="maxTokens">512</span>
                    </div>
                    <div style="margin-top: 8px;">
                        <div style="font-size: 10px; color: #666;">Progress to max</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                        </div>
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="section-title">Sliding Window</div>
                    <div class="metric">
                        <span class="metric-label">Window size:</span>
                        <span class="metric-value" id="windowSize">1</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total messages stored:</span>
                        <span class="metric-value" id="messageCount">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Sent to LLM:</span>
                        <span class="metric-value" id="windowedCount">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Dropped:</span>
                        <span class="metric-value" id="droppedCount" style="color: #e53935;">0</span>
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="section-header">
                        <div class="section-title">Payload Sent to LLM</div>
                        <button class="expand-btn" onclick="openSystemPromptModal()" title="Expand">⤢</button>
                    </div>
                    <div style="font-size: 11px; color: #888; margin-bottom: 6px;">system + windowed history + user message</div>
                    <div class="system-prompt" id="systemPrompt"></div>
                </div>

                <div class="sidebar-section">
                    <div class="section-title">Full Conversation History <span style="font-size:10px; font-weight:400; color:#888;">(stored, not all sent)</span></div>
                    <div id="historyList"></div>
                </div>

                <div class="sidebar-section">
                    <button class="secondary" onclick="clearHistory()">Clear History</button>
                </div>
            </div>

            <div class="chat-container">
                <div class="chat-header">
                    <h1>FAQ Agent — Full Context Sent Every Message</h1>
                </div>
                <div class="chat-body" id="chatBody">
                    <div class="chat-message">
                        <div class="message-bubble assistant">
                            Hi! I'm an HR assistant. Ask me about company policies or employee benefits.
                        </div>
                    </div>
                </div>
                <div class="chat-input-area">
                    <input
                        type="text"
                        class="chat-input"
                        id="chatInput"
                        placeholder="Ask a question..."
                        onkeypress="handleEnter(event)"
                    >
                    <button id="sendBtn" onclick="sendMessage(this)">Send</button>
                </div>
            </div>
        </div>

        <!-- Full Payload Modal -->
        <div class="modal" id="systemPromptModal" onclick="closeSystemPromptModal(event)">
            <div class="modal-content" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <span>Full Payload Sent to LLM</span>
                    <button class="modal-close" onclick="closeSystemPromptModal()">×</button>
                </div>
                <div class="modal-body" id="systemPromptFull"></div>
            </div>
        </div>

        <script>
            const API_BASE = window.location.origin;
            let fullPayload = [];

            function openSystemPromptModal() {
                document.getElementById("systemPromptModal").classList.add("active");
                const body = document.getElementById("systemPromptFull");
                body.innerHTML = "";
                fullPayload.forEach((msg, i) => {
                    const wrapper = document.createElement("div");
                    wrapper.style.cssText = "margin-bottom: 16px; border-left: 3px solid " +
                        (msg.role === "system" ? "#FF9800" : msg.role === "user" ? "#2196F3" : "#4CAF50") +
                        "; padding-left: 10px;";
                    wrapper.innerHTML =
                        '<div style="font-weight:700; font-size:11px; text-transform:uppercase; color:#888; margin-bottom:4px;">' +
                        escapeHtml(msg.role) + ' <span style="font-weight:400; color:#bbb;">(' + msg.tokens + ' tokens)</span></div>' +
                        '<pre style="margin:0; white-space:pre-wrap; word-wrap:break-word; font-family:monospace; font-size:12px; color:#444; line-height:1.5;">' +
                        escapeHtml(msg.content) + '</pre>';
                    body.appendChild(wrapper);
                });
            }

            function closeSystemPromptModal(event) {
                if (event && event.target.id !== "systemPromptModal") return;
                document.getElementById("systemPromptModal").classList.remove("active");
            }

            // Close modal on escape key
            document.addEventListener("keydown", (e) => {
                if (e.key === "Escape") closeSystemPromptModal();
            });

            async function sendMessage(btn) {
                const input = document.getElementById("chatInput");
                const message = input.value.trim();

                if (!message) return;

                const button = btn || document.getElementById("sendBtn");
                const originalText = button.textContent;

                addMessageToChat("user", message);
                input.value = "";
                input.focus();

                button.disabled = true;
                button.innerHTML = '<span class="spinner"></span>';

                try {
                    const response = await fetch(`${API_BASE}/api/chat`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ message }),
                    });

                    const data = await response.json();

                    if (response.ok) {
                        addMessageToChat("assistant", data.reply);
                        updateContext(data.context);
                    } else {
                        addMessageToChat("assistant", `Error: ${data.error}`);
                    }
                } catch (error) {
                    addMessageToChat("assistant", `Error: ${error.message}`);
                } finally {
                    button.disabled = false;
                    button.textContent = originalText;
                }
            }

            function addMessageToChat(role, content) {
                const chatBody = document.getElementById("chatBody");
                const messageDiv = document.createElement("div");
                messageDiv.className = `chat-message ${role}`;

                const bubble = document.createElement("div");
                bubble.className = `message-bubble ${role}`;
                bubble.textContent = content;

                messageDiv.appendChild(bubble);
                chatBody.appendChild(messageDiv);
                chatBody.scrollTop = chatBody.scrollHeight;
            }

            function updateContext(context) {
                document.getElementById("systemTokens").textContent = context.system_tokens.toLocaleString();
                document.getElementById("historyTokens").textContent = context.history_tokens.toLocaleString();
                document.getElementById("payloadTokens").textContent = (context.full_payload_tokens || 0).toLocaleString();
                document.getElementById("messageCount").textContent = context.total_history_messages || context.message_count;
                document.getElementById("windowedCount").textContent = context.windowed_history_messages || 0;
                document.getElementById("windowSize").textContent = context.window_size || 1;
                const dropped = (context.total_history_messages || 0) - (context.windowed_history_messages || 0);
                document.getElementById("droppedCount").textContent = dropped;

                const progress = Math.min(100, (context.full_payload_tokens / context.max_tokens) * 100);
                document.getElementById("progressFill").style.width = progress + "%";

                fullPayload = context.full_payload || [];
                const preview = fullPayload.map(m =>
                    "[" + m.role.toUpperCase() + "]\\n" + m.content.substring(0, 80) + (m.content.length > 80 ? "..." : "")
                ).join("\\n\\n");
                document.getElementById("systemPrompt").textContent = preview;

                // Full history — grey out messages not in windowed payload
                const windowedCount = context.windowed_history_messages || 0;
                const conversation = context.conversation || [];
                const droppedCount = conversation.length - windowedCount;
                const historyList = document.getElementById("historyList");
                historyList.innerHTML = conversation.map((msg, i) => {
                    const isDropped = i < droppedCount;
                    return `<div class="message-item ${msg.role}" style="${isDropped ? "opacity:0.35; text-decoration: line-through;" : ""}">
                        <div class="message-role">${msg.role}${isDropped ? " <span style=\\"color:#e53935;font-weight:400;\\">(dropped)</span>" : ""}</div>
                        <div class="message-content">${escapeHtml(msg.content)}</div>
                        <div class="message-tokens">${msg.tokens} tokens</div>
                    </div>`;
                }).join("");
            }

            function escapeHtml(text) {
                const div = document.createElement("div");
                div.textContent = text;
                return div.innerHTML;
            }

            async function clearHistory() {
                if (!confirm("Clear conversation history?")) return;

                try {
                    await fetch(`${API_BASE}/api/clear`, { method: "POST" });
                    document.getElementById("chatBody").innerHTML = `
                        <div class="chat-message">
                            <div class="message-bubble assistant">
                                Hi! I'm an HR assistant. Ask me about company policies or employee benefits.
                            </div>
                        </div>
                    `;

                    // Reset context display
                    document.getElementById("systemTokens").textContent = "0";
                    document.getElementById("historyTokens").textContent = "0";
                    document.getElementById("payloadTokens").textContent = "0";
                    document.getElementById("messageCount").textContent = "0";
                    document.getElementById("windowedCount").textContent = "0";
                    document.getElementById("droppedCount").textContent = "0";
                    document.getElementById("progressFill").style.width = "0%";
                    document.getElementById("historyList").innerHTML = "";
                } catch (error) {
                    alert("Error: " + error.message);
                }
            }

            function handleEnter(event) {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            }

            // Initial context load
            async function loadInitialContext() {
                try {
                    const response = await fetch(`${API_BASE}/api/context`);
                    const context = await response.json();
                    updateContext(context);
                } catch (error) {
                    console.error("Failed to load initial context:", error);
                }
            }

            loadInitialContext();
            document.getElementById("chatInput").focus();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True, port=5000)
