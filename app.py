import importlib.util
import os
import tiktoken
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

POLICY_DOCUMENT  = "policies.md"
current_exercise = 1  # switched via POST /api/switch

try:
    enc = tiktoken.encoding_for_model(os.getenv("LITELLM_MODEL", "gpt-4o-mini"))
except Exception:
    enc = tiktoken.get_encoding("cl100k_base")

conversation_history: list = []
last_compressed: bool = False


# ── Exercise loader ────────────────────────────────────────────────────────────

def get_exercise_module():
    """Reload exercise file from disk on every call — student edits take effect immediately."""
    path = Path(__file__).parent / f"exercise_{current_exercise}.py"
    spec = importlib.util.spec_from_file_location(f"exercise_{current_exercise}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_document(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "# No policy document found\n\nPlease create a policies.md file."


def count_tokens_text(text: str) -> int:
    return len(enc.encode(text)) if text else 0


def count_tokens_msgs(messages: list) -> int:
    total = 0
    for msg in messages:
        total += 4
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += count_tokens_text(content)
    return total


def serialize_msg(msg: dict, truncate_len: int = 200) -> dict:
    role    = msg.get("role", "unknown")
    content = msg.get("content") or ""
    if role == "assistant" and not content and msg.get("tool_calls"):
        names   = [tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]]
        content = f"[Tool call: {', '.join(names)}]"
    is_offloaded = isinstance(content, str) and content.startswith("[offloaded to disk:")
    short = content[:truncate_len] + "..." if len(content) > truncate_len else content
    return {
        "role":         role,
        "content":      short,
        "tokens":       count_tokens_text(content),
        "is_tool":      role == "tool",
        "is_offloaded": is_offloaded,
    }


def _trim_for_display(meta: dict, history: list) -> list:
    ws = meta.get("window_size", 0)
    if ws == 0:
        return history
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indices) <= ws:
        return history
    cutoff = user_indices[-ws]
    return history[cutoff:]


def get_context_info(user_input: str = None) -> dict:
    mod           = get_exercise_module()
    meta          = mod.EXERCISE_META
    document      = load_document(POLICY_DOCUMENT)
    system_prompt = mod.build_system_prompt(document)

    windowed    = _trim_for_display(meta, conversation_history)
    sys_tok     = count_tokens_text(system_prompt)
    hist_tok    = count_tokens_msgs(windowed)

    payload = [{"role": "system", "content": system_prompt}] + windowed
    if user_input:
        payload.append({"role": "user", "content": user_input})
    payload_tok = count_tokens_msgs(payload)

    tool_results_dir = getattr(mod, "TOOL_RESULTS_DIR", "tool_results")
    tool_msgs        = [m for m in conversation_history if m.get("role") == "tool"]
    offloaded        = [m for m in tool_msgs if str(m.get("content", "")).startswith("[offloaded to disk:")]
    offloaded_files  = len(os.listdir(tool_results_dir)) if os.path.isdir(tool_results_dir) else 0

    user_count          = sum(1 for m in conversation_history if m.get("role") == "user")
    windowed_user_count = sum(1 for m in windowed if m.get("role") == "user")

    context_window        = meta.get("context_window") or 800
    compression_threshold = meta.get("compression_threshold")

    return {
        "exercise_number":           meta["number"],
        "features":                  meta.get("features", []),
        "system_tokens":             sys_tok,
        "history_tokens":            hist_tok,
        "full_payload_tokens":       payload_tok,
        "context_window":            context_window,
        "compression_threshold":     int(compression_threshold * 100) if compression_threshold else None,
        "window_size":               meta.get("window_size", 0),
        "total_history_messages":    len(conversation_history),
        "windowed_history_messages": len(windowed),
        "total_turns":               user_count,
        "windowed_turns":            windowed_user_count,
        "tool_msg_count":            len(tool_msgs),
        "offloaded_count":           len(offloaded),
        "offloaded_files_on_disk":   offloaded_files,
        "compressed_last_turn":      last_compressed,
        "conversation":              [serialize_msg(m) for m in conversation_history],
        "windowed_conversation":     [serialize_msg(m) for m in windowed],
        "full_payload": [
            {"role": m["role"], "content": m.get("content") or "", "tokens": count_tokens_text(m.get("content") or "")}
            for m in payload
        ],
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/exercise")
def exercise_endpoint():
    mod = get_exercise_module()
    return jsonify(mod.EXERCISE_META)


@app.route("/api/switch", methods=["POST"])
def switch_exercise():
    global current_exercise, conversation_history, last_compressed
    num = int(request.json.get("exercise", 1))
    if num not in (1, 2, 3):
        return jsonify({"error": "exercise must be 1, 2, or 3"}), 400
    current_exercise     = num
    conversation_history = []
    last_compressed      = False
    mod  = get_exercise_module()
    meta = mod.EXERCISE_META
    return jsonify({"exercise": num, "meta": meta})


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    global conversation_history, last_compressed
    data       = request.json
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    try:
        mod      = get_exercise_module()
        document = load_document(POLICY_DOCUMENT)
        system   = mod.build_system_prompt(document)
        reply, conversation_history, metadata = mod.chat(system, conversation_history, user_input)
        last_compressed = metadata.get("compressed", False)
    except NotImplementedError as e:
        return jsonify({"error": str(e), "is_todo": True}), 501
    except Exception as e:
        return jsonify({"error": str(e), "is_todo": False}), 500

    context = get_context_info()
    return jsonify({"reply": reply, "context": context})


@app.route("/api/context", methods=["GET"])
def context_endpoint():
    return jsonify(get_context_info())


@app.route("/api/clear", methods=["POST"])
def clear_endpoint():
    global conversation_history, last_compressed
    conversation_history = []
    last_compressed      = False
    return jsonify({"status": "cleared"})


@app.route("/", methods=["GET"])
def index():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAQ Agent — Context Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #0d0d0d;
            color: #e8e8e8;
            height: 100vh;
            overflow: hidden;
        }

        .container { display: flex; height: 100vh; }

        /* ── Sidebar ───────────────────────────────────────────────────────── */
        .sidebar {
            width: 360px;
            background: #1a1a1a;
            border-right: 1px solid #333;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }

        .sidebar-header {
            padding: 14px 16px;
            border-bottom: 1px solid #333;
            background: #252525;
            font-weight: 600;
            color: #e8e8e8;
            font-size: 13px;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .sidebar-section { padding: 12px; border-bottom: 1px solid #333; }

        .section-title {
            font-size: 11px; font-weight: 700; color: #b0b0b0;
            text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px;
        }

        .metric { display: flex; justify-content: space-between; margin: 6px 0; font-size: 12px; }
        .metric-label { color: #a0a0a0; }
        .metric-value { font-weight: 600; color: #e8e8e8; font-family: "Monaco", monospace; font-size: 12px; }
        .metric-value.warn  { color: #ffb74d; }
        .metric-value.good  { color: #81c784; }
        .metric-value.muted { color: #808080; }

        .progress-bar { width: 100%; height: 5px; background: #333; border-radius: 3px; margin-top: 4px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #81c784, #ffb74d); transition: width 0.3s ease; }

        .badge { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 600; text-transform: uppercase; }
        .badge-dropped   { background: #5d1f1f; color: #ff9999; }
        .badge-tool      { background: #1a3a52; color: #64b5f6; }
        .badge-offloaded { background: #1d4d2f; color: #81c784; }

        .message-item { background: #252525; padding: 7px; margin: 3px 0; border-radius: 4px; border-left: 3px solid #404040; font-size: 11px; }
        .message-item.user      { border-left-color: #64b5f6; }
        .message-item.assistant { border-left-color: #81c784; }
        .message-item.tool      { border-left-color: #ce93d8; }

        .message-role    { font-weight: 700; font-size: 10px; text-transform: uppercase; color: #b0b0b0; margin-bottom: 3px; }
        .message-content { color: #e8e8e8; line-height: 1.3; word-break: break-word; font-family: monospace; font-size: 10px; }
        .message-tokens  { font-size: 10px; color: #808080; margin-top: 3px; }

        .system-prompt {
            background: #2a2a2a; padding: 7px; border-radius: 4px; border-left: 3px solid #ffb74d;
            font-family: monospace; font-size: 10px; color: #b0b0b0; line-height: 1.3;
            max-height: 110px; overflow-y: auto;
        }

        .section-header { display: flex; justify-content: space-between; align-items: center; }
        .expand-btn { background: none; border: none; color: #808080; cursor: pointer; padding: 0; font-size: 14px; }
        .expand-btn:hover { color: #b0b0b0; }

        /* ── Exercise panel ──────────────────────────────────────────────── */
        .exercise-panel {
            background: #1a237e; color: white;
            padding: 13px 14px; border-bottom: 2px solid #0d1450;
        }

        .exercise-number {
            font-size: 10px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 1px; color: #90caf9; margin-bottom: 3px;
        }

        .exercise-title  { font-size: 14px; font-weight: 700; color: white; margin-bottom: 7px; }
        .exercise-row    { font-size: 11px; color: #b3c5f5; margin-bottom: 4px; line-height: 1.4; }
        .exercise-row strong { color: #e3f2fd; }

        .todo-chip {
            display: inline-block; background: #e65100; color: white;
            font-family: monospace; font-size: 10px; padding: 2px 7px;
            border-radius: 3px; margin: 2px 2px 0 0;
        }

        /* ── Chat ────────────────────────────────────────────────────────── */
        .chat-container { flex: 1; display: flex; flex-direction: column; background: #121212; }

        .chat-header { padding: 0 16px; border-bottom: 1px solid #333; background: #1a1a1a; }

        .tab-row {
            display: flex; gap: 4px; padding: 10px 0 0;
        }

        .tab-btn {
            padding: 7px 16px;
            border: 1px solid #404040;
            border-bottom: none;
            background: #252525;
            color: #a0a0a0;
            border-radius: 6px 6px 0 0;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: background 0.15s, color 0.15s;
        }

        .tab-btn:hover { background: #333; color: #d0d0d0; }

        .tab-btn.active {
            background: #1a1a1a;
            color: #64b5f6;
            font-weight: 700;
            border-color: #555;
            border-bottom: 2px solid #1a1a1a;
            position: relative;
            top: 1px;
        }

        .tab-divider { height: 1px; background: #333; }

        .chat-subheader { padding: 8px 0 10px; font-size: 12px; color: #808080; }

        .chat-body {
            flex: 1; overflow-y: auto; padding: 16px;
            display: flex; flex-direction: column; gap: 12px;
        }

        .chat-message { display: flex; gap: 12px; animation: fadeIn 0.25s ease; }
        .chat-message.user { justify-content: flex-end; }

        @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }

        .message-bubble {
            max-width: 60%; padding: 11px 15px; border-radius: 8px;
            word-wrap: break-word; line-height: 1.45; font-size: 14px;
        }

        .message-bubble.user      { background: #1565c0; color: white; border-radius: 8px 2px 8px 8px; }
        .message-bubble.assistant { background: #2a2a2a; color: #e8e8e8; border-radius: 2px 8px 8px 8px; }
        .message-bubble.error {
            background: #3d2a1f; color: #ffb74d; border-radius: 6px;
            font-size: 13px; border: 1px solid #cc8844; max-width: 80%;
            font-family: monospace; line-height: 1.5;
        }

        .chat-input-area { padding: 14px 16px; border-top: 1px solid #333; background: #1a1a1a; display: flex; gap: 8px; }

        .chat-input {
            flex: 1; padding: 10px 12px; border: 1px solid #404040;
            border-radius: 4px; font-size: 14px; font-family: inherit;
            background: #252525; color: #e8e8e8;
        }

        .chat-input:focus { outline: none; border-color: #64b5f6; box-shadow: 0 0 0 2px rgba(100,181,246,.2); }

        .chat-input::placeholder { color: #808080; }

        button {
            padding: 10px 20px; background: #1565c0; color: white; border: none;
            border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 14px; transition: background .2s;
        }
        button:hover { background: #1976d2; }

        button.secondary { background: #555; padding: 6px 12px; font-size: 12px; }
        button.secondary:hover { background: #666; }

        .spinner {
            display: inline-block; width: 12px; height: 12px;
            border: 2px solid rgba(255,255,255,.3); border-radius: 50%;
            border-top-color: white; animation: spin .6s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* ── Modal ───────────────────────────────────────────────────────── */
        .modal {
            display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,.8); z-index: 1000; align-items: center; justify-content: center;
        }
        .modal.active { display: flex; }

        .modal-content {
            background: #1a1a1a; border-radius: 8px; width: 90%; max-width: 600px;
            max-height: 80vh; display: flex; flex-direction: column;
            box-shadow: 0 10px 40px rgba(0,0,0,.6);
        }

        .modal-header {
            padding: 14px 16px; border-bottom: 1px solid #333;
            display: flex; justify-content: space-between; align-items: center;
            font-weight: 600; color: #e8e8e8;
        }

        .modal-body {
            flex: 1; overflow-y: auto; padding: 16px;
            font-family: monospace; font-size: 12px; color: #b0b0b0; line-height: 1.5;
        }

        .modal-close { background: none; border: none; color: #808080; cursor: pointer; font-size: 22px; padding: 0; }
        .modal-close:hover { color: #d0d0d0; }

        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: #1a1a1a; }
        ::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #777; }
    </style>
</head>
<body>
<div class="container">

    <!-- ── Sidebar ─────────────────────────────────────────────────────────── -->
    <div class="sidebar">
        <div class="sidebar-header">Context Inspector</div>

        <div class="exercise-panel">
            <div class="exercise-number" id="exNumber">Exercise —</div>
            <div class="exercise-title"  id="exTitle">Loading…</div>
            <div class="exercise-row"><strong>Strategy:</strong> <span id="exStrategy"></span></div>
            <div class="exercise-row"><strong>Problem:</strong>  <span id="exProblem"></span></div>
            <div class="exercise-row" style="margin-top:6px;"><strong>Your task:</strong></div>
            <div class="exercise-row" id="exTask"></div>
            <div id="exTodos" style="margin-top:6px;"></div>
        </div>

        <div class="sidebar-section">
            <div class="section-title">Token Usage</div>
            <div class="metric"><span class="metric-label">System prompt</span><span class="metric-value" id="systemTokens">0</span></div>
            <div class="metric"><span class="metric-label">History (windowed)</span><span class="metric-value" id="historyTokens">0</span></div>
            <div class="metric"><span class="metric-label">Payload total</span><span class="metric-value" id="payloadTokens">0</span></div>
            <div class="metric"><span class="metric-label">Context window</span><span class="metric-value" id="contextWindow">—</span></div>
            <div style="margin-top:6px;">
                <div style="font-size:10px;color:#888;margin-bottom:2px;">History vs context window</div>
                <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
                <div style="font-size:10px;color:#aaa;margin-top:2px;" id="progressPct">0%</div>
            </div>
        </div>

        <div class="sidebar-section" id="sectionSlidingWindow">
            <div class="section-title">Sliding Window</div>
            <div class="metric"><span class="metric-label">Window size</span><span class="metric-value" id="windowSize">—</span></div>
            <div class="metric"><span class="metric-label">Total turns stored</span><span class="metric-value" id="totalTurns">0</span></div>
            <div class="metric"><span class="metric-label">Turns sent to LLM</span><span class="metric-value" id="windowedTurns">0</span></div>
            <div class="metric"><span class="metric-label">Turns dropped</span><span class="metric-value warn" id="droppedTurns">0</span></div>
        </div>

        <div class="sidebar-section" id="sectionSummarization">
            <div class="section-title">Summarization</div>
            <div class="metric"><span class="metric-label">Threshold</span><span class="metric-value" id="compressionThreshold">—</span></div>
            <div class="metric"><span class="metric-label">Compressed last turn</span><span class="metric-value" id="compressedLastTurn">—</span></div>
        </div>

        <div class="sidebar-section" id="sectionToolOffloading">
            <div class="section-title">Tool Output Offloading</div>
            <div class="metric"><span class="metric-label">Tool messages</span><span class="metric-value" id="toolMsgCount">0</span></div>
            <div class="metric"><span class="metric-label">Offloaded (stubs)</span><span class="metric-value good" id="offloadedCount">0</span></div>
            <div class="metric"><span class="metric-label">Files on disk</span><span class="metric-value" id="offloadedFiles">0</span></div>
        </div>

        <div class="sidebar-section">
            <div class="section-header">
                <div class="section-title">Payload Sent to LLM</div>
                <button class="expand-btn" onclick="openPayloadModal()" title="Expand">⤢</button>
            </div>
            <div style="font-size:10px;color:#999;margin-bottom:5px;">system + windowed history + user message</div>
            <div class="system-prompt" id="payloadPreview"></div>
        </div>

        <div class="sidebar-section">
            <div class="section-title">Full Conversation History
                <span style="font-weight:400;color:#aaa;font-size:10px;">(not all sent to LLM)</span>
            </div>
            <div id="historyList"></div>
        </div>

        <div class="sidebar-section">
            <button class="secondary" onclick="clearHistory()">Clear History</button>
        </div>
    </div>

    <!-- ── Chat ─────────────────────────────────────────────────────────────── -->
    <div class="chat-container">
        <div class="chat-header">
            <div class="tab-row">
                <button class="tab-btn active" id="tab1" onclick="switchExercise(1)">Exercise 1 — Sliding Window</button>
                <button class="tab-btn"         id="tab2" onclick="switchExercise(2)">Exercise 2 — Summarization</button>
                <button class="tab-btn"         id="tab3" onclick="switchExercise(3)">Exercise 3 — Tool Offloading</button>
            </div>
            <div class="tab-divider"></div>
            <div class="chat-subheader" id="chatHint">Loading…</div>
        </div>

        <div class="chat-body" id="chatBody">
            <div class="chat-message">
                <div class="message-bubble assistant">
                    Hi! I'm an HR assistant. How can i help you?.
                </div>
            </div>
        </div>

        <div class="chat-input-area">
            <input type="text" class="chat-input" id="chatInput"
                   placeholder="Ask a question…" onkeypress="handleEnter(event)">
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>
</div>

<!-- Payload modal -->
<div class="modal" id="payloadModal" onclick="closePayloadModal(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
        <div class="modal-header">
            <span>Full Payload Sent to LLM</span>
            <button class="modal-close" onclick="closePayloadModal()">×</button>
        </div>
        <div class="modal-body" id="payloadFull"></div>
    </div>
</div>

<script>
    const API = window.location.origin;
    let fullPayload = [];

    // ── Tab switching ─────────────────────────────────────────────────────────
    async function switchExercise(num) {
        const res  = await fetch(API + "/api/switch", {
            method:  "POST",
            headers: {"Content-Type": "application/json"},
            body:    JSON.stringify({exercise: num}),
        });
        const data = await res.json();
        [1, 2, 3].forEach(n => document.getElementById("tab" + n).classList.toggle("active", n === num));
        resetChatBody();
        applyExerciseMeta(data.meta);
        resetContextDisplay();
    }

    function applyExerciseMeta(meta) {
        document.getElementById("exNumber").textContent   = "Exercise " + meta.number;
        document.getElementById("exTitle").textContent    = meta.title;
        document.getElementById("exStrategy").textContent = meta.strategy;
        document.getElementById("exProblem").textContent  = meta.problem;
        document.getElementById("exTask").textContent     = meta.task;

        var hints = {
            1: "Try mentioning your employee ID, then ask about it again 3+ turns later.",
            2: "Chat for a while — watch the sidebar when compression triggers.",
            3: "Ask: \"What is the leave balance for EMP001?\" or \"How many people are in Engineering?\"",
        };
        document.getElementById("chatHint").textContent = hints[meta.number] || "";

        var todosEl = document.getElementById("exTodos");
        if (meta.todos && meta.todos.length > 0) {
            todosEl.innerHTML = meta.todos
                .map(function(t) { return "<span class=\"todo-chip\">" + escapeHtml(t) + "</span>"; })
                .join(" ");
        } else {
            todosEl.innerHTML = "<span style=\"font-size:11px;color:#a5d6a7;\">No TODOs — observe and experiment!</span>";
        }

        toggleSections(meta.features || []);
    }

    function toggleSections(features) {
        document.getElementById("sectionSlidingWindow").style.display  = features.indexOf("sliding_window")  >= 0 ? "" : "none";
        document.getElementById("sectionSummarization").style.display  = features.indexOf("summarization")   >= 0 ? "" : "none";
        document.getElementById("sectionToolOffloading").style.display = features.indexOf("tool_offloading") >= 0 ? "" : "none";
    }

    // ── Chat ──────────────────────────────────────────────────────────────────
    async function sendMessage() {
        var input   = document.getElementById("chatInput");
        var btn     = document.getElementById("sendBtn");
        var message = input.value.trim();
        if (!message) return;

        addMessageToChat("user", message);
        input.value = "";
        input.focus();

        btn.disabled  = true;
        btn.innerHTML = "<span class=\"spinner\"></span>";

        try {
            var response = await fetch(API + "/api/chat", {
                method:  "POST",
                headers: {"Content-Type": "application/json"},
                body:    JSON.stringify({message: message}),
            });
            var data = await response.json();

            if (response.status === 501) {
                addErrorToChat(data.error);
            } else if (response.ok) {
                addMessageToChat("assistant", data.reply);
                updateContext(data.context);
            } else {
                addErrorToChat("Server error: " + data.error);
            }
        } catch (err) {
            addErrorToChat("Network error: " + err.message);
        } finally {
            btn.disabled    = false;
            btn.textContent = "Send";
        }
    }

    function addMessageToChat(role, content) {
        var chatBody = document.getElementById("chatBody");
        var div      = document.createElement("div");
        div.className = "chat-message " + role;
        var bubble    = document.createElement("div");
        bubble.className  = "message-bubble " + role;
        bubble.textContent = content;
        div.appendChild(bubble);
        chatBody.appendChild(div);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function addErrorToChat(msg) {
        var chatBody = document.getElementById("chatBody");
        var div      = document.createElement("div");
        div.className = "chat-message";
        var bubble    = document.createElement("div");
        bubble.className = "message-bubble error";
        bubble.innerHTML =
            "<strong>TODO not implemented yet</strong><br><br>" +
            escapeHtml(msg) +
            "<br><br>Implement the function(s) shown in the sidebar, then send again.";
        div.appendChild(bubble);
        chatBody.appendChild(div);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    // ── Context display ───────────────────────────────────────────────────────
    function updateContext(ctx) {
        document.getElementById("systemTokens").textContent  = ctx.system_tokens.toLocaleString();
        document.getElementById("historyTokens").textContent = ctx.history_tokens.toLocaleString();
        document.getElementById("payloadTokens").textContent = ctx.full_payload_tokens.toLocaleString();
        document.getElementById("contextWindow").textContent = ctx.context_window ? ctx.context_window.toLocaleString() : "—";
        document.getElementById("windowSize").textContent    = ctx.window_size || "—";
        document.getElementById("totalTurns").textContent    = ctx.total_turns;
        document.getElementById("windowedTurns").textContent = ctx.windowed_turns;
        document.getElementById("droppedTurns").textContent  = ctx.total_turns - ctx.windowed_turns;

        var ct = ctx.compression_threshold;
        document.getElementById("compressionThreshold").textContent = ct != null ? ct + "%" : "—";
        document.getElementById("toolMsgCount").textContent   = ctx.tool_msg_count;
        document.getElementById("offloadedCount").textContent = ctx.offloaded_count;
        document.getElementById("offloadedFiles").textContent = ctx.offloaded_files_on_disk;

        var compressed = ctx.compressed_last_turn;
        var compEl     = document.getElementById("compressedLastTurn");
        compEl.textContent = compressed ? "Yes" : "No";
        compEl.className   = "metric-value " + (compressed ? "warn" : "muted");

        var pct = ctx.context_window
            ? Math.min(100, Math.round(ctx.history_tokens / ctx.context_window * 100))
            : 0;
        document.getElementById("progressFill").style.width = pct + "%";
        document.getElementById("progressPct").textContent  = pct + "%";

        fullPayload = ctx.full_payload || [];
        var preview = fullPayload.map(function(m) {
            return "[" + m.role.toUpperCase() + "]\n" +
                (m.content || "").substring(0, 80) + ((m.content || "").length > 80 ? "..." : "");
        }).join("\n\n");
        document.getElementById("payloadPreview").textContent = preview;

        var conversation  = ctx.conversation || [];
        var windowedCount = ctx.windowed_history_messages || 0;
        var droppedCount  = conversation.length - windowedCount;
        document.getElementById("historyList").innerHTML = conversation.map(function(msg, i) {
            var isDropped   = i < droppedCount;
            var isTool      = msg.is_tool;
            var isOffloaded = msg.is_offloaded;
            var badges = "";
            if (isDropped)   badges += " <span class=\"badge badge-dropped\">dropped</span>";
            if (isTool)      badges += " <span class=\"badge badge-tool\">tool</span>";
            if (isOffloaded) badges += " <span class=\"badge badge-offloaded\">offloaded</span>";
            return "<div class=\"message-item " + msg.role + "\" style=\"" + (isDropped ? "opacity:.35;" : "") + "\">" +
                "<div class=\"message-role\">" + msg.role + badges + "</div>" +
                "<div class=\"message-content\">" + escapeHtml(msg.content) + "</div>" +
                "<div class=\"message-tokens\">" + msg.tokens + " tokens</div>" +
                "</div>";
        }).join("");

        if (ctx.features) toggleSections(ctx.features);
    }

    function resetContextDisplay() {
        ["systemTokens","historyTokens","payloadTokens","totalTurns",
         "windowedTurns","droppedTurns","toolMsgCount","offloadedCount","offloadedFiles"]
            .forEach(function(id) { document.getElementById(id).textContent = "0"; });
        document.getElementById("progressFill").style.width   = "0%";
        document.getElementById("progressPct").textContent    = "0%";
        document.getElementById("compressedLastTurn").textContent = "—";
        document.getElementById("historyList").innerHTML      = "";
        document.getElementById("payloadPreview").textContent = "";
    }

    function resetChatBody() {
        document.getElementById("chatBody").innerHTML =
            "<div class=\"chat-message\">" +
            "<div class=\"message-bubble assistant\">Hi! I'm an HR assistant. How can i help you? </div>" +
            "</div>";
    }

    async function clearHistory() {
        if (!confirm("Clear conversation history?")) return;
        await fetch(API + "/api/clear", {method: "POST"});
        resetChatBody();
        resetContextDisplay();
    }

    // ── Payload modal ─────────────────────────────────────────────────────────
    function roleColor(role) {
        return role === "system" ? "#FF9800" : role === "user" ? "#2196F3" : role === "tool" ? "#9C27B0" : "#4CAF50";
    }

    function openPayloadModal() {
        document.getElementById("payloadModal").classList.add("active");
        var body = document.getElementById("payloadFull");
        body.innerHTML = "";
        fullPayload.forEach(function(msg) {
            var w = document.createElement("div");
            w.style.cssText = "margin-bottom:16px;border-left:3px solid " + roleColor(msg.role) + ";padding-left:10px;";
            w.innerHTML =
                "<div style=\"font-weight:700;font-size:11px;text-transform:uppercase;color:#888;margin-bottom:4px;\">" +
                escapeHtml(msg.role) + " <span style=\"font-weight:400;color:#bbb;\">(" + msg.tokens + " tokens)</span></div>" +
                "<pre style=\"margin:0;white-space:pre-wrap;word-wrap:break-word;font-size:12px;color:#444;line-height:1.5;\">" +
                escapeHtml(msg.content || "") + "</pre>";
            body.appendChild(w);
        });
    }

    function closePayloadModal(event) {
        if (event && event.target.id !== "payloadModal") return;
        document.getElementById("payloadModal").classList.remove("active");
    }

    document.addEventListener("keydown", function(e) { if (e.key === "Escape") closePayloadModal(); });

    function handleEnter(e) {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    }

    function escapeHtml(text) {
        var d = document.createElement("div");
        d.textContent = text || "";
        return d.innerHTML;
    }

    // ── Init ──────────────────────────────────────────────────────────────────
    async function init() {
        try {
            var res  = await fetch(API + "/api/exercise");
            var meta = await res.json();
            applyExerciseMeta(meta);
            var ctxRes = await fetch(API + "/api/context");
            var ctx    = await ctxRes.json();
            updateContext(ctx);
        } catch (e) {
            console.error("Init failed:", e);
        }
        document.getElementById("chatInput").focus();
    }

    init();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("\n  FAQ Agent — Context Visualization")
    print("  Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)
