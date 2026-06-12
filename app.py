"""
Backend API for FAQ Agent with context visualization.
Strategies: sliding window + token-threshold summarization + tool output offloading.
"""

import json
import os
import tiktoken
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DOCUMENT       = "policies.md"
MODEL                 = "gpt-3.5-turbo"
MAX_TOKENS            = 512
CONTEXT_WINDOW        = 400
COMPRESSION_THRESHOLD = 0.6
WINDOW_SIZE           = 2
TOOL_RESULTS_DIR      = "tool_results"
# ──────────────────────────────────────────────────────────────────────────────

# ── Employee data ──────────────────────────────────────────────────────────────
EMPLOYEE_DB = {
    "EMP001": {"name": "Alice Johnson",  "department": "Engineering", "leave_casual": 8,  "leave_sick": 5,  "leave_earned": 12, "manager": "Carol White"},
    "EMP002": {"name": "Bob Smith",      "department": "HR",          "leave_casual": 6,  "leave_sick": 3,  "leave_earned": 15, "manager": "David Lee"},
    "EMP003": {"name": "Priya Nair",     "department": "Finance",     "leave_casual": 10, "leave_sick": 7,  "leave_earned": 8,  "manager": "Carol White"},
    "EMP004": {"name": "James Okafor",   "department": "Engineering", "leave_casual": 5,  "leave_sick": 2,  "leave_earned": 20, "manager": "Carol White"},
    "EMP005": {"name": "Sarah Chen",     "department": "Marketing",   "leave_casual": 9,  "leave_sick": 4,  "leave_earned": 6,  "manager": "David Lee"},
}

DEPARTMENT_DB = {
    "Engineering": {"headcount": 24, "manager": "Carol White", "location": "Floor 3"},
    "HR":          {"headcount": 8,  "manager": "David Lee",   "location": "Floor 1"},
    "Finance":     {"headcount": 12, "manager": "Ravi Menon",  "location": "Floor 2"},
    "Marketing":   {"headcount": 10, "manager": "David Lee",   "location": "Floor 2"},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_leave_balance",
            "description": "Get remaining leave days for an employee by employee ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string", "description": "Employee ID, e.g. EMP001"}
                },
                "required": ["employee_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_department_info",
            "description": "Get headcount, manager, and office location for a department",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "Department name, e.g. Engineering"}
                },
                "required": ["department"],
            },
        },
    },
]
# ──────────────────────────────────────────────────────────────────────────────

client = OpenAI()
enc = tiktoken.encoding_for_model(MODEL)
conversation_history = []
last_compressed = False


def load_document(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "# No policy document found\n\nPlease create a policies.md file."


def build_system_prompt(document: str) -> str:
    return f"""You are a helpful HR assistant. You have access to tools to look up live employee and department data.

You should:
- Answer only the question that was just asked. Do not re-summarize data from previous tool calls.
- Answer policy questions using the policy document.
- Use tools to look up employee or department data when asked.
- Remember information the user tells you during the conversation.

Policy information takes precedence if there is a conflict.

--- POLICY DOCUMENT ---
{document}
--- END DOCUMENT ---"""


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


def execute_tool(name: str, args: dict) -> str:
    if name == "get_leave_balance":
        emp = EMPLOYEE_DB.get(args.get("employee_id", ""))
        if not emp:
            return json.dumps({"error": "Employee not found"})
        return json.dumps({**emp, "employee_id": args["employee_id"]})
    if name == "get_department_info":
        dept = DEPARTMENT_DB.get(args.get("department", ""))
        if not dept:
            return json.dumps({"error": "Department not found"})
        return json.dumps({**dept, "department": args["department"]})
    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Sliding window ─────────────────────────────────────────────────────────────

def trim_history(history: list) -> list:
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indices) <= WINDOW_SIZE:
        return history
    cutoff = user_indices[-WINDOW_SIZE]
    return history[cutoff:]


# ── Summarization ──────────────────────────────────────────────────────────────

def summarize_history(history: list) -> list:
    transcript = "\n".join(
        f"{m['role'].upper()}: {m.get('content') or '[tool call]'}" for m in history
    )
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a conversation summarizer. Summarize the conversation below "
                    "into a short paragraph. Capture key facts the user shared and questions "
                    "they asked. Be concise — details may be lost."
                ),
            },
            {"role": "user", "content": transcript},
        ],
    )
    summary = response.choices[0].message.content
    return [{"role": "assistant", "content": f"[Summary of earlier conversation]: {summary}"}]


def maybe_compress(history: list) -> tuple:
    token_budget = int(CONTEXT_WINDOW * COMPRESSION_THRESHOLD)
    if count_tokens_msgs(history) >= token_budget:
        return summarize_history(history), True
    return history, False


# ── Tool output offloading ─────────────────────────────────────────────────────

def offload_tool_result(tool_call_id: str, content: str) -> str:
    """
    Save tool result to disk. Return a short stub string.

    TODO:
      1. os.makedirs(TOOL_RESULTS_DIR, exist_ok=True)
      2. Write {"tool_call_id": ..., "content": ...} as JSON to
         f"{TOOL_RESULTS_DIR}/{tool_call_id}.json"
      3. Return f"[offloaded to disk: {tool_call_id}]"
    """
    raise NotImplementedError("TODO: implement offload_tool_result()")


def compact_tool_messages(messages: list) -> list:
    """
    Offload all tool results except the most recent one.
    Returns the modified message list.

    TODO:
      1. Find indices of all messages where role == "tool"
      2. Leave the last one untouched (still relevant to the current LLM turn)
      3. For all earlier tool messages: call offload_tool_result() and
         replace their "content" field with the returned stub
      4. Return the modified list
    """
    raise NotImplementedError("TODO: implement compact_tool_messages()")


# ── Chat ───────────────────────────────────────────────────────────────────────

def chat(history: list, user_input: str):
    global last_compressed
    document = load_document(POLICY_DOCUMENT)
    system_prompt = build_system_prompt(document)

    history = trim_history(history)
    history, compressed = maybe_compress(history)
    last_compressed = compressed

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role":       "assistant",
                "content":    msg.content,
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                result = execute_tool(tc.function.name, json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            # TODO: uncomment after implementing compact_tool_messages
            # messages = compact_tool_messages(messages)
        else:
            new_history = messages[1:]  # strip system prompt
            return msg.content, new_history, compressed


def serialize_msg(msg: dict, truncate_len: int = 200) -> dict:
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""

    if role == "assistant" and not content and msg.get("tool_calls"):
        names = [tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]]
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


def get_context_info(user_input: str = None) -> dict:
    document  = load_document(POLICY_DOCUMENT)
    system_prompt = build_system_prompt(document)

    windowed  = trim_history(conversation_history)
    sys_tok   = count_tokens_text(system_prompt)
    hist_tok  = count_tokens_msgs(windowed)

    payload = [{"role": "system", "content": system_prompt}] + windowed
    if user_input:
        payload.append({"role": "user", "content": user_input})
    payload_tok = count_tokens_msgs(payload)

    tool_msgs     = [m for m in conversation_history if m.get("role") == "tool"]
    offloaded     = [m for m in tool_msgs if str(m.get("content", "")).startswith("[offloaded to disk:")]
    offloaded_files = len(os.listdir(TOOL_RESULTS_DIR)) if os.path.isdir(TOOL_RESULTS_DIR) else 0

    user_count = sum(1 for m in conversation_history if m.get("role") == "user")
    windowed_user_count = sum(1 for m in windowed if m.get("role") == "user")

    return {
        "system_tokens":            sys_tok,
        "history_tokens":           hist_tok,
        "full_payload_tokens":      payload_tok,
        "max_tokens":               MAX_TOKENS,
        "context_window":           CONTEXT_WINDOW,
        "compression_threshold":    int(COMPRESSION_THRESHOLD * 100),
        "window_size":              WINDOW_SIZE,
        "total_history_messages":   len(conversation_history),
        "windowed_history_messages": len(windowed),
        "total_turns":              user_count,
        "windowed_turns":           windowed_user_count,
        "tool_msg_count":           len(tool_msgs),
        "offloaded_count":          len(offloaded),
        "offloaded_files_on_disk":  offloaded_files,
        "compressed_last_turn":     last_compressed,
        "conversation":             [serialize_msg(m) for m in conversation_history],
        "windowed_conversation":    [serialize_msg(m) for m in windowed],
        "full_payload":             [
            {"role": m["role"], "content": m.get("content") or "", "tokens": count_tokens_text(m.get("content") or "")}
            for m in payload
        ],
    }


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    global conversation_history
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    try:
        reply, conversation_history, compressed = chat(conversation_history, user_input)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    context = get_context_info()
    return jsonify({"reply": reply, "context": context})


@app.route("/api/context", methods=["GET"])
def context_endpoint():
    return jsonify(get_context_info())


@app.route("/api/clear", methods=["POST"])
def clear_endpoint():
    global conversation_history, last_compressed
    conversation_history = []
    last_compressed = False
    return jsonify({"status": "cleared"})


@app.route("/", methods=["GET"])
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FAQ Agent — Context Visualization</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background: #fafafa;
                height: 100vh;
                overflow: hidden;
            }

            .container { display: flex; height: 100vh; }

            .sidebar {
                width: 360px;
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

            .metric-label { color: #555; }

            .metric-value {
                font-weight: 600;
                color: #333;
                font-family: "Monaco", monospace;
            }

            .metric-value.warn  { color: #e65100; }
            .metric-value.good  { color: #2e7d32; }
            .metric-value.muted { color: #999; }

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

            .badge {
                display: inline-block;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }

            .badge-compressed { background: #fff3e0; color: #e65100; }
            .badge-offloaded  { background: #e8f5e9; color: #2e7d32; }
            .badge-tool       { background: #e3f2fd; color: #1565c0; }
            .badge-dropped    { background: #ffebee; color: #c62828; }

            .message-item {
                background: white;
                padding: 8px;
                margin: 4px 0;
                border-radius: 4px;
                border-left: 3px solid #ddd;
                font-size: 12px;
            }

            .message-item.user      { border-left-color: #2196F3; }
            .message-item.assistant { border-left-color: #4CAF50; }
            .message-item.tool      { border-left-color: #9C27B0; }

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

            .message-tokens { font-size: 10px; color: #999; margin-top: 4px; }

            .chat-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: white;
            }

            .chat-header {
                padding: 14px 16px;
                border-bottom: 1px solid #e0e0e0;
                background: white;
            }

            .chat-header h1 { font-size: 16px; color: #333; }
            .chat-header p  { font-size: 12px; color: #888; margin-top: 2px; }

            .chat-body {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .chat-message { display: flex; gap: 12px; animation: fadeIn 0.3s ease; }
            .chat-message.user { justify-content: flex-end; }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(8px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            .message-bubble {
                max-width: 60%;
                padding: 12px 16px;
                border-radius: 8px;
                word-wrap: break-word;
                line-height: 1.4;
                font-size: 14px;
            }

            .message-bubble.user      { background: #2196F3; color: white; border-radius: 8px 2px 8px 8px; }
            .message-bubble.assistant { background: #f0f0f0; color: #333; border-radius: 2px 8px 8px 8px; }

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
                box-shadow: 0 0 0 2px rgba(33,150,243,0.1);
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

            button:hover  { background: #1976D2; }
            button:active { transform: scale(0.98); }

            button.secondary {
                background: #666;
                padding: 6px 12px;
                font-size: 12px;
            }

            button.secondary:hover { background: #555; }

            .spinner {
                display: inline-block;
                width: 12px;
                height: 12px;
                border: 2px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 0.6s linear infinite;
            }

            @keyframes spin { to { transform: rotate(360deg); } }

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
            }

            .expand-btn:hover { color: #333; }

            .modal {
                display: none;
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background: rgba(0,0,0,0.5);
                z-index: 1000;
                align-items: center;
                justify-content: center;
            }

            .modal.active { display: flex; }

            .modal-content {
                background: white;
                border-radius: 8px;
                width: 90%;
                max-width: 600px;
                max-height: 80vh;
                display: flex;
                flex-direction: column;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
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
            }

            .modal-close:hover { color: #333; }

            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #999; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="sidebar" id="sidebar">
                <div class="sidebar-header">Context Inspector</div>

                <div class="sidebar-section">
                    <div class="section-title">Token Usage</div>
                    <div class="metric">
                        <span class="metric-label">System prompt:</span>
                        <span class="metric-value" id="systemTokens">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">History (windowed):</span>
                        <span class="metric-value" id="historyTokens">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Payload total:</span>
                        <span class="metric-value" id="payloadTokens">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Context window:</span>
                        <span class="metric-value" id="contextWindow">400</span>
                    </div>
                    <div style="margin-top: 8px;">
                        <div style="font-size: 10px; color: #666; margin-bottom: 2px;">History vs context window</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                        </div>
                        <div style="font-size: 10px; color: #aaa; margin-top: 2px;" id="progressPct">0%</div>
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="section-title">Sliding Window</div>
                    <div class="metric">
                        <span class="metric-label">Window size:</span>
                        <span class="metric-value" id="windowSize">2</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total turns stored:</span>
                        <span class="metric-value" id="totalTurns">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Turns sent to LLM:</span>
                        <span class="metric-value" id="windowedTurns">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Turns dropped:</span>
                        <span class="metric-value warn" id="droppedTurns">0</span>
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="section-title">Summarization</div>
                    <div class="metric">
                        <span class="metric-label">Threshold:</span>
                        <span class="metric-value" id="compressionThreshold">60%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Compressed last turn:</span>
                        <span class="metric-value" id="compressedLastTurn">—</span>
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="section-title">Tool Output Offloading</div>
                    <div class="metric">
                        <span class="metric-label">Tool messages in history:</span>
                        <span class="metric-value" id="toolMsgCount">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Offloaded (stubs):</span>
                        <span class="metric-value good" id="offloadedCount">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Files on disk:</span>
                        <span class="metric-value" id="offloadedFiles">0</span>
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="section-header">
                        <div class="section-title">Payload Sent to LLM</div>
                        <button class="expand-btn" onclick="openPayloadModal()" title="Expand">⤢</button>
                    </div>
                    <div style="font-size: 11px; color: #888; margin-bottom: 6px;">system + windowed history + user message</div>
                    <div class="system-prompt" id="payloadPreview"></div>
                </div>

                <div class="sidebar-section">
                    <div class="section-title">
                        Full Conversation History
                        <span style="font-size:10px; font-weight:400; color:#888;">(not all sent to LLM)</span>
                    </div>
                    <div id="historyList"></div>
                </div>

                <div class="sidebar-section">
                    <button class="secondary" onclick="clearHistory()">Clear History</button>
                </div>
            </div>

            <div class="chat-container">
                <div class="chat-header">
                    <h1>FAQ Agent — Sliding Window + Summarization + Tool Offloading</h1>
                    <p>Try: "What is the leave balance for EMP001?" &nbsp;·&nbsp; "How many people are in Engineering?"</p>
                </div>
                <div class="chat-body" id="chatBody">
                    <div class="chat-message">
                        <div class="message-bubble assistant">
                            Hi! I'm an HR assistant. I can answer policy questions and look up employee or department data. Try asking about leave balances (EMP001–EMP005) or department info.
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
                    <button id="sendBtn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

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
            const API_BASE = window.location.origin;
            let fullPayload = [];

            function roleColor(role) {
                return role === "system" ? "#FF9800" : role === "user" ? "#2196F3" : role === "tool" ? "#9C27B0" : "#4CAF50";
            }

            function openPayloadModal() {
                document.getElementById("payloadModal").classList.add("active");
                const body = document.getElementById("payloadFull");
                body.innerHTML = "";
                fullPayload.forEach(msg => {
                    const wrapper = document.createElement("div");
                    wrapper.style.cssText = "margin-bottom: 16px; border-left: 3px solid " + roleColor(msg.role) + "; padding-left: 10px;";
                    wrapper.innerHTML =
                        '<div style="font-weight:700; font-size:11px; text-transform:uppercase; color:#888; margin-bottom:4px;">' +
                        escapeHtml(msg.role) + ' <span style="font-weight:400; color:#bbb;">(' + msg.tokens + ' tokens)</span></div>' +
                        '<pre style="margin:0; white-space:pre-wrap; word-wrap:break-word; font-size:12px; color:#444; line-height:1.5;">' +
                        escapeHtml(msg.content || "") + '</pre>';
                    body.appendChild(wrapper);
                });
            }

            function closePayloadModal(event) {
                if (event && event.target.id !== "payloadModal") return;
                document.getElementById("payloadModal").classList.remove("active");
            }

            document.addEventListener("keydown", e => { if (e.key === "Escape") closePayloadModal(); });

            async function sendMessage() {
                const input  = document.getElementById("chatInput");
                const btn    = document.getElementById("sendBtn");
                const message = input.value.trim();
                if (!message) return;

                addMessageToChat("user", message);
                input.value = "";
                input.focus();

                btn.disabled = true;
                btn.innerHTML = '<span class="spinner"></span>';

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
                        addMessageToChat("assistant", "Error: " + data.error);
                    }
                } catch (err) {
                    addMessageToChat("assistant", "Error: " + err.message);
                } finally {
                    btn.disabled = false;
                    btn.textContent = "Send";
                }
            }

            function addMessageToChat(role, content) {
                const chatBody = document.getElementById("chatBody");
                const div = document.createElement("div");
                div.className = "chat-message " + role;
                const bubble = document.createElement("div");
                bubble.className = "message-bubble " + role;
                bubble.textContent = content;
                div.appendChild(bubble);
                chatBody.appendChild(div);
                chatBody.scrollTop = chatBody.scrollHeight;
            }

            function updateContext(ctx) {
                document.getElementById("systemTokens").textContent         = ctx.system_tokens.toLocaleString();
                document.getElementById("historyTokens").textContent        = ctx.history_tokens.toLocaleString();
                document.getElementById("payloadTokens").textContent        = ctx.full_payload_tokens.toLocaleString();
                document.getElementById("contextWindow").textContent        = ctx.context_window.toLocaleString();
                document.getElementById("windowSize").textContent           = ctx.window_size;
                document.getElementById("totalTurns").textContent           = ctx.total_turns;
                document.getElementById("windowedTurns").textContent        = ctx.windowed_turns;
                document.getElementById("droppedTurns").textContent         = ctx.total_turns - ctx.windowed_turns;
                document.getElementById("compressionThreshold").textContent = ctx.compression_threshold + "%";
                document.getElementById("toolMsgCount").textContent         = ctx.tool_msg_count;
                document.getElementById("offloadedCount").textContent       = ctx.offloaded_count;
                document.getElementById("offloadedFiles").textContent       = ctx.offloaded_files_on_disk;

                const compressed = ctx.compressed_last_turn;
                const compEl = document.getElementById("compressedLastTurn");
                compEl.textContent = compressed ? "Yes" : "No";
                compEl.className = "metric-value " + (compressed ? "warn" : "muted");

                const pct = Math.min(100, Math.round(ctx.history_tokens / ctx.context_window * 100));
                document.getElementById("progressFill").style.width = pct + "%";
                document.getElementById("progressPct").textContent  = pct + "%";

                // Payload preview
                fullPayload = ctx.full_payload || [];
                const preview = fullPayload.map(m =>
                    "[" + m.role.toUpperCase() + "]\\n" + (m.content || "").substring(0, 80) + ((m.content || "").length > 80 ? "..." : "")
                ).join("\\n\\n");
                document.getElementById("payloadPreview").textContent = preview;

                // Full history list
                const conversation = ctx.conversation || [];
                const windowedCount = ctx.windowed_history_messages || 0;
                const droppedCount  = conversation.length - windowedCount;
                document.getElementById("historyList").innerHTML = conversation.map((msg, i) => {
                    const isDropped    = i < droppedCount;
                    const isOffloaded  = msg.is_offloaded;
                    const isTool       = msg.is_tool;
                    let badges = "";
                    if (isDropped)   badges += ' <span class="badge badge-dropped">dropped</span>';
                    if (isTool)      badges += ' <span class="badge badge-tool">tool</span>';
                    if (isOffloaded) badges += ' <span class="badge badge-offloaded">offloaded</span>';

                    return `<div class="message-item ${msg.role}" style="${isDropped ? "opacity:0.35;" : ""}">
                        <div class="message-role">${msg.role}${badges}</div>
                        <div class="message-content">${escapeHtml(msg.content)}</div>
                        <div class="message-tokens">${msg.tokens} tokens</div>
                    </div>`;
                }).join("");
            }

            function escapeHtml(text) {
                const div = document.createElement("div");
                div.textContent = text || "";
                return div.innerHTML;
            }

            async function clearHistory() {
                if (!confirm("Clear conversation history?")) return;
                await fetch(`${API_BASE}/api/clear`, { method: "POST" });
                document.getElementById("chatBody").innerHTML = `
                    <div class="chat-message">
                        <div class="message-bubble assistant">
                            Hi! I'm an HR assistant. I can answer policy questions and look up employee or department data. Try asking about leave balances (EMP001–EMP005) or department info.
                        </div>
                    </div>`;
                ["systemTokens","historyTokens","payloadTokens","totalTurns","windowedTurns",
                 "droppedTurns","toolMsgCount","offloadedCount","offloadedFiles"].forEach(id => {
                    document.getElementById(id).textContent = "0";
                });
                document.getElementById("progressFill").style.width = "0%";
                document.getElementById("progressPct").textContent  = "0%";
                document.getElementById("compressedLastTurn").textContent = "—";
                document.getElementById("historyList").innerHTML = "";
                document.getElementById("payloadPreview").textContent = "";
            }

            function handleEnter(event) {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            }

            async function loadInitialContext() {
                try {
                    const res = await fetch(`${API_BASE}/api/context`);
                    const ctx = await res.json();
                    updateContext(ctx);
                } catch (e) {
                    console.error("Failed to load initial context:", e);
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
