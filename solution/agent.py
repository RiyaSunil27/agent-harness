"""
FAQ Agent — sliding window + token-threshold summarization + tool output offloading (SOLUTION).
"""

import json
import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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
# ──────────────────────────────────────────────────────────────────────────────

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

_encoder = tiktoken.encoding_for_model(MODEL)


def load_document(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


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


def count_tokens(messages: list[dict]) -> int:
    """Approximate token count for a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += len(_encoder.encode(content))
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

def trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last WINDOW_SIZE user-turn anchors and everything after them."""
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indices) <= WINDOW_SIZE:
        return history
    cutoff = user_indices[-WINDOW_SIZE]
    return history[cutoff:]


# ── Summarization ──────────────────────────────────────────────────────────────

def summarize_history(client: OpenAI, history: list[dict]) -> list[dict]:
    """Call the LLM to compress history into a single summary message."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m.get('content') or '[tool call]'}" for m in history
    )
    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = 300,
        messages   = [
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


def maybe_compress(client: OpenAI, history: list[dict]) -> tuple[list[dict], bool]:
    """Return (possibly compressed history, did_compress)."""
    token_budget = int(CONTEXT_WINDOW * COMPRESSION_THRESHOLD)
    if count_tokens(history) >= token_budget:
        return summarize_history(client, history), True
    return history, False


# ── Tool output offloading ─────────────────────────────────────────────────────

def offload_tool_result(tool_call_id: str, content: str) -> str:
    """Save tool result to disk. Return a short stub string."""
    os.makedirs(TOOL_RESULTS_DIR, exist_ok=True)
    path = os.path.join(TOOL_RESULTS_DIR, f"{tool_call_id}.json")
    with open(path, "w") as f:
        json.dump({"tool_call_id": tool_call_id, "content": content}, f)
    return f"[offloaded to disk: {tool_call_id}]"


def compact_tool_messages(messages: list[dict]) -> list[dict]:
    """Offload all tool results except the most recent one."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_indices[:-1]:  # all but the last
        msg = messages[i]
        stub = offload_tool_result(msg["tool_call_id"], msg["content"])
        messages[i] = {**msg, "content": stub}
    return messages


# ── Chat ───────────────────────────────────────────────────────────────────────

def chat(client: OpenAI, system: str, history: list[dict], user_input: str) -> tuple[str, list[dict], bool]:
    history = trim_history(history)
    history, compressed = maybe_compress(client, history)

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_input}]

    while True:
        response = client.chat.completions.create(
            model      = MODEL,
            max_tokens = MAX_TOKENS,
            messages   = messages,
            tools      = TOOLS,
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
            messages = compact_tool_messages(messages)
        else:
            new_history = messages[1:]  # strip system prompt
            return msg.content, new_history, compressed


def main():
    client   = OpenAI()
    document = load_document(POLICY_DOCUMENT)
    system   = build_system_prompt(document)
    history: list[dict] = []

    print("=== FAQ Agent (Sliding Window + Summarization + Tool Offloading) — SOLUTION ===")
    print(f"Policy document       : {POLICY_DOCUMENT}")
    print(f"Model                 : {MODEL}")
    print(f"Context window        : {CONTEXT_WINDOW} tokens  (artificially small)")
    print(f"Compression threshold : {int(COMPRESSION_THRESHOLD * 100)}%  ({int(CONTEXT_WINDOW * COMPRESSION_THRESHOLD)} tokens)")
    print(f"Window size           : {WINDOW_SIZE} message pairs")
    print(f"Tool results dir      : {TOOL_RESULTS_DIR}/")
    print("Type 'quit' to exit.\n")
    print("Try: 'What is the leave balance for EMP001?'")
    print("     'How many people are in Engineering?'")
    print("     'What is the leave policy for sick leave?'\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply, history, compressed = chat(client, system, history, user_input)

        print(f"\nAgent: {reply}\n")
        tokens = count_tokens(history)
        tool_msg_count = sum(1 for m in history if m.get("role") == "tool")
        offloaded_count = sum(
            1 for m in history
            if m.get("role") == "tool" and str(m.get("content", "")).startswith("[offloaded")
        )
        print(
            f"[tokens: {tokens} / {CONTEXT_WINDOW}"
            f"  |  tool messages: {tool_msg_count} ({offloaded_count} offloaded)"
            f"  |  compressed: {compressed}]\n"
        )


if __name__ == "__main__":
    main()
