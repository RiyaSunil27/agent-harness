"""
FAQ Agent — Tool Output Offloading  (SOLUTION)
"""

import json
import os
import litellm
from lib.config   import MODEL, API_KEY, API_BASE, MAX_TOKENS
from lib.utils    import load_document
from lib.hr_data  import TOOLS, execute_tool
from lib.context  import count_tokens, trim_history, maybe_compress

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DOCUMENT       = "policies.md"
CONTEXT_WINDOW        = 400
COMPRESSION_THRESHOLD = 0.6
WINDOW_SIZE           = 2
TOOL_RESULTS_DIR      = "tool_results"
# ──────────────────────────────────────────────────────────────────────────────

EXERCISE_META = {
    "number":                3,
    "title":                 "Tool Output Offloading",
    "features":              ["sliding_window", "summarization", "tool_offloading"],
    "window_size":           WINDOW_SIZE,
    "context_window":        CONTEXT_WINDOW,
    "compression_threshold": COMPRESSION_THRESHOLD,
    "has_tools":             True,
    "strategy":              "Old tool results are saved to disk and replaced with short stubs. Sliding window + summarization also active.",
    "problem":               "Tool results are large JSON blobs. Multiple tool calls across turns eat the context window quickly.",
    "task":                  "Implement offload_tool_result() and compact_tool_messages(). Watch the sidebar show offloaded stubs.",
    "todos":                 ["offload_tool_result()", "compact_tool_messages()"],
}


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


def offload_tool_result(tool_call_id: str, content: str) -> str:
    os.makedirs(TOOL_RESULTS_DIR, exist_ok=True)
    path = f"{TOOL_RESULTS_DIR}/{tool_call_id}.json"
    with open(path, "w") as f:
        json.dump({"tool_call_id": tool_call_id, "content": content}, f)
    return f"[offloaded to disk: {tool_call_id}]"


def compact_tool_messages(messages: list[dict]) -> list[dict]:
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= 1:
        return messages
    for i in tool_indices[:-1]:
        stub = offload_tool_result(messages[i]["tool_call_id"], messages[i]["content"])
        messages[i] = {**messages[i], "content": stub}
    return messages


def chat(system: str, history: list[dict], user_input: str) -> tuple[str, list[dict], dict]:
    history = trim_history(history, WINDOW_SIZE)
    history, compressed = maybe_compress(
        history, CONTEXT_WINDOW, COMPRESSION_THRESHOLD, MODEL, API_KEY, API_BASE
    )

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_input}]

    while True:
        response = litellm.completion(
            model      = MODEL,
            max_tokens = MAX_TOKENS,
            messages   = messages,
            tools      = TOOLS,
            api_key    = API_KEY,
            api_base   = API_BASE,
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
            new_history     = messages[1:]
            tool_count      = sum(1 for m in new_history if m.get("role") == "tool")
            offloaded_count = sum(
                1 for m in new_history
                if m.get("role") == "tool" and str(m.get("content", "")).startswith("[offloaded")
            )
            return msg.content, new_history, {
                "compressed":      compressed,
                "tool_count":      tool_count,
                "offloaded_count": offloaded_count,
            }


def main():
    document = load_document(POLICY_DOCUMENT)
    system   = build_system_prompt(document)
    history: list[dict] = []

    print("=== FAQ Agent (Tool Output Offloading) — SOLUTION ===")
    print("Try: 'What is the leave balance for EMP001?'")
    print("     'How many people are in Engineering?'\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply, history, metadata = chat(system, history, user_input)
        print(f"\nAgent: {reply}\n")
        tokens = count_tokens(history)
        print(
            f"[tokens: {tokens}/{CONTEXT_WINDOW}"
            f"  |  tool: {metadata['tool_count']} ({metadata['offloaded_count']} offloaded)"
            f"  |  compressed: {metadata['compressed']}]\n"
        )


if __name__ == "__main__":
    main()
