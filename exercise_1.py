"""
FAQ Agent — Sliding Window  (Exercise 1)

Problem: the agent can only "see" the last WINDOW_SIZE turns.
Ask it something, then ask about it again 3+ turns later — it forgets.

Your task: change WINDOW_SIZE below and observe the trade-off
between memory depth and token usage in the sidebar.
No code to write — just experiment!
"""

import litellm
from lib.config import MODEL, API_KEY, API_BASE, MAX_TOKENS
from lib.utils  import load_document, build_system_prompt

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DOCUMENT = "policies.md"
WINDOW_SIZE     = 1               # ← try 3, 5, 10 — watch the sidebar change
# ──────────────────────────────────────────────────────────────────────────────

EXERCISE_META = {
    "number":                1,
    "title":                 "Sliding Window",
    "features":              ["sliding_window"],
    "window_size":           WINDOW_SIZE,
    "context_window":        800,
    "compression_threshold": None,
    "has_tools":             False,
    "strategy":              "Only the last WINDOW_SIZE message pairs are sent to the model. Older turns are silently dropped.",
    "problem":               "Ask the agent about something from 3+ turns ago — it has no memory of it.",
    "task":                  "Change WINDOW_SIZE (line 16) and re-run. Observe the trade-off between memory and token usage.",
    "todos":                 [],
}


def trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last WINDOW_SIZE user+assistant pairs."""
    keep = WINDOW_SIZE * 2
    return history[-keep:] if len(history) > keep else history


def chat(system: str, history: list[dict], user_input: str) -> tuple[str, list[dict], dict]:
    windowed = trim_history(history)
    messages  = [{"role": "system", "content": system}] + windowed + [{"role": "user", "content": user_input}]

    response = litellm.completion(
        model      = MODEL,
        max_tokens = MAX_TOKENS,
        messages   = messages,
        api_key    = API_KEY,
        api_base   = API_BASE,
    )
    reply = response.choices[0].message.content
    new_history = history + [
        {"role": "user",      "content": user_input},
        {"role": "assistant", "content": reply},
    ]
    return reply, new_history, {"compressed": False, "tool_count": 0, "offloaded_count": 0}


def main():
    document = load_document(POLICY_DOCUMENT)
    system   = build_system_prompt(document)
    history: list[dict] = []

    print("=== FAQ Agent (Sliding Window) ===")
    print(f"Window size : {WINDOW_SIZE} pair(s)  ← change WINDOW_SIZE to see the difference")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply, history, _ = chat(system, history, user_input)
        print(f"\nAgent: {reply}\n")
        print(f"[history: {len(history)} messages total | sending last {WINDOW_SIZE * 2} to model]\n")


if __name__ == "__main__":
    main()
