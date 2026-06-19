"""
FAQ Agent — Summarization  (Exercise 2)

Problem: summarization is lossy — earlier details may not survive compression.
Chat until the sidebar shows compression triggered, then ask about an early detail.

Your task: implement summarize_history() and should_compress() below.
"""

import litellm
from lib.config  import MODEL, API_KEY, API_BASE, MAX_TOKENS
from lib.utils   import load_document, build_system_prompt
from lib.context import count_tokens

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DOCUMENT       = "policies.md"
CONTEXT_WINDOW        = 600   # artificially small — triggers compression quickly
COMPRESSION_THRESHOLD = 0.6   # compress when history reaches 60% of context window
# ──────────────────────────────────────────────────────────────────────────────

EXERCISE_META = {
    "number":                2,
    "title":                 "Summarization",
    "features":              ["summarization"],
    "window_size":           0,
    "context_window":        CONTEXT_WINDOW,
    "compression_threshold": COMPRESSION_THRESHOLD,
    "has_tools":             False,
    "strategy":              "When history tokens exceed a threshold, the LLM compresses the entire history into one summary message.",
    "problem":               "Summarization is lossy — specific details from earlier turns may not survive compression.",
    "task":                  "Implement summarize_history() and should_compress(). Watch the sidebar show compression events.",
    "todos":                 ["summarize_history()", "should_compress()"],
}


# ── TODO 1 ─────────────────────────────────────────────────────────────────────

def summarize_history(history: list[dict]) -> list[dict]:
    """Call the LLM to compress history into a single summary message.

    Steps:
      1. Build a transcript string — for each message:
             f"{msg['role'].upper()}: {msg.get('content') or '[tool call]'}"
      2. Call litellm.completion() with model=MODEL, max_tokens=300,
         api_key=API_KEY, api_base=API_BASE.
         Use a system prompt that tells the LLM to summarize concisely.
      3. Return: [{"role": "assistant", "content": f"[Summary of earlier conversation]: {summary}"}]
    """
    transcript = ""
    # TODO: build transcript from history

    response = litellm.completion(
        model      = MODEL,
        max_tokens = 300,
        api_key    = API_KEY,
        api_base   = API_BASE,
        messages   = [
            {
                "role":    "system",
                "content": "",  # TODO: write a summarization system prompt
            },
            {"role": "user", "content": transcript},
        ],
    )
    summary = response.choices[0].message.content
    return [{"role": "assistant", "content": f"[Summary of earlier conversation]: {summary}"}]


# ── TODO 2 ─────────────────────────────────────────────────────────────────────

def should_compress(history: list[dict]) -> tuple[list[dict], bool]:
    """Check token count; compress history if over threshold.

    Steps:
      1. token_budget = int(CONTEXT_WINDOW * COMPRESSION_THRESHOLD)
      2. If count_tokens(history) >= token_budget → return (summarize_history(history), True)
      3. Otherwise → return (history, False)
    """
    raise NotImplementedError("TODO: implement should_compress()")


# ── Chat loop (read to understand, no changes needed) ──────────────────────────

def chat(system: str, history: list[dict], user_input: str) -> tuple[str, list[dict], dict]:
    history, compressed = should_compress(history)
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_input}]

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
    return reply, new_history, {"compressed": compressed, "tool_count": 0, "offloaded_count": 0}


def main():
    document = load_document(POLICY_DOCUMENT)
    system   = build_system_prompt(document)
    history: list[dict] = []

    print("=== FAQ Agent (Summarization) ===")
    print(f"Context window        : {CONTEXT_WINDOW} tokens")
    print(f"Compression threshold : {int(COMPRESSION_THRESHOLD * 100)}%  ({int(CONTEXT_WINDOW * COMPRESSION_THRESHOLD)} tokens)")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply, history, metadata = chat(system, history, user_input)
        print(f"\nAgent: {reply}\n")
        tokens = count_tokens(history)
        print(f"[tokens: {tokens}/{CONTEXT_WINDOW}  |  compressed this turn: {metadata['compressed']}]\n")


if __name__ == "__main__":
    main()
