"""
FAQ Agent — token-threshold lossy summarization (SOLUTION).
"""

import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DOCUMENT       = "policies.md"   # swap this to use a different document
MODEL                 = "gpt-3.5-turbo"
MAX_TOKENS            = 512
CONTEXT_WINDOW        = 600             # artificially small to trigger compression quickly
COMPRESSION_THRESHOLD = 0.6             # compress when history hits this fraction of window
# ──────────────────────────────────────────────────────────────────────────────

_encoder = tiktoken.encoding_for_model(MODEL)


def load_document(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


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


def count_tokens(messages: list[dict]) -> int:
    """Approximate token count for a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        total += len(_encoder.encode(msg["content"]))
    return total


def summarize_history(client: OpenAI, history: list[dict]) -> list[dict]:
    """Call the LLM to compress history into a single summary message."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
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


def chat(client: OpenAI, system: str, history: list[dict], user_input: str) -> tuple[str, list[dict], bool]:
    history, compressed = maybe_compress(client, history)
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_input}]

    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = MAX_TOKENS,
        messages   = messages,
    )
    return response.choices[0].message.content, history, compressed


def main():
    client   = OpenAI()
    document = load_document(POLICY_DOCUMENT)
    system   = build_system_prompt(document)
    history: list[dict] = []

    print("=== FAQ Agent (Token-Threshold Compression) ===")
    print(f"Policy document       : {POLICY_DOCUMENT}")
    print(f"Model                 : {MODEL}")
    print(f"Context window        : {CONTEXT_WINDOW} tokens  (artificially small)")
    print(f"Compression threshold : {int(COMPRESSION_THRESHOLD * 100)}%  ({int(CONTEXT_WINDOW * COMPRESSION_THRESHOLD)} tokens)")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply, history, compressed = chat(client, system, history, user_input)

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": reply})

        print(f"\nAgent: {reply}\n")
        tokens = count_tokens(history)
        print(f"[tokens in history: {tokens} / {CONTEXT_WINDOW}  |  compressed this turn: {compressed}]\n")


if __name__ == "__main__":
    main()
