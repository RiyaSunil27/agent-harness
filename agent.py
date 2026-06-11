"""
FAQ Agent — token-threshold lossy summarization.
When history tokens exceed a threshold, the LLM compresses history into a summary.
Problem: summarization is lossy — earlier details may not survive.
Exercise: implement summarize_history() and maybe_compress() below.
"""

import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DOCUMENT       = "policies.md"  
MODEL                 = "gpt-3.5-turbo"
MAX_TOKENS            = 512
CONTEXT_WINDOW        = 600            
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
    
    """
    TODO:
     1. Build a transcript string from history (role + content for each message).
    """
    transcript = ''
    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = 300,
        messages   = [
            {
                "role": "system",
                # "content": (
                    # TODO: Write a system prompt to summarize the conversation
                    # ),
            },
            {"role": "user", "content": transcript},
        ],
    )
    summary = response.choices[0].message.content
    return [{"role": "assistant", "content": f"[Summary of earlier conversation]: {summary}"}]

def should_compress(client: OpenAI, history: list[dict]) -> tuple[list[dict], bool]:
    """
    Check token count and compress history if over threshold.
    Returns (history, did_compress).

    TODO: implement this function.
      1. Compute the token budget: CONTEXT_WINDOW * COMPRESSION_THRESHOLD (cast to int).
      2. If count_tokens(history) >= token_budget, call summarize_history() and return
         (summary, True).
      3. Otherwise return (history, False).
    """
    raise NotImplementedError("TODO: implement maybe_compress()")


def chat(client: OpenAI, system: str, history: list[dict], user_input: str) -> tuple[str, list[dict], bool]:
    history, compressed = should_compress(client, history)
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
