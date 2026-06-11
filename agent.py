"""
FAQ Agent — sliding window context.
Only the last N message pairs are sent to the model.
Problem: ask about something outside the window and the agent forgets.
Exercise: increase WINDOW_SIZE and see how more context helps recall.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DOCUMENT = "policies.md"   # swap this to use a different document
MODEL           = "gpt-3.5-turbo"
MAX_TOKENS      = 512
WINDOW_SIZE     = 1               # keep last N user+assistant pairs — try increasing this!
# ──────────────────────────────────────────────────────────────────────────────


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


def trim_history(history: list[dict], window_size: int) -> list[dict]:
    """Sliding window: keep only the last `window_size` user+assistant pairs."""
    keep = window_size * 2
    return history[-keep:] if len(history) > keep else history


def chat(client: OpenAI, system: str, history: list[dict], user_input: str) -> str:
    windowed = trim_history(history, WINDOW_SIZE)
    messages  = [{"role": "system", "content": system}] + windowed + [{"role": "user", "content": user_input}]

    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = MAX_TOKENS,
        messages   = messages,
    )
    return response.choices[0].message.content


def main():
    client   = OpenAI()
    document = load_document(POLICY_DOCUMENT)
    system   = build_system_prompt(document)
    history: list[dict] = []

    print("=== FAQ Agent (Sliding Window) ===")
    print(f"Policy document : {POLICY_DOCUMENT}")
    print(f"Model           : {MODEL}")
    print(f"Window size     : {WINDOW_SIZE} pair(s)  ← change WINDOW_SIZE to see the difference")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply = chat(client, system, history, user_input)

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": reply})

        print(f"\nAgent: {reply}\n")
        print(f"[history: {len(history)} messages total | sending last {WINDOW_SIZE * 2} to model]\n")


if __name__ == "__main__":
    main()
