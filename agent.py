"""
FAQ Agent — base version.
No context management. Full history sent every turn.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DOCUMENT = "policies.md"   # swap this to use a different document
MODEL           = "gpt-3.5-turbo"
MAX_TOKENS      = 512
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


def chat(client: OpenAI, system: str, history: list[dict], user_input: str) -> str:
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_input}]

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

    print("=== FAQ Agent ===")
    print(f"Policy document : {POLICY_DOCUMENT}")
    print(f"Model           : {MODEL}")
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
        print(f"[history: {len(history)} messages]\n")


if __name__ == "__main__":
    main()
