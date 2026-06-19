"""
Context management helpers used by Exercise 3.
Not part of the exercise — provided so exercise_3.py stays readable.
"""

import tiktoken
import litellm

_encoder = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(messages: list[dict]) -> int:
    """Count tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += len(_encoder.encode(content))
    return total


def trim_history(history: list[dict], window_size: int) -> list[dict]:
    """
    Sliding window: keep only the last `window_size` user-turn anchors
    and all messages that follow them (assistant replies, tool calls, etc.).
    """
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indices) <= window_size:
        return history
    cutoff = user_indices[-window_size]
    return history[cutoff:]


def summarize_history(
    history: list[dict],
    model: str,
    api_key: str,
    api_base: str,
) -> list[dict]:
    """
    Call the LLM to compress `history` into a single summary message.
    Returns a one-element list: [{"role": "assistant", "content": "[Summary …]"}]
    """
    transcript = "\n".join(
        f"{m['role'].upper()}: {m.get('content') or '[tool call]'}" for m in history
    )
    response = litellm.completion(
        model    = model,
        max_tokens = 300,
        api_key  = api_key,
        api_base = api_base,
        messages = [
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


def maybe_compress(
    history: list[dict],
    context_window: int,
    compression_threshold: float,
    model: str,
    api_key: str,
    api_base: str,
) -> tuple[list[dict], bool]:
    """
    Compress history if its token count exceeds the threshold fraction of the context window.
    Returns (history, did_compress).
    """
    token_budget = int(context_window * compression_threshold)
    if count_tokens(history) >= token_budget:
        return summarize_history(history, model, api_key, api_base), True
    return history, False
