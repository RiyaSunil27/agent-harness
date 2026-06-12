# FAQ Agent — Token-Threshold Summarization

Context management via LLM summarization: when history token count exceeds a threshold, the model compresses the conversation into a summary message. Keeps context bounded while preserving more information than a fixed window.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI key to `.env`:

```
OPENAI_API_KEY=sk-...
```

## Run

**CLI:**
```bash
python agent.py
```

**Web UI:**
```bash
python app.py
```
Open http://localhost:5000 in browser.

## The Approach

`should_compress()` in `agent.py` monitors token count each turn. When history exceeds `COMPRESSION_THRESHOLD × CONTEXT_WINDOW` tokens, it calls `summarize_history()` to compress the full history into one summary message.

```python
CONTEXT_WINDOW        = 600   # token budget for history
COMPRESSION_THRESHOLD = 0.6   # compress at 60% of window
```

## The Tradeoff

- Context stays within a token budget
- Summarization is **lossy** — specific details from early turns may not survive
- Extra LLM call on compression events (latency + cost)

## Exercise

`summarize_history()` and `should_compress()` have TODOs. Implement them:

1. Build a transcript string from the history messages.
2. Write a system prompt that asks the model to summarize the conversation.
3. In `should_compress()`, count tokens and trigger compression when over threshold.
4. Return `(compressed_history, True)` when compressed, `(history, False)` otherwise.

Check `solution/agent.py` for a reference implementation.

## Files

- `agent.py` — CLI agent with summarization TODOs
- `app.py` — Flask backend + frontend with context visualization
- `policies.md` — Sample HR policy document
- `solution/` — Reference implementation
