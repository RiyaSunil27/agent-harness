# FAQ Agent — Sliding Window Context

Context management via sliding window: only the last N message pairs are sent to the model. Simple and fast, but the agent forgets anything outside the window.

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

`trim_history()` in `agent.py` keeps only the last `WINDOW_SIZE` user+assistant pairs before each API call.

```python
WINDOW_SIZE = 1  # keep last N user+assistant pairs
```

Try increasing `WINDOW_SIZE` and observe how recall improves — at the cost of more tokens per request.

## The Tradeoff

- Tokens per request are bounded (predictable cost)
- Agent forgets anything older than the window
- Ask about something said 5 turns ago → wrong answer

## Exercise

1. Set `WINDOW_SIZE = 1`, have a multi-turn conversation, then ask about the first thing you said.
2. Increase `WINDOW_SIZE` until recall improves.
3. Check the Context Inspector in the web UI to see token usage change.

## Files

- `agent.py` — CLI agent with sliding window
- `app.py` — Flask backend + frontend with context visualization
- `policies.md` — Sample HR policy document
- `solution/` — Reference implementation
