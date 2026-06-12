# FAQ Agent — Base Version

Starting point for the context management training series. No context management — full conversation history sent to the LLM on every request.

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

```bash
python agent.py
```

## The Problem

Every turn, the agent sends:
1. Full system prompt with policy document (~611 tokens)
2. Entire conversation history (grows each turn)

After 10–20 exchanges, you're sending thousands of redundant tokens per request.

## Branches

Each branch in this repo adds a different context management strategy:

| Branch | Strategy |
|--------|----------|
| `master` | No management — baseline |
| `1-sliding-window` | Keep only last N message pairs |
| `2-summarization` | Compress history when token count exceeds threshold |
| `3-tool-output-offloading` | Offload large tool outputs out of context |

## Files

- `agent.py` — CLI agent, full history every turn
- `policies.md` — Sample HR policy document used as context
- `solution/` — Reference implementations
