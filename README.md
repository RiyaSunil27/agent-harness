# FAQ Agent — Context Visualization Demo

Visual demo showing how context bloat occurs when sending full conversation history to LLM on every request.

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open http://localhost:5000 in browser.

## Features

**Left Sidebar (Context Inspector)**
- **Token Usage**: System prompt tokens + history tokens + total
- **Progress bar**: Shows tokens as % of max tokens per response
- **Message Count**: How many messages in history
- **System Prompt**: Full text of system prompt
- **Conversation History**: Every message sent/received with token count

**Right Side (Chat Interface)**
- Normal chat UI to interact with agent
- Sends message + gets response
- Chat displays realtime as you talk

## The Problem

Every time you send a message, the system:
1. Loads the full policy document (~600 tokens)
2. Builds system prompt with document (~611 tokens)
3. Adds entire conversation history (~grows each turn)
4. Sends all of it to OpenAI API

After 10-20 exchanges, you're sending thousands of tokens that are just repetition. You can see this happen in real-time on the left sidebar.

**Token count after each message in this demo:**
- After msg 1: 705 total (94 history)
- After msg 2: 754 total (143 history)
- After msg 3: 823 total (212 history)

## Solution Ideas

The agent harness in `agent-harness/` explores several approaches:
- Context windowing (keep only last N messages)
- Summarization (compress old context)
- Semantic search (retrieve only relevant messages)
- Hierarchical memory (semantic + token counting)

## Files

- `agent.py` — Original CLI agent (full history sent every turn)
- `app.py` — Flask backend + frontend with context visualization
- `policies.md` — Sample HR policy document
- `agent-harness/` — Experiments with context management strategies
