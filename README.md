---
title: Customer Support OpenEnv
emoji: ??
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# 🎧 Customer Support OpenEnv

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-brightgreen)](https://github.com/openenv)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **OpenEnv Hackathon submission** — a production-grade customer support simulation environment for training and evaluating LLM-based support agents.

---

## 🌍 Why This Environment?

Customer support is one of the most common enterprise AI use cases, yet no standard RL/agent benchmark exists for it.  
This environment fills that gap: it provides a **realistic, multi-difficulty, fully graded** benchmark where agents must:

- 🏷️ **Classify** support tickets into the correct category  
- ✍️ **Respond** empathetically and accurately to customer issues  
- 🗣️ **Hold multi-turn conversations** — clarify, resolve, and close tickets  

All graded automatically with clear, deterministic rubrics (0.0–1.0) and meaningful **partial rewards** on every turn.

---

## ⚙️ How It Works — Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AGENT LOOP                                  │
│                                                                     │
│   ┌──────────┐   POST /reset   ┌─────────────────────────────────┐  │
│   │          │ ──────────────► │         FastAPI Server          │  │
│   │  Agent   │                 │         (server/app.py)         │  │
│   │ (client) │ ◄────────────── │                                 │  │
│   │          │   Observation   │  ┌──────────────────────────┐   │  │
│   │          │                 │  │   SupportEnvironment     │   │  │
│   │          │   POST /step    │  │   (environment.py)       │   │  │
│   │          │ ──────────────► │  │                          │   │  │
│   │          │   obs+reward    │  │  15 scenarios × 5 types  │   │  │
│   │          │ ◄────────────── │  │  3 graders (easy/med/hrd)│   │  │
│   └──────────┘                 │  │  Cumulative reward logic │   │  │
│                                │  └──────────────────────────┘   │  │
│   GET /state  ──────────────►  │                                 │  │
│   POST /grader ─────────────►  │  Score + turn breakdown         │  │
│   GET /tasks  ──────────────►  │  Task list + action schemas     │  │
│   POST /baseline ───────────►  │  Built-in oracle agent scores   │  │
└───────────────────────────────────────────────────────────────────-─┘
```

**Episode lifecycle:**
1. Agent calls `POST /reset` → gets the opening customer message
2. Agent sends replies via `POST /step` → gets observation + reward each turn
3. Agent calls `POST /grader` → gets full score breakdown with `turn_scores`
4. Episode ends when `done: true` in the observation

---

## 🗂️ Environment Description

Simulates a customer support system across **5 issue categories** with **15 unique scenarios** (3 per category):

| Category | Sample Scenarios |
|---|---|
| **Refund** | Missing delivery refund, damaged product return, cancelled-order refund delay |
| **Technical** | App crash after update, Slack webhook error, PDF export blank |
| **Shipping** | 3-week missing order, wrong-door delivery, split shipment partial arrival |
| **Billing** | Duplicate subscription charge, unauthorized upgrade charge, invoice ghost charge |
| **Account** | Password reset email missing, account suspended, email address transfer |

---

## 📥 Action Space

Each agent step submits a `SupportAction`:

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | `str` | ✅ | The agent's text reply to the customer |
| `intent` | `str` | ❌ | Declared intent: `"classify"`, `"respond"`, `"clarify"`, `"escalate"`, or `"close"` |

---

## 📤 Observation Space

The environment returns a `SupportObservation` after each step:

| Field | Type | Description |
|---|---|---|
| `conversation` | `List[str]` | Full message history (alternating customer / agent) |
| `customer_query` | `str` | The latest customer message the agent must address |
| `task_name` | `str` | Difficulty tier: `"easy"`, `"medium"`, or `"hard"` |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float \| None` | Step-level reward (0.0–1.0); `None` on opening observation |
| `cumulative_reward` | `float` | Running average reward across all turns so far |
| `turn_scores` | `List[float]` | Per-turn reward breakdown (useful for analysis) |
| `info` | `str \| None` | Optional context or hints for the agent |

---

## 📋 Tasks

### 🟢 Easy — Ticket Classification
**Objective:** Output the correct issue category for a customer message.

**Scoring:**
- ✅ `1.0` — Exact category in reply (`refund`, `billing`, etc.)
- 🟡 `0.5` — Partial keyword match  
- ❌ `0.0` — Wrong or missing category
- Max steps: **1**

---

### 🟡 Medium — Single-Turn Response
**Objective:** Write a complete, empathetic reply resolving the customer's issue.

**Scoring rubric:**
| Component | Reward |
|---|---|
| +0.20 per matching keyword (max 4) | 0.00–0.80 |
| Empathy detected (apologize/sorry/understand) | +0.10 |
| Reply length > 80 chars | +0.10 |
| Unnecessary escalation penalty | −0.20 |

All scores clamped to `[0.0, 1.0]`. Max steps: **1**

---

### 🔴 Hard — Multi-Turn Conversation
**Objective:** Handle a 3-turn dialogue: clarify → resolve → close.

| Turn | Behaviour | Max Reward |
|---|---|---|
| 1 (clarify) | Ask a `?` question (bonus if on-topic) | +0.40 |
| 2 (resolve) | Keyword + empathy + detail | +0.50 |
| 3 (close) | Polite closing phrase | +0.30 |

`cumulative_reward = mean(turn_scores)`, clamped to `[0.0, 1.0]`. Max steps: **10**

---

## 🎯 Reward Function Design

| Signal | When | Magnitude |
|---|---|---|
| Correct classification | Easy, step 1 | +1.0 / +0.5 / 0.0 |
| Keyword coverage | Medium & Hard turn 2 | +0.20 per keyword |
| Empathy language | Medium & Hard turn 2 | +0.10–+0.12 |
| Response detail (length) | Medium & Hard turn 2 | +0.08–+0.10 |
| Clarifying question | Hard turn 1 | +0.30 (+0.10 bonus) |
| Polite close | Hard turn 3 | +0.30 |
| Unnecessary escalation | Medium | −0.20 |
| Exceeding max steps | All tasks | Episode terminates |

**Key property:** Every turn gives a partial signal — agents never wait until the end to learn if they did well.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an agent action |
| `GET` | `/state` | Current session internal state |
| `GET` | `/tasks` | Task list with typed action schemas |
| `POST` | `/grader` | Score + turn breakdown for completed episode |
| `POST` | `/baseline` | Run built-in oracle agent, return average scores |

---

## 🚀 Setup & Usage

### 1. Clone

```bash
git clone https://huggingface.co/spaces/sanathkumarps/customer_support_env
cd customer_support_env
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Visit `http://localhost:7860/docs` for interactive Swagger docs.

### 4. Run with Docker

```bash
docker build -t customer_support_env .
docker run -p 7860:7860 customer_support_env
```

### 5. Run the baseline evaluation

```bash
export GROQ_API_KEY="gsk-..."          # required
export ENV_BASE_URL="http://localhost:7860"   # optional
export GROQ_MODEL="llama-3.1-8b-instant"       # optional, default: llama-3.1-8b-instant

python run_baseline.py
```

Results print to console and save to `baseline_scores.json`.

### 6. Quick API test

```bash
# Health check
curl http://localhost:7860/

# Start easy episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy","seed":42}'

# Submit action (replace SESSION_ID)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID","message":"refund","intent":"classify"}'

# Get grader score
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID"}'
```

---

## 📊 Baseline Scores

| Task | Avg Score | Model | Episodes |
|---|---|---|---|
| Easy | 0.90 | llama-3.1-8b-instant | 5 |
| Medium | 0.55 | llama-3.1-8b-instant | 5 |
| Hard | 0.40 | llama-3.1-8b-instant | 5 |

> Run `python run_baseline.py` after starting the server to generate fresh scores.

---

## 📁 Project Structure

```
customer_support_env/
├── models.py              # Pydantic models: SupportAction, SupportObservation, SupportState
├── client.py              # OpenEnv typed client library
├── openenv.yaml           # OpenEnv metadata manifest
├── requirements.txt       # Python dependencies (groq, fastapi, uvicorn…)
├── run_baseline.py        # llama-3.1-8b-instant baseline evaluation script
├── README.md              # This file
├── Dockerfile             # Container config for Hugging Face Spaces
├── .dockerignore          # Excludes venv, __pycache__, .env from image
├── baseline_scores.json   # Auto-generated baseline results
└── server/
    ├── __init__.py        # Package marker
    ├── app.py             # FastAPI server — all 7 endpoints
    └── environment.py     # Core env: 15 scenarios, 3 graders, reward logic
```

---

## 🏗️ Architecture

```
models.py                  ← Pydantic contracts (Action / Observation / State)
    │
    ├── server/environment.py   ← Episode logic, 15 scenarios, 3 graders
    │       │
    │       └── Reward signals: per-turn partial rewards + cumulative score
    │
    └── server/app.py           ← FastAPI: /reset /step /state /tasks /grader /baseline
            │
            └── client.py       ← Typed client for external scripts
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  Built for the <strong>OpenEnv Hackathon</strong> 🚀 &nbsp;|&nbsp; Customer support AI benchmark that fills a real gap.
</p>

