---
title: Customer Support OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🎧 Customer Support OpenEnv

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-brightgreen?logo=openai&logoColor=white)
![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Spaces-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)

---

## Overview

**Customer Support OpenEnv** is a realistic customer support simulation environment built on the [OpenEnv](https://github.com/openenv) framework. It challenges AI agents to classify support tickets, craft helpful single-turn replies, and manage multi-turn conversations across five real-world issue categories. The environment provides a standardised REST API, deterministic grading rubrics, and a built-in baseline agent — making it ideal for benchmarking LLM-powered support agents in hackathons, research, and production evaluation pipelines.

---

## Environment Description

The environment simulates a customer support system where an AI agent interacts with customers seeking help across **five issue categories**:

| Category | Example Scenario |
|---|---|
| **Refund** | Customer requests a refund for a recent order |
| **Technical** | App crashes after the latest update |
| **Shipping** | Order charged but never delivered |
| **Billing** | Duplicate subscription charge |
| **Account** | Unable to log in, password reset email not arriving |

Each episode presents the agent with a customer message drawn from a scenario bank. The agent must respond appropriately — and its performance is graded automatically by task-specific rubrics that evaluate classification accuracy, response quality, and conversational skill.

---

## Action Space

The agent submits a `SupportAction` each turn:

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | `str` | ✅ | The agent's text reply to the customer |
| `intent` | `str` | ❌ | Declared intent: `"classify"`, `"respond"`, `"clarify"`, `"escalate"`, or `"close"` |

---

## Observation Space

The environment returns a `SupportObservation` after each step:

| Field | Type | Description |
|---|---|---|
| `conversation` | `List[str]` | Full message history (alternating customer/agent) |
| `customer_query` | `str` | The latest customer message the agent must address |
| `task_name` | `str` | Current task tier: `"easy"`, `"medium"`, or `"hard"` |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float \| None` | Grader-assigned score (0.0–1.0); `None` until episode ends |
| `info` | `str \| None` | Optional extra hints or context |

---

## Tasks

### 🟢 Easy: Ticket Classification

**Objective:** Given a single customer message, output the correct issue category.

**Scoring:**
- ✅ `1.0` — Correct category appears in the agent's reply
- ❌ `0.0` — Incorrect or missing category
- Single attempt only (1 step)

**Example interaction:**
```
Customer: "I want a refund for my order #8821 placed last week."
Agent:    "refund"
Score:    1.0 ✅
```

---

### 🟡 Medium: Single-Turn Response

**Objective:** Write a single, helpful reply that resolves the customer's issue.

**Scoring rubric:**
- +0.25 per relevant keyword found (e.g., "refund", "processed", "initiated")
- +0.10 bonus for detailed responses (> 80 characters)
- −0.20 penalty for unnecessary escalation language ("escalate", "human agent")
- Score clamped to `[0.0, 1.0]`

**Example interaction:**
```
Customer: "I was charged twice for my subscription this month."
Agent:    "I sincerely apologize for the inconvenience. I can see the duplicate charge
           on your account and have initiated a refund for the extra payment. The credit
           should appear within 3-5 business days."
Score:    0.85 ✅
```

---

### 🔴 Hard: Multi-Turn Conversation

**Objective:** Handle a full 3-turn support dialogue — clarify, resolve, close.

**Scoring across turns:**

| Turn | Expected Behaviour | Reward |
|---|---|---|
| 1 | Ask a clarifying question (must contain `?`) | +0.2 if question asked, −0.1 otherwise |
| 2 | Provide a concrete solution (keyword-based scoring) | 0.0–1.0 based on keyword matches |
| 3+ | Close the ticket politely | +0.5 if closing phrase detected |

**Example interaction:**
```
Turn 1 — Agent:    "Could you please share your device type and OS version?"        → +0.2
         Customer: "Sure, here are the details: iPhone 14, iOS 17.2"

Turn 2 — Agent:    "Please try clearing the app cache and reinstalling the app."    → +0.50
         Customer: "Ok, I'll try that. Thanks."

Turn 3 — Agent:    "Happy to help! Is there anything else I can assist you with?"   → +0.50
```

---

## Reward Function

- **Keyword matching** — Agents earn partial credit for including relevant solution keywords
- **Detail bonus** — Longer, more thorough responses receive a +0.10 bonus
- **Escalation penalty** — Unnecessary escalation to human agents incurs a −0.20 penalty
- **Clarification reward** — Asking clarifying questions in multi-turn tasks earns +0.20
- **Polite closing** — Detecting polite closing phrases in hard tasks earns +0.50
- **Max-step enforcement** — Episodes exceeding `max_steps` are forcibly terminated
- **All scores clamped** to the `[0.0, 1.0]` range

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns `{"status": "ok", "env": "customer_support_env"}` |
| `POST` | `/reset` | Start a new episode with a given task tier and optional seed |
| `POST` | `/step` | Submit an agent action and receive the next observation |
| `GET` | `/state` | Retrieve the current internal state for a session |
| `GET` | `/tasks` | List all 3 task tiers with descriptions and action schemas |
| `POST` | `/grader` | Get the final score for a completed episode |
| `POST` | `/baseline` | Run the built-in rule-based agent on all tasks and return scores |

---

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer_support_env.git
cd customer_support_env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally with Uvicorn

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

The API will be available at `http://localhost:7860`. Visit `http://localhost:7860/docs` for interactive Swagger documentation.

### 4. Run with Docker

```bash
docker build -t customer_support_env .
docker run -p 7860:7860 customer_support_env
```

### 5. Run the baseline evaluation

```bash
export GROQ_API_KEY="your-api-key"
python run_baseline.py
```

Results are printed to the console and saved to `baseline_scores.json`.

---

## Baseline Scores

| Task | Baseline Score | Model Used | Episodes |
|---|---|---|---|
| Easy | 0.80 | llama-3.1-8b-instant | 5 |
| Medium | 0.22 | llama-3.1-8b-instant | 5 |
| Hard | 0.30 | llama-3.1-8b-instant | 5 |

> Run `python run_baseline.py` after starting the server to populate these scores.

---

## Project Structure

```
customer_support_env/
├── models.py              # Pydantic models: SupportAction, SupportObservation, SupportState
├── client.py              # OpenEnv client library (JSON ↔ Pydantic)
├── openenv.yaml           # OpenEnv metadata manifest
├── requirements.txt       # Python dependencies
├── run_baseline.py        # llama-3.1-8b-instant baseline evaluation script
├── README.md              # This file
├── Dockerfile             # Container config for Hugging Face Spaces
├── baseline_scores.json   # Auto-generated baseline results
└── server/
    ├── __init__.py        # Package marker
    ├── app.py             # FastAPI server with all endpoints
    └── environment.py     # Core environment: scenarios, graders, step logic
```

---

## Architecture

```
┌─────────────┐     POST /reset      ┌──────────────────┐
│             │────────────────────▶│                  │
│   Agent     │     POST /step       │   FastAPI Server │
│  (client)   │◀────────────────────│   (server/app)   │
│             │    observation+reward │                  │
└─────────────┘                      └────────┬─────────┘
                                              │
                                     ┌────────▼─────────┐
                                     │   Environment    │
                                     │  (environment.py)│
                                     │                  │
                                     │  5 scenarios     │
                                     │  3 graders       │
                                     └──────────────────┘
```

---

## License

MIT

---

<p align="center">
  Built for the <strong>OpenEnv Hackathon</strong> 🚀
</p>
