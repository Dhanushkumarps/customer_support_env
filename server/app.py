"""
server/app.py - FastAPI web server exposing the Customer Support OpenEnv as a REST API.

Endpoints:
    GET  /          Health check
    POST /reset     Start a new episode
    POST /step      Submit an agent action
    GET  /state     Retrieve current session state
    GET  /tasks     List available task tiers with action schemas
    POST /grader    Get the final score for a completed episode
    POST /baseline  Run a built-in rule-based agent across all tasks
"""

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.environment import SupportEnvironment
from models import SupportAction


# ------------------------------------------------------------------ #
#  App setup
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Customer Support OpenEnv",
    version="0.2.0",
    description=(
        "An OpenEnv-compatible customer support simulation environment. "
        "Challenges AI agents to classify tickets, craft empathetic responses, "
        "and manage multi-turn support conversations across 5 real-world issue categories."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> SupportEnvironment
sessions: Dict[str, SupportEnvironment] = {}


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def get_or_create_session(session_id: str) -> SupportEnvironment:
    """Return an existing session or create a new one."""
    if session_id not in sessions:
        sessions[session_id] = SupportEnvironment()
    return sessions[session_id]


# ------------------------------------------------------------------ #
#  Request / Response schemas
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    """Body for the POST /reset endpoint."""
    session_id: Optional[str] = None
    task_name: str = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    """Body for the POST /step endpoint."""
    session_id: str
    message: str
    intent: Optional[str] = None


class GraderRequest(BaseModel):
    """Body for the POST /grader endpoint."""
    session_id: str
    episode_summary: Optional[str] = None


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@app.get("/")
async def health_check() -> Dict[str, str]:
    """Health-check endpoint.

    Returns:
        Status, environment name, and version.
    """
    return {
        "status": "ok",
        "env": "customer_support_env",
        "version": "0.2.0",
        "description": "OpenEnv customer support simulation — classify, respond, and resolve.",
    }


@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    """Start a new episode.

    Creates or reuses a session, resets the environment with the requested
    task tier, and returns the opening observation.
    """
    try:
        body = await request.json()
    except Exception:
        body = None

    if not body or not isinstance(body, dict):
        body = {}
        
    session_id = body.get("session_id") or str(uuid.uuid4())
    task_name = body.get("task_name", "easy")
    seed = body.get("seed")
    
    env = get_or_create_session(session_id)

    observation = env.reset(
        seed=request.seed,
        episode_id=session_id,
        task_name=request.task_name,
    )

    return {
        "session_id": session_id,
        "observation": observation.model_dump(),
    }


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    """Submit an agent action and receive the next observation.

    Args:
        request: StepRequest with session_id, message, and optional intent.

    Returns:
        The updated observation dict including reward and cumulative_reward.

    Raises:
        HTTPException 404: If the session_id is not found.
    """
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Call /reset first.",
        )

    env = sessions[request.session_id]
    action = SupportAction(message=request.message, intent=request.intent)
    observation = env.step(action)

    return {
        "session_id": request.session_id,
        "observation": observation.model_dump(),
    }


@app.get("/state")
async def get_state(
    session_id: str = Query(..., description="Session ID to look up")
) -> Dict[str, Any]:
    """Retrieve the current internal state for a session.

    Args:
        session_id: The session to query (passed as a query parameter).

    Returns:
        The full state dict for the session.

    Raises:
        HTTPException 404: If the session_id is not found.
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    env = sessions[session_id]
    return {"session_id": session_id, "state": env.state.model_dump()}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List all available task tiers with descriptions and action schemas.

    Returns:
        A dict containing a list of task descriptors, each with name,
        description, difficulty, max_steps, and action_schema.
    """
    return {
        "tasks": [
            {
                "name": "easy",
                "description": (
                    "Ticket Classification: given a single customer message, "
                    "output the correct issue category (refund, technical, "
                    "shipping, billing, or account)."
                ),
                "difficulty": "easy",
                "max_steps": 1,
                "action_schema": {
                    "message": {
                        "type": "string",
                        "description": "Write exactly the category name in your reply",
                        "example": "refund",
                    },
                    "intent": {
                        "type": "string",
                        "description": "Must be 'classify'",
                        "example": "classify",
                    },
                },
            },
            {
                "name": "medium",
                "description": (
                    "Single-Turn Response: write a helpful, empathetic reply "
                    "that resolves the customer's issue in one message."
                ),
                "difficulty": "medium",
                "max_steps": 1,
                "action_schema": {
                    "message": {
                        "type": "string",
                        "description": "Your full support reply (empathetic, actionable, ≤150 words)",
                        "example": "I sincerely apologize for the inconvenience. I have initiated a refund...",
                    },
                    "intent": {
                        "type": "string",
                        "description": "Must be 'respond'",
                        "example": "respond",
                    },
                },
            },
            {
                "name": "hard",
                "description": (
                    "Multi-Turn Conversation: handle a full 3-turn support "
                    "dialogue — clarify the issue, provide a resolution, "
                    "and politely close the ticket."
                ),
                "difficulty": "hard",
                "max_steps": 10,
                "action_schema": {
                    "message": {
                        "type": "string",
                        "description": "Your reply for this turn",
                        "example": "Could you please share your order number?",
                    },
                    "intent": {
                        "type": "string",
                        "description": "One of: 'clarify', 'respond', 'close'",
                        "example": "clarify",
                    },
                },
            },
        ]
    }


@app.post("/grader")
async def grader(request: GraderRequest) -> Dict[str, Any]:
    """Return the final score for a completed episode.

    For hard (multi-turn) tasks, returns the cumulative_reward calculated
    across all turns plus a per-turn breakdown via turn_scores.

    Args:
        request: GraderRequest with session_id and optional episode_summary.

    Returns:
        score (float 0.0–1.0), task name, step count, cumulative reward,
        and per-turn score breakdown.
    """
    if request.session_id not in sessions:
        return {"score": None, "message": "Session not found — call /reset first."}

    env = sessions[request.session_id]
    state = env.state

    if state.step_count == 0:
        return {"score": None, "message": "Episode not started — call /step first."}

    # Use cumulative_reward as the primary score
    score = round(state.cumulative_reward, 4)

    return {
        "score": score,
        "task": state.task_name,
        "steps": state.step_count,
        "cumulative_reward": score,
        "turn_scores": state.turn_scores,
        "resolved": state.resolved,
        "issue_type": state.issue_type,
    }


@app.post("/baseline")
async def run_baseline() -> Dict[str, Any]:
    """Run a built-in deterministic rule-based agent on all 3 task tiers.

    Executes 5 episodes per task using deterministic heuristic agents:
      - Easy:   Echoes the correct issue type keyword.
      - Medium: Sends a multi-keyword empathetic refund reply.
      - Hard:   3-turn script (clarify → resolve → close).

    Returns:
        Average cumulative_reward per task tier, plus per-task details.
    """
    num_episodes = 5
    results: Dict[str, Any] = {}

    # ---- Easy baseline ---- #
    easy_rewards = []
    for i in range(num_episodes):
        env = SupportEnvironment()
        env.reset(seed=i, task_name="easy")
        # Oracle agent: always use the ground-truth issue type
        action = SupportAction(message=env.state.issue_type, intent="classify")
        obs = env.step(action)
        easy_rewards.append(obs.cumulative_reward)
    results["easy"] = {
        "average_score": round(sum(easy_rewards) / len(easy_rewards), 4),
        "scores": [round(r, 4) for r in easy_rewards],
    }

    # ---- Medium baseline ---- #
    medium_rewards = []
    medium_reply = (
        "I sincerely apologize for the inconvenience you've experienced. "
        "I have investigated your account and initiated a full refund. "
        "The credit should be processed and reflected within 3–5 business days. "
        "If you have any tracking concerns or billing queries, please don't hesitate "
        "to contact us again. Thank you for your patience."
    )
    for i in range(num_episodes):
        env = SupportEnvironment()
        env.reset(seed=i, task_name="medium")
        action = SupportAction(message=medium_reply, intent="respond")
        obs = env.step(action)
        medium_rewards.append(obs.cumulative_reward)
    results["medium"] = {
        "average_score": round(sum(medium_rewards) / len(medium_rewards), 4),
        "scores": [round(r, 4) for r in medium_rewards],
    }

    # ---- Hard baseline ---- #
    hard_rewards = []
    for i in range(num_episodes):
        env = SupportEnvironment()
        env.reset(seed=i, task_name="hard")

        action1 = SupportAction(
            message="Could you please share more details so I can investigate this for you?",
            intent="clarify",
        )
        obs = env.step(action1)

        action2 = SupportAction(
            message=(
                "Thank you for those details. I have investigated the issue, "
                "initiated a refund, and escalated the tracking query to our "
                "logistics team. You should see a resolution within 3–5 business days."
            ),
            intent="respond",
        )
        obs = env.step(action2)

        action3 = SupportAction(
            message="Happy to help! Is there anything else I can assist you with today?",
            intent="close",
        )
        obs = env.step(action3)

        hard_rewards.append(obs.cumulative_reward)
    results["hard"] = {
        "average_score": round(sum(hard_rewards) / len(hard_rewards), 4),
        "scores": [round(r, 4) for r in hard_rewards],
    }

    return {
        "easy": results["easy"]["average_score"],
        "medium": results["medium"]["average_score"],
        "hard": results["hard"]["average_score"],
        "details": results,
    }
