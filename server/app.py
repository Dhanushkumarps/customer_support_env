"""
server/app.py - FastAPI web server exposing the Customer Support OpenEnv as a REST API.

Endpoints:
    GET  /          Health check
    POST /reset     Start a new episode
    POST /step      Submit an agent action
    GET  /state     Retrieve current session state
    GET  /tasks     List available task tiers with schemas
    POST /grader    Get the final score for a completed episode
    POST /baseline  Run a built-in rule-based agent across all tasks
"""

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.environment import SupportEnvironment
from models import SupportAction


# ------------------------------------------------------------------ #
#  App setup
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Customer Support OpenEnv",
    version="0.1.0",
    description="An OpenEnv-compatible customer support simulation environment.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (session_id -> SupportEnvironment)
sessions: Dict[str, SupportEnvironment] = {}


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def get_or_create_session(session_id: str) -> SupportEnvironment:
    """Return an existing session or create a new one.

    Args:
        session_id: Unique identifier for the session.

    Returns:
        The SupportEnvironment instance for the given session.
    """
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
        Status and environment name.
    """
    return {"status": "ok", "env": "customer_support_env"}


@app.post("/reset")
async def reset(request: ResetRequest) -> Dict[str, Any]:
    """Start a new episode.

    Creates or reuses a session, resets the environment with the requested
    task tier, and returns the opening observation.

    Args:
        request: ResetRequest with optional session_id, task_name, and seed.

    Returns:
        The opening observation dict plus the session_id.
    """
    session_id = request.session_id or str(uuid.uuid4())
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
        The updated observation dict including done and reward.

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
async def get_state(session_id: str = Query(..., description="Session ID to look up")) -> Dict[str, Any]:
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
        A dict containing a list of task descriptors.
    """
    return {
        "tasks": [
            {
                "name": "easy",
                "description": (
                    "Ticket Classification: given a customer message, "
                    "output the correct issue category."
                ),
                "difficulty": "easy",
                "action_schema": {
                    "message": "string — write the category name in your reply",
                    "intent": "classify",
                },
            },
            {
                "name": "medium",
                "description": (
                    "Single-Turn Response: write a helpful reply resolving "
                    "the customer issue in one message."
                ),
                "difficulty": "medium",
                "action_schema": {
                    "message": "string — your full support reply",
                    "intent": "respond",
                },
            },
            {
                "name": "hard",
                "description": (
                    "Multi-Turn Conversation: handle a full support dialogue "
                    "across 3 turns: clarify, resolve, close."
                ),
                "difficulty": "hard",
                "action_schema": {
                    "message": "string — your reply for this turn",
                    "intent": "clarify | respond | close",
                },
            },
        ]
    }


@app.post("/grader")
async def grader(request: GraderRequest) -> Dict[str, Any]:
    """Return the final score for a completed episode.

    Args:
        request: GraderRequest with session_id and optional episode_summary.

    Returns:
        Score, task name, and step count if the episode is done;
        otherwise a null score with an explanatory message.
    """
    if request.session_id not in sessions:
        return {"score": None, "message": "Episode not complete"}

    env = sessions[request.session_id]
    state = env.state

    # Check whether the episode has actually finished
    if not state.resolved and state.step_count == 0:
        return {"score": None, "message": "Episode not complete"}

    return {
        "score": float(state.resolved),
        "task": state.task_name,
        "steps": state.step_count,
    }


@app.post("/baseline")
async def run_baseline() -> Dict[str, float]:
    """Run a built-in rule-based agent on all 3 task tiers and return average scores.

    Executes 5 episodes per task using deterministic heuristic agents:
      - Easy:   Echoes the issue type keyword.
      - Medium: Sends a generic refund acknowledgement reply.
      - Hard:   3-turn script (clarify → resolve → close).

    Returns:
        Average reward per task tier, e.g. {"easy": 0.8, "medium": 0.55, "hard": 0.4}.
    """
    num_episodes = 5
    results: Dict[str, float] = {}

    # ---- Easy baseline ---- #
    easy_rewards = []
    for i in range(num_episodes):
        env = SupportEnvironment()
        obs = env.reset(seed=i, task_name="easy")
        # Strategy: reply with the correct issue type from the state
        action = SupportAction(message=env.state.issue_type, intent="classify")
        obs = env.step(action)
        easy_rewards.append(obs.reward or 0.0)
    results["easy"] = sum(easy_rewards) / len(easy_rewards)

    # ---- Medium baseline ---- #
    medium_rewards = []
    for i in range(num_episodes):
        env = SupportEnvironment()
        obs = env.reset(seed=i, task_name="medium")
        action = SupportAction(
            message=(
                "I have processed your refund request and it will reflect "
                "within 3-5 business days."
            ),
            intent="respond",
        )
        obs = env.step(action)
        medium_rewards.append(obs.reward or 0.0)
    results["medium"] = sum(medium_rewards) / len(medium_rewards)

    # ---- Hard baseline ---- #
    hard_rewards = []
    for i in range(num_episodes):
        env = SupportEnvironment()
        obs = env.reset(seed=i, task_name="hard")

        # Turn 1: Ask for clarification
        action1 = SupportAction(
            message="Could you please share more details?",
            intent="clarify",
        )
        obs = env.step(action1)

        # Turn 2: Attempt resolution
        action2 = SupportAction(
            message="I will escalate this to resolve it.",
            intent="respond",
        )
        obs = env.step(action2)

        # Turn 3: Close politely
        action3 = SupportAction(
            message="Happy to help! Is there anything else?",
            intent="close",
        )
        obs = env.step(action3)

        hard_rewards.append(obs.reward or 0.0)
    results["hard"] = sum(hard_rewards) / len(hard_rewards)

    return results
