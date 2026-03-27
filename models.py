"""
models.py - Pydantic data models for the OpenEnv Customer Support environment.

Defines the action, observation, and state schemas used by the environment server,
client, and baseline agent.
"""

from typing import List, Optional

from pydantic import BaseModel
try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    # Fallback for when the hackathon's OpenEnv framework is not installed
    Action = BaseModel
    Observation = BaseModel
    State = BaseModel


class SupportAction(Action):
    """An action taken by the support agent in response to a customer query.

    The agent sends a text message and optionally declares its intent
    (e.g., classify the ticket, respond with a solution, ask for clarification,
    escalate to a human, or close the ticket).
    """

    message: str
    """The agent's text reply to the customer."""

    intent: Optional[str] = None
    """The agent's declared intent for this action.

    Must be one of: "classify", "respond", "clarify", "escalate", "close".
    If None, the environment will infer intent from the message content.
    """


class SupportObservation(Observation):
    """An observation returned by the environment after each step.

    Contains the full conversation history, the latest customer message
    the agent must address, the current task difficulty, and scoring info.
    """

    conversation: List[str]
    """Full list of all messages exchanged so far (alternating customer/agent)."""

    customer_query: str
    """The latest customer message the agent must respond to."""

    task_name: str
    """Difficulty tier for the current episode: "easy", "medium", or "hard"."""

    info: Optional[str] = None
    """Extra hints or context provided to the agent (e.g., knowledge-base snippets)."""

    done: bool = False
    """Whether the episode has ended (resolved, escalated, or max steps reached)."""

    reward: Optional[float] = None
    """Grader-assigned reward for the agent's performance (0.0–1.0). None until episode ends."""


class SupportState(State):
    """Internal state of the customer support environment for a single episode.

    Tracks the issue metadata, conversation progress, and grading context.
    The environment uses this to drive step logic and evaluate the agent.
    """

    issue_type: str = ""
    """Category of the customer issue, e.g. "refund", "technical", "shipping", "billing", "account"."""

    step_count: int = 0
    """Number of agent actions taken so far in this episode."""

    resolved: bool = False
    """Whether the customer's issue has been successfully resolved."""

    episode_id: str = ""
    """Unique identifier for the current episode."""

    task_name: str = "easy"
    """Difficulty tier for the current episode: "easy", "medium", or "hard"."""

    conversation_history: List[str] = []
    """Running log of all messages exchanged between the customer and the agent."""

    correct_answer: str = ""
    """Ground-truth answer used by graders to evaluate the agent's reply."""

    max_steps: int = 10
    """Maximum number of agent steps allowed before the episode is forcibly ended."""
