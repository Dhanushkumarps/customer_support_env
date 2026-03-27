"""
server/environment.py - Core environment logic for the OpenEnv Customer Support simulation.

Implements the SupportEnvironment class with three graded task tiers:
  - Easy:   Ticket classification (single-attempt, score 0.0 or 1.0)
  - Medium: Single-turn response quality (keyword-based scoring, 0.0–1.0)
  - Hard:   Multi-turn conversation (clarify → resolve → close, 0.0–1.0)
"""

import random
import uuid
from typing import Optional, Tuple

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass
from models import SupportAction, SupportObservation, SupportState


# ---------------------------------------------------------------------------
# Scenario bank — each issue type has a customer message, grading keywords,
# and a clarification hint for the hard (multi-turn) task.
# ---------------------------------------------------------------------------

SCENARIOS = {
    "refund": {
        "message": "I want a refund for my order #8821 placed last week.",
        "answer_keywords": ["refund", "processed", "initiated", "issued"],
        "clarify_hint": "Ask for order number and reason for return.",
    },
    "technical": {
        "message": "My app keeps crashing every time I try to open it after the latest update.",
        "answer_keywords": ["reinstall", "update", "clear cache", "restart", "troubleshoot"],
        "clarify_hint": "Ask for device type and OS version.",
    },
    "shipping": {
        "message": "I was charged for my order but it hasn't arrived after 3 weeks.",
        "answer_keywords": ["investigate", "track", "reship", "contact carrier", "replacement"],
        "clarify_hint": "Ask for tracking number and delivery address.",
    },
    "billing": {
        "message": "I was charged twice for my subscription this month.",
        "answer_keywords": ["refund", "duplicate charge", "reversed", "credit", "corrected"],
        "clarify_hint": "Ask for account email and transaction ID.",
    },
    "account": {
        "message": "I can't log in to my account and the password reset email never arrives.",
        "answer_keywords": ["reset", "email", "verify", "support team", "alternative"],
        "clarify_hint": "Ask for registered email and whether they checked spam folder.",
    },
}

CATEGORIES = list(SCENARIOS.keys())


class SupportEnvironment(Environment):
    """OpenEnv-compatible customer support simulation environment.

    The environment presents the agent with a customer support ticket and
    grades the agent's responses according to the selected task tier.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        """Initialise the environment with a blank state."""
        super().__init__()
        self._state = SupportState()
        self._resolved = False

    # ------------------------------------------------------------------ #
    #  reset
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "easy",
        **kwargs,
    ) -> SupportObservation:
        """Start a new episode.

        Args:
            seed: Optional RNG seed for reproducibility.
            episode_id: Optional unique ID; auto-generated if omitted.
            task_name: Difficulty tier — "easy", "medium", or "hard".

        Returns:
            The opening SupportObservation with the customer's first message.
        """
        # Seed the RNG for reproducible scenario selection
        if seed is not None:
            random.seed(seed)

        # Pick a random issue type and fetch its scenario
        issue_type = random.choice(CATEGORIES)
        scenario = SCENARIOS[issue_type]

        # Build the initial state
        self._state = SupportState(
            issue_type=issue_type,
            step_count=0,
            resolved=False,
            episode_id=episode_id or str(uuid.uuid4()),
            task_name=task_name,
            conversation_history=[scenario["message"]],
            correct_answer=scenario["answer_keywords"][0],
            max_steps=10,
        )
        self._resolved = False

        # Return the first observation
        return SupportObservation(
            conversation=[scenario["message"]],
            customer_query=scenario["message"],
            task_name=task_name,
            info=None,
            done=False,
            reward=None,
        )

    # ------------------------------------------------------------------ #
    #  step
    # ------------------------------------------------------------------ #

    def step(self, action: SupportAction, **kwargs) -> SupportObservation:
        """Process one agent action and return the next observation.

        Routes grading to the appropriate task-specific grader, enforces the
        max-step limit, and appends messages to the conversation history.

        Args:
            action: The agent's SupportAction (message + optional intent).

        Returns:
            A SupportObservation reflecting the updated environment state.
        """
        state = self._state

        # Track the step and record the agent's message
        state.step_count += 1
        state.conversation_history.append(action.message)

        # Route to the correct grader based on task difficulty
        if state.task_name == "easy":
            reward, done = self._grade_easy(action)
        elif state.task_name == "medium":
            reward, done = self._grade_medium(action)
        elif state.task_name == "hard":
            reward, done = self._grade_hard(action)
        else:
            # Unknown task — fail gracefully
            reward, done = 0.0, True

        # Enforce the maximum step limit
        if state.step_count >= state.max_steps:
            done = True

        # Mark resolution on the state
        if done:
            state.resolved = self._resolved

        # Determine the latest customer query (last customer message in history)
        customer_messages = [
            msg for i, msg in enumerate(state.conversation_history) if i % 2 == 0
        ]
        latest_query = customer_messages[-1] if customer_messages else ""

        return SupportObservation(
            conversation=list(state.conversation_history),
            customer_query=latest_query,
            task_name=state.task_name,
            info=None,
            done=done,
            reward=reward,
        )

    # ------------------------------------------------------------------ #
    #  Graders
    # ------------------------------------------------------------------ #

    def _grade_easy(self, action: SupportAction) -> Tuple[float, bool]:
        """EASY TASK — Ticket Classification.

        The agent must output the correct issue category somewhere in its
        message.  Single attempt: reward is 1.0 (correct) or 0.0 (wrong).

        Args:
            action: The agent's action.

        Returns:
            (reward, done) tuple.
        """
        message_lower = action.message.lower()

        if self._state.issue_type in message_lower:
            # Correct classification
            self._resolved = True
            return 1.0, True
        else:
            # Incorrect — no second chance
            return 0.0, True

    def _grade_medium(self, action: SupportAction) -> Tuple[float, bool]:
        """MEDIUM TASK — Single-turn response quality.

        Scores the agent's reply based on how many relevant keywords it
        contains, with a bonus for detailed responses and a penalty for
        unnecessary escalation.

        Args:
            action: The agent's action.

        Returns:
            (reward, done) tuple.  Always done after one turn.
        """
        message_lower = action.message.lower()
        scenario = SCENARIOS[self._state.issue_type]

        # Keyword-based scoring: +0.25 per keyword found, capped at 1.0
        score = 0.0
        for keyword in scenario["answer_keywords"]:
            if keyword in message_lower:
                score += 0.25

        score = min(score, 1.0)

        # Bonus: +0.1 for a sufficiently detailed response (> 80 chars)
        if len(action.message) > 80:
            score += 0.1

        # Penalty: -0.2 for unnecessary escalation language
        if "escalate" in message_lower or "human agent" in message_lower:
            score -= 0.2

        # Clamp to valid range
        reward = min(1.0, max(0.0, score))
        self._resolved = reward >= 0.5
        return reward, True

    def _grade_hard(self, action: SupportAction) -> Tuple[float, bool]:
        """HARD TASK — Multi-turn conversation.

        Evaluates a 3-turn dialogue:
          Turn 1: Agent should ask a clarifying question.
          Turn 2: Agent should address the issue with relevant keywords.
          Turn 3+: Agent should close the ticket politely.

        Args:
            action: The agent's action.

        Returns:
            (reward, done) tuple.
        """
        state = self._state
        message_lower = action.message.lower()
        scenario = SCENARIOS[state.issue_type]

        # ---- Exceeded max steps ---- #
        if state.step_count > state.max_steps:
            return -0.5, True

        # ---- Turn 1: Clarification ---- #
        if state.step_count == 1:
            # Agent should ask a clarifying question
            if "?" in action.message:
                reward = 0.2
            else:
                reward = -0.1

            # Simulate the customer providing the requested information
            customer_reply = (
                f"Sure, here are the details: {scenario['clarify_hint']}"
            )
            state.conversation_history.append(customer_reply)
            return reward, False

        # ---- Turn 2: Resolution attempt ---- #
        if state.step_count == 2:
            # Score with the same keyword logic as the medium grader
            score = 0.0
            for keyword in scenario["answer_keywords"]:
                if keyword in message_lower:
                    score += 0.25
            reward = min(1.0, max(0.0, score))

            # Simulate customer acknowledgement
            state.conversation_history.append("Ok, I'll try that. Thanks.")
            return reward, False

        # ---- Turn 3+: Closing ---- #
        closing_phrases = ["anything else", "happy to help", "resolved", "thank"]
        if any(phrase in message_lower for phrase in closing_phrases):
            self._resolved = True
            return 0.5, True
        else:
            return 0.0, True

    # ------------------------------------------------------------------ #
    #  State accessor
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> SupportState:
        """Return the current internal environment state."""
        return self._state
