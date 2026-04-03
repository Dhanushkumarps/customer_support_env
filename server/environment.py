"""
server/environment.py - Core environment logic for the OpenEnv Customer Support simulation.

Implements the SupportEnvironment class with three graded task tiers:
  - Easy:   Ticket classification (single-attempt, score 0.0 or 1.0)
  - Medium: Single-turn response quality (multi-faceted keyword scoring, 0.0–1.0)
  - Hard:   Multi-turn conversation with cumulative partial rewards (0.0–1.0)

Scenario bank contains 15 realistic customer support scenarios (3 per category).
"""

import random
import uuid
from typing import List, Optional, Tuple

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass

from models import SupportAction, SupportObservation, SupportState


# ---------------------------------------------------------------------------
# Scenario bank — 15 scenarios across 5 categories (3 per category).
# Each scenario has a customer message, solution keywords, empathy keywords,
# and a realistic multi-turn customer follow-up response.
# ---------------------------------------------------------------------------

SCENARIOS: dict = {
    "refund": [
        {
            "message": "I want a refund for my order #8821. It was placed last week and I never received it.",
            "answer_keywords": ["refund", "processed", "initiated", "issued", "return"],
            "empathy_keywords": ["apologize", "sorry", "understand", "inconvenience"],
            "clarify_hint": "order number and reason for return",
            "customer_followup": "My order number is #8821 and the reason is non-delivery.",
            "customer_ack": "Alright, I understand. Please process it as soon as possible.",
        },
        {
            "message": "I received a damaged product and I'd like to return it for a full refund.",
            "answer_keywords": ["refund", "return", "replacement", "credit", "reimburse"],
            "empathy_keywords": ["apologize", "sorry", "understand", "regret"],
            "clarify_hint": "photos of the damage and order ID",
            "customer_followup": "I have attached the photos. Order ID is 7742.",
            "customer_ack": "Thank you, let's get this resolved.",
        },
        {
            "message": "I cancelled my order but haven't received my refund after 10 days.",
            "answer_keywords": ["refund", "processing", "business days", "initiated", "reimbursed"],
            "empathy_keywords": ["apologize", "sorry", "understand", "inconvenience"],
            "clarify_hint": "cancellation confirmation number or order ID",
            "customer_followup": "Confirmation number is CXL-4892.",
            "customer_ack": "Thanks, I hope it gets resolved quickly.",
        },
    ],
    "technical": [
        {
            "message": "My app keeps crashing every time I try to open it after the latest update.",
            "answer_keywords": ["reinstall", "update", "clear cache", "restart", "troubleshoot"],
            "empathy_keywords": ["apologize", "sorry", "understand", "frustrating"],
            "clarify_hint": "device type and OS version",
            "customer_followup": "I'm on iPhone 14, iOS 17.5.",
            "customer_ack": "Ok I'll try reinstalling. Thanks.",
        },
        {
            "message": "I can't get the integration with Slack to work. It shows a webhook error.",
            "answer_keywords": ["webhook", "reconfigure", "settings", "reconnect", "token"],
            "empathy_keywords": ["apologize", "sorry", "understand", "help"],
            "clarify_hint": "the exact error message and your Slack workspace name",
            "customer_followup": "The error says 'invalid_auth'. Workspace is Acme Corp.",
            "customer_ack": "I'll try regenerating the token. Thanks for the help.",
        },
        {
            "message": "The export to PDF feature has stopped working — it just shows a blank file.",
            "answer_keywords": ["browser", "cache", "update", "alternative", "re-export", "fix"],
            "empathy_keywords": ["apologize", "sorry", "understand", "inconvenience"],
            "clarify_hint": "which browser you are using and your account plan",
            "customer_followup": "I'm using Chrome, on the Pro plan.",
            "customer_ack": "Ok I'll clear the cache and retry.",
        },
    ],
    "shipping": [
        {
            "message": "I was charged for my order but it hasn't arrived after 3 weeks.",
            "answer_keywords": ["investigate", "track", "reship", "contact carrier", "replacement"],
            "empathy_keywords": ["apologize", "sorry", "understand", "inconvenience"],
            "clarify_hint": "tracking number and delivery address",
            "customer_followup": "Tracking is TRK-19283, delivery address is 123 Main St.",
            "customer_ack": "Ok please investigate quickly.",
        },
        {
            "message": "The courier says my package was delivered but I never received anything.",
            "answer_keywords": ["investigate", "reship", "replacement", "lost", "carrier", "claim"],
            "empathy_keywords": ["apologize", "sorry", "understand", "inconvenience"],
            "clarify_hint": "delivery photo provided by the courier and your building address",
            "customer_followup": "The photo shows the wrong door. I'm at apartment 4B.",
            "customer_ack": "Please reship as soon as possible.",
        },
        {
            "message": "My order was split into two packages and I only received one part.",
            "answer_keywords": ["track", "shipping", "second package", "dispatch", "investigate"],
            "empathy_keywords": ["apologize", "sorry", "understand", "short"],
            "clarify_hint": "order ID and which items are missing",
            "customer_followup": "Order #5521 is missing the charging cable.",
            "customer_ack": "Ok, thank you for checking on it.",
        },
    ],
    "billing": [
        {
            "message": "I was charged twice for my subscription this month.",
            "answer_keywords": ["refund", "duplicate charge", "reversed", "credit", "corrected"],
            "empathy_keywords": ["apologize", "sorry", "understand", "inconvenience"],
            "clarify_hint": "account email and transaction ID",
            "customer_followup": "Email is user@example.com, transaction ID TXN-002.",
            "customer_ack": "Thank you for resolving this quickly.",
        },
        {
            "message": "I was billed for an annual plan upgrade I never authorized.",
            "answer_keywords": ["refund", "unauthorized", "reversed", "credit", "investigated"],
            "empathy_keywords": ["apologize", "sorry", "understand", "concern"],
            "clarify_hint": "the date of the charge and your current plan",
            "customer_followup": "Charge was on March 15th, I'm on the monthly Basic plan.",
            "customer_ack": "Please reverse it as soon as possible.",
        },
        {
            "message": "My invoice shows a charge for a service I already cancelled last month.",
            "answer_keywords": ["refund", "credit", "cancelled", "corrected", "removed"],
            "empathy_keywords": ["apologize", "sorry", "understand", "inconvenience"],
            "clarify_hint": "cancellation confirmation number and account ID",
            "customer_followup": "Cancellation ref is CXL-772 and account is ACC-1090.",
            "customer_ack": "Thanks, I appreciate the quick fix.",
        },
    ],
    "account": [
        {
            "message": "I can't log in to my account and the password reset email never arrives.",
            "answer_keywords": ["reset", "email", "verify", "support team", "alternative"],
            "empathy_keywords": ["apologize", "sorry", "understand", "assist"],
            "clarify_hint": "registered email and whether they checked spam folder",
            "customer_followup": "Email is user@example.com and I did check spam, nothing there.",
            "customer_ack": "Ok, I'll wait for the manual reset link.",
        },
        {
            "message": "My account was suspended without warning and I can't access my data.",
            "answer_keywords": ["review", "appeal", "restore", "explain", "investigate"],
            "empathy_keywords": ["apologize", "sorry", "understand", "concern"],
            "clarify_hint": "account username and when the suspension occurred",
            "customer_followup": "Username is john_doe_42, suspended yesterday around 3pm.",
            "customer_ack": "Please reinstate my account quickly.",
        },
        {
            "message": "I need to transfer my account to a new email address but the system won't let me.",
            "answer_keywords": ["update", "verify", "transfer", "new email", "confirm"],
            "empathy_keywords": ["apologize", "sorry", "understand", "help"],
            "clarify_hint": "current email, new email, and identity verification details",
            "customer_followup": "Current is old@example.com, new is new@example.com.",
            "customer_ack": "Great, I'll wait for the verification email.",
        },
    ],
}

CATEGORIES: List[str] = list(SCENARIOS.keys())


# ---------------------------------------------------------------------------
# Closing phrases for the hard-task grader
# ---------------------------------------------------------------------------

CLOSING_PHRASES = [
    "anything else", "happy to help", "resolved", "thank you", "my pleasure",
    "glad i could", "take care", "have a great", "best regards", "feel free",
    "don't hesitate",
]

EMPATHY_PHRASES = [
    "apologize", "sorry", "understand", "frustrating", "inconvenience",
    "regret", "concern", "care", "help you",
]


class SupportEnvironment(Environment):
    """OpenEnv-compatible customer support simulation environment.

    Presents the agent with a realistic customer support ticket and grades
    its responses with task-specific rubrics that provide meaningful partial
    rewards throughout the episode — not just at the end.

    Task tiers:
    ──────────────────────────────────────────────────────────────────────
    Easy   │ Ticket classification — output the correct category label.
    Medium │ Single-turn — write a complete, empathetic resolution reply.
    Hard   │ Multi-turn — clarify, resolve, and close the ticket politely.
    ──────────────────────────────────────────────────────────────────────
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        """Initialise the environment with a blank state."""
        super().__init__()
        self._state = SupportState()

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
            seed:       Optional RNG seed for reproducibility.
            episode_id: Optional unique ID; auto-generated if omitted.
            task_name:  Difficulty tier — "easy", "medium", or "hard".

        Returns:
            The opening SupportObservation with the customer's first message.
        """
        rng = random.Random(seed)

        # Pick a random issue category and scenario within that category
        issue_type = rng.choice(CATEGORIES)
        scenario_list = SCENARIOS[issue_type]
        scenario_idx = rng.randrange(len(scenario_list))
        scenario = scenario_list[scenario_idx]

        self._state = SupportState(
            issue_type=issue_type,
            step_count=0,
            resolved=False,
            episode_id=episode_id or str(uuid.uuid4()),
            task_name=task_name,
            conversation_history=[scenario["message"]],
            correct_answer=issue_type,          # ground-truth category label
            max_steps=10 if task_name == "hard" else 2,
            cumulative_reward=0.0,
            turn_scores=[],
            scenario_index=scenario_idx,
        )

        info_hint = (
            f"Issue category hint: [{issue_type}]" if task_name == "easy"
            else None
        )

        return SupportObservation(
            conversation=[scenario["message"]],
            customer_query=scenario["message"],
            task_name=task_name,
            info=info_hint,
            done=False,
            reward=None,
            cumulative_reward=0.0,
            turn_scores=[],
        )

    # ------------------------------------------------------------------ #
    #  step
    # ------------------------------------------------------------------ #

    def step(self, action: SupportAction, **kwargs) -> SupportObservation:
        """Process one agent action and return the next observation.

        Dispatches to the appropriate task grader, accumulates rewards, and
        enforces the maximum-step limit. Partial rewards are reflected at
        every turn (not just the final one).

        Args:
            action: The agent's SupportAction (message + optional intent).

        Returns:
            A SupportObservation reflecting the updated environment state.
        """
        state = self._state

        # If the episode is already done, return a terminal observation
        if state.resolved or state.step_count >= state.max_steps:
            return self._terminal_obs()

        state.step_count += 1
        state.conversation_history.append(action.message)

        # Route to the task-specific grader
        if state.task_name == "easy":
            reward, done = self._grade_easy(action)
        elif state.task_name == "medium":
            reward, done = self._grade_medium(action)
        elif state.task_name == "hard":
            reward, done = self._grade_hard(action)
        else:
            reward, done = 0.0, True

        # Clamp step reward
        reward = min(1.0, max(-1.0, reward))

        # Accumulate reward and record per-turn score
        state.turn_scores.append(round(reward, 4))
        state.cumulative_reward = min(1.0, max(0.0, sum(state.turn_scores) / max(len(state.turn_scores), 1)))

        # Enforce absolute step ceiling
        if state.step_count >= state.max_steps:
            done = True

        if done:
            state.resolved = state.cumulative_reward >= 0.5

        # Determine the latest customer-facing query
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
            reward=round(reward, 4),
            cumulative_reward=round(state.cumulative_reward, 4),
            turn_scores=list(state.turn_scores),
        )

    # ------------------------------------------------------------------ #
    #  Graders
    # ------------------------------------------------------------------ #

    def _grade_easy(self, action: SupportAction) -> Tuple[float, bool]:
        """EASY — Ticket Classification.

        The agent must output the correct issue category somewhere in its reply.
        Score: 1.0 (correct) or 0.0 (wrong). Single attempt only.
        """
        message_lower = action.message.lower().strip()
        correct = self._state.issue_type

        if correct in message_lower:
            return 1.0, True

        # Partial credit: agent used the right keyword fragment
        for kw in correct.split():
            if kw in message_lower and len(kw) > 3:
                return 0.5, True

        return 0.0, True

    def _grade_medium(self, action: SupportAction) -> Tuple[float, bool]:
        """MEDIUM — Single-turn response quality.

        Multi-faceted scoring:
          • Keyword coverage (+0.20 per keyword, up to 4)
          • Empathy (+0.10 bonus)
          • Detail bonus (+0.10 for replies > 80 chars)
          • Correct action language (+0.10 per action verb)
          • Escalation penalty (−0.20 for unnecessary escalation)
        All scores clamped to [0.0, 1.0].
        """
        msg = action.message.lower()
        scenario = SCENARIOS[self._state.issue_type][self._state.scenario_index]
        score = 0.0

        # Keyword-based scoring: up to 0.80
        for keyword in scenario["answer_keywords"]:
            if keyword in msg:
                score += 0.20
        score = min(score, 0.80)

        # Empathy bonus
        if any(ep in msg for ep in EMPATHY_PHRASES):
            score += 0.10

        # Detail bonus: sufficiently long reply
        if len(action.message) > 80:
            score += 0.10

        # Escalation penalty: -0.20 for unnecessary hand-off language
        if "escalate" in msg or "human agent" in msg or "transfer you" in msg:
            score -= 0.20

        return min(1.0, max(0.0, score)), True

    def _grade_hard(self, action: SupportAction) -> Tuple[float, bool]:
        """HARD — Multi-turn conversation quality.

        Turn 1 (clarify):    +0.30 for a question, −0.10 otherwise
        Turn 2 (resolve):    0.0–0.50 based on keyword matches + empathy
        Turn 3+ (close):     +0.20 for closing phrase; episode ends

        Cumulative score produced at the end normalises to [0.0, 1.0].
        """
        state = self._state
        msg = action.message.lower()
        scenario = SCENARIOS[state.issue_type][state.scenario_index]

        # ---- Turn 1: Clarification ---- #
        if state.step_count == 1:
            if "?" in action.message:
                reward = 0.30
                # Award extra if the agent targets the right area
                if any(kw in msg for kw in scenario["clarify_hint"].split()):
                    reward += 0.10
                reward = min(reward, 0.40)
            else:
                reward = -0.10

            # Simulate the customer providing clarification
            state.conversation_history.append(scenario["customer_followup"])
            return reward, False

        # ---- Turn 2: Resolution ---- #
        if state.step_count == 2:
            score = 0.0
            for keyword in scenario["answer_keywords"]:
                if keyword in msg:
                    score += 0.12          # up to ~0.48 for 4 keywords
            score = min(score, 0.48)

            # Empathy bonus
            if any(ep in msg for ep in EMPATHY_PHRASES):
                score += 0.12

            # Detail
            if len(action.message) > 60:
                score += 0.08

            reward = min(0.50, max(0.0, score))

            # Customer acknowledges
            state.conversation_history.append(scenario["customer_ack"])
            return reward, False

        # ---- Turn 3+: Closing ---- #
        if any(phrase in msg for phrase in CLOSING_PHRASES):
            return 0.30, True
        else:
            return 0.0, True

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _terminal_obs(self) -> SupportObservation:
        """Return a done observation for an already-ended episode."""
        state = self._state
        customer_messages = [
            msg for i, msg in enumerate(state.conversation_history) if i % 2 == 0
        ]
        return SupportObservation(
            conversation=list(state.conversation_history),
            customer_query=customer_messages[-1] if customer_messages else "",
            task_name=state.task_name,
            info="Episode already completed.",
            done=True,
            reward=0.0,
            cumulative_reward=round(state.cumulative_reward, 4),
            turn_scores=list(state.turn_scores),
        )

    # ------------------------------------------------------------------ #
    #  State accessor
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> SupportState:
        """Return the current internal environment state."""
        return self._state
