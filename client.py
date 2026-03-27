"""
client.py — OpenEnv client for the Customer Support Environment.

Translates between Python objects and the JSON API so that external scripts
(e.g. run_baseline.py) can interact with the running environment server
without dealing with raw HTTP.
"""

from typing import Any, Dict, Optional

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
except ImportError:
    from typing import Generic, TypeVar
    from pydantic import BaseModel
    ActionType = TypeVar("ActionType")
    ObsType = TypeVar("ObsType")
    StateType = TypeVar("StateType")
    
    class EnvClient(Generic[ActionType, ObsType, StateType]):
        pass
        
    class StepResult(BaseModel):
        observation: Any = None
        reward: float = 0.0
        done: bool = False
        info: dict = {}
from models import SupportAction, SupportObservation, SupportState


class SupportEnvClient(EnvClient[SupportAction, SupportObservation, SupportState]):
    """Client for the Customer Support OpenEnv.

    Wraps the JSON API with typed Python objects for actions, observations,
    and state.

    Usage::

        env = SupportEnvClient(base_url="http://localhost:8000")
        obs = env.reset(task_name="easy")
        result = env.step(SupportAction(message="refund", intent="classify"))
        print(result.observation.reward)
    """

    # ------------------------------------------------------------------ #
    #  Serialisation helpers
    # ------------------------------------------------------------------ #

    def _step_payload(self, action: SupportAction) -> Dict[str, Any]:
        """Convert a SupportAction into the JSON dict expected by POST /step.

        Args:
            action: The agent's action object.

        Returns:
            Dict with session_id, message, and intent ready for the API.
        """
        return {
            "session_id": self.session_id,
            "message": action.message,
            "intent": action.intent,
        }

    # ------------------------------------------------------------------ #
    #  Deserialisation helpers
    # ------------------------------------------------------------------ #

    def _parse_observation(self, payload: Dict[str, Any]) -> SupportObservation:
        """Parse a raw server response dict into a SupportObservation.

        Handles missing keys gracefully by falling back to sensible defaults.

        Args:
            payload: The JSON dict returned by the server (usually nested
                     under an ``"observation"`` key).

        Returns:
            A fully-populated SupportObservation instance.
        """
        # The server wraps the observation under an "observation" key
        obs_data = payload.get("observation", payload)

        return SupportObservation(
            conversation=obs_data.get("conversation", []),
            customer_query=obs_data.get("customer_query", ""),
            task_name=obs_data.get("task_name", "easy"),
            info=obs_data.get("info", None),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", None),
        )

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        """Parse a raw server response into a StepResult.

        Extracts the observation, reward, done flag, and any extra info
        from the API response.

        Args:
            payload: The JSON dict returned by POST /step.

        Returns:
            A StepResult containing the observation, reward, done, and info.
        """
        observation = self._parse_observation(payload)

        return StepResult(
            observation=observation,
            reward=observation.reward if observation.reward is not None else 0.0,
            done=observation.done,
            info={
                "session_id": payload.get("session_id", self.session_id),
            },
        )

    # ------------------------------------------------------------------ #
    #  High-level API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        task_name: str = "easy",
        seed: Optional[int] = None,
    ) -> SupportObservation:
        """Start a new episode on the server.

        Sends POST /reset with the requested task tier and optional seed,
        stores the returned session_id, and returns the opening observation.

        Args:
            task_name: Difficulty tier — ``"easy"``, ``"medium"``, or ``"hard"``.
            seed: Optional RNG seed for reproducible scenario selection.

        Returns:
            The opening SupportObservation with the customer's first message.
        """
        request_body: Dict[str, Any] = {
            "task_name": task_name,
        }

        # Include session_id if we already have one (re-use session)
        if hasattr(self, "session_id") and self.session_id:
            request_body["session_id"] = self.session_id

        # Include seed only if provided
        if seed is not None:
            request_body["seed"] = seed

        # POST /reset and parse the response
        response = self._post("/reset", json=request_body)
        payload = response.json()

        # Store the session_id for subsequent calls
        self.session_id = payload.get("session_id", "")

        return self._parse_observation(payload)
