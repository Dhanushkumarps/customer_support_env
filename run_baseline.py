"""
run_baseline.py — Run a Groq-powered agent against all 3 task tiers of the
Customer Support OpenEnv and record scores.

Usage:
    # Start the server first:
    #   uvicorn server.app:app --host 0.0.0.0 --port 7860
    #
    # Then run:
    #   python run_baseline.py

Environment variables:
    GROQ_API_KEY     — Required. Your Groq API key.
    ENV_BASE_URL     — Optional. Defaults to http://localhost:7860.
"""

import json
import os
import sys
from typing import Any, Dict, List

import httpx
from groq import Groq
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY environment variable is not set.")
    sys.exit(1)

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL = "llama-3.1-8b-instant"
EPISODES_PER_TASK = 5

# ------------------------------------------------------------------ #
#  System prompt
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are a professional customer support agent. Your job is to help customers \
resolve their issues efficiently and politely.

For the EASY task: Read the customer message and reply with ONLY the category label.
  Valid categories are: refund, technical, shipping, billing, account

For the MEDIUM task: Write a single, complete, helpful reply that addresses the \
customer's issue.
  Include specific actions you are taking (e.g. "I have initiated a refund...").
  Keep it under 150 words.

For the HARD task (multi-turn):
  - Turn 1: Ask ONE clarifying question to better understand the issue.
  - Turn 2: Provide a concrete solution based on what the customer told you.
  - Turn 3: Close the conversation politely \
(e.g. "Happy to help! Is there anything else I can assist you with?")
"""

# ------------------------------------------------------------------ #
#  Groq client
# ------------------------------------------------------------------ #

ai_client = Groq(api_key=GROQ_API_KEY)


def get_agent_reply(conversation: List[str], task_name: str, turn: int) -> str:
    """Ask Groq for the next agent reply.

    Args:
        conversation: Full conversation history so far.
        task_name: Current task tier (easy, medium, hard).
        turn: Current turn number (1-indexed).

    Returns:
        The agent's text reply.
    """
    # Build the chat messages from conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for i, msg in enumerate(conversation):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg})

    # Add a turn-specific hint for hard tasks
    if task_name == "hard":
        hints = {
            1: "This is turn 1. Ask a clarifying question.",
            2: "This is turn 2. Provide a concrete solution.",
            3: "This is turn 3. Close the conversation politely.",
        }
        hint = hints.get(turn, "Continue the conversation appropriately.")
        messages.append({"role": "system", "content": f"[HINT FOR THIS TURN: {hint}]"})

    try:
        response = ai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [Groq error] {e}")
        return "I apologize for the inconvenience. Let me help you with that."


# ------------------------------------------------------------------ #
#  Environment API helpers
# ------------------------------------------------------------------ #

def env_reset(client: httpx.Client, task_name: str, seed: int) -> Dict[str, Any]:
    """POST /reset — start a new episode."""
    response = client.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name, "seed": seed},
    )
    response.raise_for_status()
    return response.json()


def env_step(client: httpx.Client, session_id: str, message: str, intent: str = None) -> Dict[str, Any]:
    """POST /step — submit an agent action."""
    payload = {"session_id": session_id, "message": message}
    if intent:
        payload["intent"] = intent

    response = client.post(f"{ENV_BASE_URL}/step", json=payload)
    response.raise_for_status()
    return response.json()


# ------------------------------------------------------------------ #
#  Run episodes
# ------------------------------------------------------------------ #

def run_task(client: httpx.Client, task_name: str) -> List[float]:
    """Run EPISODES_PER_TASK episodes for a given task tier."""
    rewards = []

    for ep in range(EPISODES_PER_TASK):
        try:
            reset_data = env_reset(client, task_name, seed=ep)
            session_id = reset_data["session_id"]
            obs = reset_data.get("observation", {})

            done = obs.get("done", False)
            reward = obs.get("reward", None)
            turn = 0

            while not done:
                turn += 1
                conversation = obs.get("conversation", [])

                # Get the agent's reply from Groq
                agent_reply = get_agent_reply(conversation, task_name, turn)

                if task_name == "easy":
                    intent = "classify"
                elif task_name == "medium":
                    intent = "respond"
                else:
                    intent_map = {1: "clarify", 2: "respond", 3: "close"}
                    intent = intent_map.get(turn, "respond")

                step_data = env_step(client, session_id, agent_reply, intent)
                obs = step_data.get("observation", {})
                done = obs.get("done", False)
                reward = obs.get("reward", None)

                if turn >= 15:
                    print(f"  [Warning] Episode {ep + 1} exceeded 15 turns, breaking.")
                    break

            episode_reward = reward if reward is not None else 0.0
            rewards.append(episode_reward)
            print(f"  Episode {ep + 1}/{EPISODES_PER_TASK}: reward = {episode_reward:.2f}")

        except Exception as e:
            print(f"  Episode {ep + 1}/{EPISODES_PER_TASK}: ERROR — {e}")
            rewards.append(0.0)

    return rewards


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    print("=" * 60)
    print("  Customer Support OpenEnv — Baseline Evaluation")
    print(f"  Model: {MODEL}")
    print(f"  Server: {ENV_BASE_URL}")
    print(f"  Episodes per task: {EPISODES_PER_TASK}")
    print("=" * 60)

    results = {}

    with httpx.Client(timeout=60.0) as client:
        for task_name in ["easy", "medium", "hard"]:
            print(f"\n{'─' * 40}")
            print(f"  Task: {task_name.upper()}")
            print(f"{'─' * 40}")

            rewards = run_task(client, task_name)
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            results[task_name] = {
                "average_score": round(avg_reward, 4),
                "scores": [round(r, 4) for r in rewards],
                "episodes": len(rewards),
            }

    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Task':<12} {'Avg Score':<12} {'Episodes':<10} {'Scores'}")
    print(f"  {'─' * 50}")
    for task_name in ["easy", "medium", "hard"]:
        r = results[task_name]
        scores_str = ", ".join(f"{s:.2f}" for s in r["scores"])
        print(f"  {task_name:<12} {r['average_score']:<12.4f} {r['episodes']:<10} [{scores_str}]")
    print(f"{'=' * 60}\n")

    output_path = os.path.join(os.path.dirname(__file__), "baseline_scores.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
