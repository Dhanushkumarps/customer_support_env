import json
import os
import sys
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()

# ------------------------------------------------------------------ #
#  Configuration Required for Hackathon Submission
# ------------------------------------------------------------------ #

# Required variables exactly as strictly specified in the submission prompt
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Environment Endpoint config
ENV_API_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
EPISODES_PER_TASK = 5

# ------------------------------------------------------------------ #
#  OpenAI client configured via the required variables
# ------------------------------------------------------------------ #

# Using HF_TOKEN or another key. We pass it through easily.
# If testing locally, you can export OPENAI_API_KEY.
api_key = os.getenv("OPENAI_API_KEY", HF_TOKEN) if not HF_TOKEN else HF_TOKEN

ai_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=api_key or "DUMMY_KEY",  # Fallback for environments that don't enforce keys
)

# ------------------------------------------------------------------ #
#  System prompt
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are a professional customer support agent. Your job is to help customers \
resolve their issues efficiently and politely.

For the EASY task: Read the customer message and reply with ONLY the category label.
  Valid categories are: refund, technical, shipping, billing, account

For the MEDIUM task: Write a single, complete, empathetic reply that addresses the \
customer's issue in one message.

For the HARD task (multi-turn):
  - Turn 1: Ask ONE clarifying question to better understand the issue.
  - Turn 2: Provide a concrete solution based on what the customer told you.
  - Turn 3: Close the conversation politely.
"""


def get_agent_reply(conversation: List[str], task_name: str, turn: int) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, msg in enumerate(conversation):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg})

    if task_name == "hard":
        hints = {
            1: "This is turn 1. Ask ONE clarifying question only.",
            2: "This is turn 2. Provide a concrete, actionable solution.",
            3: "This is turn 3. Close the conversation politely.",
        }
        hint = hints.get(turn, "Continue the support conversation appropriately.")
        messages.append({"role": "system", "content": f"[HINT FOR THIS TURN: {hint}]"})

    try:
        response = ai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [OpenAI error] {e}")
        return "I apologize for the inconvenience. Let me help you with that."


# ------------------------------------------------------------------ #
#  Environment API helpers
# ------------------------------------------------------------------ #

def env_reset(client: httpx.Client, task_name: str, seed: int) -> Dict[str, Any]:
    response = client.post(
        f"{ENV_API_URL}/reset",
        json={"task_name": task_name, "seed": seed},
    )
    response.raise_for_status()
    return response.json()


def env_step(
    client: httpx.Client, session_id: str, message: str, intent: str = None
) -> Dict[str, Any]:
    payload = {"session_id": session_id, "message": message}
    if intent:
        payload["intent"] = intent
    response = client.post(f"{ENV_API_URL}/step", json=payload)
    response.raise_for_status()
    return response.json()


# ------------------------------------------------------------------ #
#  Run episodes strictly logging START/STEP/END
# ------------------------------------------------------------------ #

def run_task(client: httpx.Client, task_name: str) -> List[float]:
    rewards = []

    for ep in range(EPISODES_PER_TASK):
        print("START")  # REQUIRED LOGGING
        
        try:
            reset_data = env_reset(client, task_name, seed=ep)
            session_id = reset_data["session_id"]
            obs = reset_data.get("observation", {})

            done = obs.get("done", False)
            cumulative = obs.get("cumulative_reward", 0.0)
            turn = 0

            while not done:
                print("STEP")  # REQUIRED LOGGING
                turn += 1
                conversation = obs.get("conversation", [])

                agent_reply = get_agent_reply(conversation, task_name, turn)

                if task_name == "easy":
                    intent = "classify"
                elif task_name == "medium":
                    intent = "respond"
                else:
                    intent_map = {1: "clarify", 2: "respond", 3: "close"}
                    intent = intent_map.get(turn, "close")

                step_data = env_step(client, session_id, agent_reply, intent)
                obs = step_data.get("observation", {})
                done = obs.get("done", False)
                cumulative = obs.get("cumulative_reward", 0.0)

                if turn >= 15:
                    print(f"  [Warning] Episode exceeded bounds, breaking.")
                    break
            
            print("END")  # REQUIRED LOGGING
            
            episode_reward = cumulative if cumulative is not None else 0.0
            rewards.append(episode_reward)

        except Exception as e:
            print(f"  [ERROR] {e}")
            rewards.append(0.0)

    return rewards


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    results = {}
    with httpx.Client(timeout=90.0) as client:
        for task_name in ["easy", "medium", "hard"]:
            rewards = run_task(client, task_name)
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            results[task_name] = {
                "average_score": round(avg_reward, 4),
                "scores": [round(r, 4) for r in rewards],
                "episodes": len(rewards),
                "model": MODEL_NAME,
            }

    output_path = os.path.join(os.path.dirname(__file__), "baseline_scores.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
