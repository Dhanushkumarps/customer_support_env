# Customer Support OpenEnv Workflow

This document explains how a full episode flows through the environment, including scoring, turn management, and interactions between the agent and the REST API.

## 1. Episode Initialization

### Agent Action
The agent (or evaluation client) calls `POST /reset` with the desired task tier.
```json
// POST /reset
{
  "task_name": "hard",
  "seed": 42
}
```

### Environment Response
The `SupportEnvironment` selects 1 of the 15 scenarios based on the `seed`. It returns the initial observation.
```json
// Response
{
  "session_id": "a1b2c3d4",
  "observation": {
    "conversation": ["My app keeps crashing after the update."],
    "customer_query": "My app keeps crashing after the update.",
    "task_name": "hard",
    "done": false,
    "reward": null,
    "cumulative_reward": 0.0,
    "turn_scores": []
  }
}
```

## 2. Agent Interaction Loop (Multi-Turn)

The agent now enters a loop, proposing actions until `done: true` is returned.

### Turn 1: Clarification (Agent Action)
```json
// POST /step
{
  "session_id": "a1b2c3d4",
  "message": "I apologize for the inconvenience. Could you please share your OS version?",
  "intent": "clarify"
}
```

### Turn 1: Environment Grader & Response
The hard Grader detects the `?` question mark and rewards the agent based on relevance. It then simulates the customer's response.
```json
// Response
{
  "observation": {
    "conversation": [
      "My app keeps crashing after the update.",
      "I apologize for the inconvenience. Could you please share your OS version?",
      "I'm on iOS 17.5."
    ],
    "customer_query": "I'm on iOS 17.5.",
    "done": false,
    "reward": 0.40,
    "cumulative_reward": 0.40,
    "turn_scores": [0.40]
  }
}
```

### Turn 2: Resolution (Agent Action)
```json
// POST /step
{
  "session_id": "a1b2c3d4",
  "message": "Thank you. Please try reinstalling the app and clearing the cache.",
  "intent": "respond"
}
```

### Turn 2: Environment Grader & Response
The hard Grader searches for keywords like `reinstall` and `cache`, detects empathetic phrasing, and calculates the reward.
```json
// Response
{
  "observation": {
    "conversation": [
       "...", 
       "Ok I'll try reinstalling. Thanks."
    ],
    "customer_query": "Ok I'll try reinstalling. Thanks.",
    "done": false,
    "reward": 0.60,
    "cumulative_reward": 0.50, // (0.40 + 0.60) / 2
    "turn_scores": [0.40, 0.60]
  }
}
```

### Turn 3: Closing (Agent Action)
```json
// POST /step
{
  "session_id": "a1b2c3d4",
  "message": "Happy to help! Is there anything else you need?",
  "intent": "close"
}
```

### Turn 3: Environment Grader & Response (Terminal)
The hard Grader detects a polite closing phrase (`Happy to help!`), rewards the agent, and marks the episode as `done`.
```json
// Response
{
  "observation": {
    "done": true,
    "reward": 0.30,
    "cumulative_reward": 0.4333,
    "turn_scores": [0.40, 0.60, 0.30],
    "info": "Episode already completed."
  }
}
```


## 3. Final Scoring

Once the episode terminates, the agent can call `/grader` to retrieve its finalized multi-dimensional score.

```json
// POST /grader
{
  "session_id": "a1b2c3d4"
}
```

```json
// Response
{
  "score": 0.4333,
  "task": "hard",
  "steps": 3,
  "cumulative_reward": 0.4333,
  "turn_scores": [0.40, 0.60, 0.30],
  "resolved": false,
  "issue_type": "technical"
}
```
