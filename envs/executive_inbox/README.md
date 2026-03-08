---
title: Executive Inbox Environment (RL)
emoji: ⏱️
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Executive Inbox Environment — Native Architecture

A robust text-based OpenEnv environment built on top of the native PyTorch `openenv.core.env_server.Environment` class architecture. This environment simulates a stressful day in the life of an Executive Assistant or busy executive, forcing the `Action` space to parse emails, check calendars, reschedule personal conflicts around work emergencies, and reply via thread IDs.

Developed strictly to align with OpenEnv Statement 3.2 (Personalized Tasks & Email/Calendar Parsing).

## Quick Start

You can test a programmatic AI policy (looping over actions to find the solution) by running the evaluator test:

```bash
uv run python examples/heuristic_policy.py
```
This loop runs **10 distinct episodes** against our generative procedural combinatorics.

You can also run the deterministic single-step example:
```bash
uv run python examples/executive_inbox_example.py
```

## Features for Deep Reinforcement Learning

The environment was ripped off the `FastMCP` architecture and rebuilt precisely to support Reinforcement Learning evaluations:

1. **Native Observations & Validations**: 
   - We utilize `ExecutiveInboxAction(action_type=...)` and our `step()` function returns a structured `ExecutiveInboxObservation(output: Any, error: str, done: bool, reward: float)`.
2. **Procedural Complexity (No Memorizing)**:
   - Built on `data_pools.py`, we randomly sample combinations of "Crisis Emails", "Personal Conflicts", and "Noise Events" and drop them randomly into a 9-to-5 calendar layout every single episode. 
   - All email IDs (e.g. `e_1x9w`) and meeting IDs (`c_00k4`) are randomly generated on `reset()`.
3. **Double Overlaps**:
   - The environment purposefully seeds **two separate conflicts** occurring simultaneously at distinct times.
   - For example: *A VIP crisis at 10:00 AM overlaps a flight, and a server outage at 3:00 PM overlaps a doctor's appointment.*
4. **Rich Constraints**:
   - `move_meeting` enforces that the new meeting block cannot be placed on an occupied time block (like 2:00 PM), failing gracefully and penalizing the agent.
5. **Delegations Tools**:
   - Includes full implementations of `reply_to_email` and `delegate_meeting` to fulfill the "Personalized Constraints" and "Delegation" feedback block from evaluators.

## Reward & Shaping Mechanics

- Max Step count is set to **15 steps**. Reaching the limit terminates the episode with `Done=True` and `Reward=-1.0`.
- Every valid tool call returns a `-0.01` negative penalty to enforce efficient operations (the heuristic optimal solver takes 6 steps to win).
- Every **Invalid Action** (missing parameters, or hallucinating an `email_id` that does not exist in the current random layout) throws a `-0.1` invalid step penalty and increments the verification `state.invalid_actions_taken`.
- Resolving **Both Conflicts** and sending **Both Confirmatory Emails** terminates the episode with `Done=True` and awards `+1.0`.

## Architecture

```
executive_inbox/
├── models.py                   # Pydantic schemas (ExecutiveInboxAction/Observation)
├── client.py                   # Model dump overrides for local testing
├── server/
│   ├── executive_inbox_environment.py # Native environment loop + validations
│   ├── data_pools.py           # Combinatorial random pools of scenarios
│   └── app.py                  # OpenEnv FastAPI app
└── README.md
```
