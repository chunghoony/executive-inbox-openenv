# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Executive Inbox Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import ExecutiveInboxAction, ExecutiveInboxObservation, ExecutiveInboxState


class ExecutiveInboxEnv(
    EnvClient[ExecutiveInboxAction, ExecutiveInboxObservation, ExecutiveInboxState]
):
    """
    Client for the Executive Inbox Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ExecutiveInboxEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.output)
        ...
        ...     result = client.step(ExecutiveInboxAction(action_type="get_calendar"))
        ...     print(result.observation.output)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ExecutiveInboxEnv.from_docker_image("executive_inbox-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ExecutiveInboxAction(action_type="get_calendar"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ExecutiveInboxAction) -> Dict:
        """
        Convert ExecutiveInboxAction to JSON payload for step message.
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ExecutiveInboxObservation]:
        """
        Parse server response into StepResult[ExecutiveInboxObservation].
        """
        obs_data = payload.get("observation", {})
        
        observation = ExecutiveInboxObservation(
            output=obs_data.get("output", None),
            error=obs_data.get("error", None),
            done=obs_data.get("done", payload.get("done", False)),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ExecutiveInboxState:
        """
        Parse server response into ExecutiveInboxState object.
        """
        return ExecutiveInboxState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            conflict_resolved=payload.get("conflict_resolved", False),
            partial_conflicts_resolved=payload.get("partial_conflicts_resolved", 0),
            emails_sent=payload.get("emails_sent", 0),
            crisis_emails_opened=payload.get("crisis_emails_opened", 0),
            correct_meeting_moves=payload.get("correct_meeting_moves", 0),
            correct_replies=payload.get("correct_replies", 0),
            invalid_actions_taken=payload.get("invalid_actions_taken", 0),
            is_timeout=payload.get("is_timeout", False)
        )
