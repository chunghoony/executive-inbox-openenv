# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Executive Inbox Environment.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field

# We use openenv.core.env_server.types for State based on the previous working file
from openenv.core.env_server.types import State

class ExecutiveInboxAction(BaseModel):
    """Native action schema for the Executive Inbox Environment."""
    action_type: str = Field(..., description="Action to perform (e.g., 'read_inbox', 'move_meeting', 'delegate_meeting')")
    
    # Optional arguments depending on the tool chosen
    email_id: Optional[str] = None
    to: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    meeting_id: Optional[str] = None
    new_time: Optional[str] = None
    delegate_email: Optional[str] = None

class ExecutiveInboxObservation(BaseModel):
    """Native observation schema passed back to the RL policy."""
    output: Any = Field(description="The functional output string or dict array from the environment tool.")
    error: Optional[str] = Field(default=None, description="Detailed error message if the action failed.")
    done: bool = Field(default=False, description="Whether the episode has terminated.")
    reward: float = Field(default=0.0, description="The reward accumulated on this step.")

class ExecutiveInboxState(State):
    """Episode state metadata for the Executive Inbox environment."""
    conflict_resolved: bool = Field(default=False, description="Whether the schedule conflict is resolved")
    partial_conflicts_resolved: int = Field(default=0, description="Number of individual conflict pairs resolved")
    emails_sent: int = Field(default=0, description="Number of emails sent")
    crisis_emails_opened: int = Field(default=0, description="Number of true crisis emails opened")
    correct_meeting_moves: int = Field(default=0, description="Number of correct crisis meetings moved or delegated")
    correct_replies: int = Field(default=0, description="Number of correct crisis-thread replies with resolution language")
    invalid_actions_taken: int = Field(default=0, description="Counter for hallucinated or invalid tool calls")
    is_timeout: bool = Field(default=False, description="Whether the episode ended due to hitting the max step limit")
