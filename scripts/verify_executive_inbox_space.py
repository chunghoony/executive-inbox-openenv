#!/usr/bin/env python3
"""
Smoke-test the deployed Executive Inbox OpenEnv Space.

This verifies that reset/step/state work over the remote WebSocket client and
prints a short trajectory that can also be reused as a packaging sanity check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def setup_paths() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd().resolve(),
        script_dir.parent,
        Path("/home/jovyan/OpenEnv"),
    ]
    repo = next((c for c in candidates if (c / "envs" / "executive_inbox").exists()), None)
    if repo is None:
        raise FileNotFoundError("OpenEnv repo not found.")
    for path in [repo, repo / "src", repo / "envs"]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo


REPO_ROOT = setup_paths()

from executive_inbox import ExecutiveInboxAction, ExecutiveInboxEnv


def summarize(value: object) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=True)
    else:
        text = str(value)
    return text if len(text) <= 300 else text[:297] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Executive Inbox HF Space")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EXECUTIVE_INBOX_BASE_URL", "https://hoony-executive-inbox.hf.space"),
        help="Remote OpenEnv base URL (add /web if HF Space uses base_path)",
    )
    args = parser.parse_args()

    print(f"Connecting to {args.base_url}")
    with ExecutiveInboxEnv(base_url=args.base_url).sync() as client:
        result = client.reset()
        print("reset:", summarize(result.observation.output))

        result = client.step(ExecutiveInboxAction(action_type="read_inbox"))
        print("read_inbox:", summarize(result.observation.output))

        inbox_items = result.observation.output if isinstance(result.observation.output, list) else []
        chosen_email_id = None
        for item in inbox_items:
            if not isinstance(item, dict):
                continue
            sender = str(item.get("sender", "")).lower()
            if sender.startswith(("boss@", "cto@", "vp.", "head.", "cmo@", "general.counsel@")):
                chosen_email_id = str(item.get("id", ""))
                break
        if chosen_email_id:
            result = client.step(
                ExecutiveInboxAction(action_type="read_inbox", email_id=chosen_email_id)
            )
            print("read_inbox(email):", summarize(result.observation.output))

        result = client.step(ExecutiveInboxAction(action_type="get_calendar"))
        print("get_calendar:", summarize(result.observation.output))

        state = client.state()
        print(
            "state:",
            json.dumps(
                {
                    "step_count": state.step_count,
                    "conflict_resolved": state.conflict_resolved,
                    "partial_conflicts_resolved": state.partial_conflicts_resolved,
                    "crisis_emails_opened": state.crisis_emails_opened,
                    "correct_meeting_moves": state.correct_meeting_moves,
                    "correct_replies": state.correct_replies,
                    "emails_sent": state.emails_sent,
                    "invalid_actions_taken": state.invalid_actions_taken,
                    "is_timeout": state.is_timeout,
                },
                ensure_ascii=True,
            ),
        )


if __name__ == "__main__":
    main()
