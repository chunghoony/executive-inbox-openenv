#!/usr/bin/env python3
"""
Standalone REINFORCE training script for OpenEnv Executive Inbox with Unsloth.

Designed to mirror the notebook flow but run unattended overnight.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

# Must be set before importing unsloth.
os.environ.setdefault("UNSLOTH_ENABLE_FLEX_ATTENTION", "0")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import torch
from unsloth import FastLanguageModel

AVAILABLE_TIMES = ["9:00 AM", "10:00 AM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"]


def setup_openenv_paths() -> Path:
    """Resolve repo root and add OpenEnv paths to sys.path."""
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd.parent,
        script_dir.parent,
        Path("/home/jovyan/OpenEnv"),
    ]
    repo = next((c for c in candidates if (c / "envs" / "executive_inbox").exists()), None)
    if repo is None:
        raise FileNotFoundError(
            "OpenEnv repo not found. Run from repo root or adjust setup_openenv_paths()."
        )

    for p in [repo, repo / "src", repo / "envs"]:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    return repo


REPO_ROOT = setup_openenv_paths()

from torch.optim import AdamW

from executive_inbox import ExecutiveInboxEnv
from executive_inbox.models import ExecutiveInboxAction
from executive_inbox.server.executive_inbox_environment import ExecutiveInboxEnvironment


def parse_llm_action(response_text: str) -> ExecutiveInboxAction:
    """Parse model output into an ExecutiveInboxAction."""
    cleaned = response_text.strip()
    if cleaned.startswith('"action_type"'):
        cleaned = "{" + cleaned
    match = re.search(r"\{.*\}", cleaned.replace("\n", " "))
    if match:
        try:
            data = json.loads(match.group(0))
            return ExecutiveInboxAction(**data)
        except Exception:
            pass
    return ExecutiveInboxAction(action_type="unknown")


def compute_discounted_returns(
    rewards: list[float], gamma: float, device: torch.device
) -> torch.Tensor:
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    out = torch.tensor(returns, dtype=torch.float32, device=device)
    if len(out) > 1:
        out = (out - out.mean()) / (out.std() + 1e-8)
    return out


def format_observation(output: object, error: Optional[str]) -> str:
    """Convert env outputs into compact prompt text."""
    if error:
        return f"Error: {error}"
    if output is None:
        return "No output."
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=True)
    return str(output)


def summarize_output(output: object, error: Optional[str], limit: int = 240) -> str:
    text = format_observation(output, error)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


SYSTEM_PROMPT = """You are an executive assistant acting inside a tool-using environment.
Return exactly one valid JSON object for the next action.
Do not explain your reasoning.
Do not include markdown fences.
Do not invent IDs, email addresses, or calendar times."""

DEFAULT_REPLY_BODY = "I rescheduled the conflicting meeting and updated the schedule."


class EnvAdapter:
    """Unify local and remote environment access for the training loop."""

    MAX_STEPS = ExecutiveInboxEnvironment.MAX_STEPS

    def __init__(self, num_conflicts: int, base_url: Optional[str] = None):
        self._base_url = base_url
        self._client_cm = None
        self._client = None
        if base_url:
            self._client_cm = ExecutiveInboxEnv(base_url=base_url).sync()
            self._client = self._client_cm.__enter__()
            self._state = self._client.state()
            self._step_count = self._state.step_count
        else:
            self._env = ExecutiveInboxEnvironment(num_conflicts=num_conflicts)
            self._state = self._env.state
            self._step_count = self._env._step_count

    @property
    def state(self):
        return self._state

    def reset(self):
        if self._base_url:
            result = self._client.reset()
            self._refresh_state()
            return result.observation
        obs = self._env.reset()
        self._refresh_from_local()
        return obs

    def step(self, action: ExecutiveInboxAction):
        if self._base_url:
            result = self._client.step(action)
            self._refresh_state()
            return result.observation
        obs = self._env.step(action)
        self._refresh_from_local()
        return obs

    def close(self) -> None:
        if self._client_cm is not None:
            self._client_cm.__exit__(None, None, None)
            self._client_cm = None
            self._client = None

    def _refresh_state(self) -> None:
        if self._base_url:
            self._state = self._client.state()
            self._step_count = self._state.step_count

    def _refresh_from_local(self) -> None:
        self._state = self._env.state
        self._step_count = self._env._step_count


def format_state(env: EnvAdapter, history: list[str]) -> str:
    return f"""You are an Executive Assistant. Your goal is to resolve scheduling conflicts by checking the inbox and calendar.
You must solve BOTH conflicts before timeout.
Use only evidence from prior observations.
If you still need an ID or more context, gather more information first.

Available Actions (Output ONLY valid JSON, no prose):
1. {{"action_type": "read_inbox"}}
2. {{"action_type": "read_inbox", "email_id": "e_xxxx"}}
3. {{"action_type": "get_calendar"}}
4. {{"action_type": "move_meeting", "meeting_id": "c_xxxx", "new_time": "3:00 PM"}}
5. {{"action_type": "delegate_meeting", "meeting_id": "c_xxxx", "delegate_email": "owner@company.com"}}
6. {{"action_type": "reply_to_email", "email_id": "e_xxxx", "body": "I rescheduled the conflicting meeting and updated the schedule."}}
7. {{"action_type": "send_email", "to": "name@company.com", "subject": "Schedule updated", "body": "I moved or delegated the meeting and resolved the conflict."}}

State:
- Steps Taken: {env._step_count}/15
- Conflict Resolved: {env._state.conflict_resolved}
- Emails Sent: {env._state.emails_sent}
- Invalid Actions: {env._state.invalid_actions_taken}

Recent Interaction History:
{chr(10).join(history[-8:]) if history else "No prior actions yet."}

What is your next action JSON? Start with {{ and end with }}.
"""


def build_prompt(tokenizer, env: EnvAdapter, history: list[str]) -> str:
    user_prompt = format_state(env, history)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


def dedupe_actions(actions: list[ExecutiveInboxAction]) -> list[ExecutiveInboxAction]:
    unique: dict[str, ExecutiveInboxAction] = {}
    for action in actions:
        key = action.model_dump_json(exclude_none=True)
        unique.setdefault(key, action)
    return list(unique.values())


def extract_crisis_pairs(
    inbox_summary: list[dict[str, object]] | None,
    calendar_snapshot: list[dict[str, object]] | None,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    if inbox_summary is None:
        return []
    if calendar_snapshot is None:
        return []

    meetings_by_time: dict[str, list[dict[str, object]]] = {}
    for meeting in calendar_snapshot:
        meeting_time = str(meeting.get("time", ""))
        meetings_by_time.setdefault(meeting_time, []).append(meeting)

    conflict_times = [time for time, meetings in meetings_by_time.items() if len(meetings) >= 2]
    crisis_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for time in conflict_times:
        crisis_meeting = next(
            (
                meeting
                for meeting in meetings_by_time[time]
                if meeting.get("participants") and len(meeting.get("participants", [])) > 0
            ),
            None,
        )
        if crisis_meeting is None:
            continue
        participants = {str(p).lower() for p in crisis_meeting.get("participants", [])}
        crisis_email = next(
            (
                email
                for email in inbox_summary
                if str(email.get("sender", "")).lower() in participants
            ),
            None,
        )
        if crisis_email is not None:
            crisis_pairs.append((crisis_email, crisis_meeting))
    return crisis_pairs


def extract_conflict_meetings(
    calendar_snapshot: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    if calendar_snapshot is None:
        return []

    meetings_by_time: dict[str, list[dict[str, object]]] = {}
    for meeting in calendar_snapshot:
        meeting_time = str(meeting.get("time", ""))
        meetings_by_time.setdefault(meeting_time, []).append(meeting)

    conflict_meetings: list[dict[str, object]] = []
    for meetings in meetings_by_time.values():
        if len(meetings) >= 2:
            conflict_meetings.extend(meetings)
    return conflict_meetings


def extract_likely_crisis_email_ids(
    opened_emails: dict[str, dict[str, object]],
) -> set[str]:
    likely_ids: set[str] = set()
    executive_prefixes = (
        "boss@",
        "cto@",
        "vp.",
        "head.",
        "cmo@",
        "general.counsel@",
    )
    for email_id, email in opened_emails.items():
        sender = str(email.get("sender", "")).lower()
        body = str(email.get("body", "")).lower()
        if sender.startswith(executive_prefixes) or "delegate the meeting to chief.of.staff" in body:
            likely_ids.add(email_id)
    return likely_ids


def build_candidate_actions(
    inbox_summary: list[dict[str, object]] | None,
    calendar_snapshot: list[dict[str, object]] | None,
    opened_emails: dict[str, dict[str, object]],
    opened_email_ids: set[str],
    replied_email_ids: set[str],
    resolved_meeting_ids: set[str],
) -> list[ExecutiveInboxAction]:
    actions: list[ExecutiveInboxAction] = []
    if inbox_summary is None:
        actions.append(ExecutiveInboxAction(action_type="read_inbox"))
    else:
        for email in inbox_summary:
            email_id = str(email.get("id", ""))
            if email_id and email_id not in opened_email_ids:
                actions.append(
                    ExecutiveInboxAction(action_type="read_inbox", email_id=email_id)
                )
    if calendar_snapshot is None:
        actions.append(ExecutiveInboxAction(action_type="get_calendar"))
    if inbox_summary is None or calendar_snapshot is None:
        return dedupe_actions(actions)

    likely_crisis_email_ids = extract_likely_crisis_email_ids(opened_emails)
    candidate_meetings = [
        meeting
        for meeting in (extract_conflict_meetings(calendar_snapshot) or calendar_snapshot)
        if "chief.of.staff@company.com" in {str(p).lower() for p in meeting.get("participants", [])}
        or any("vip@" in str(p).lower() for p in meeting.get("participants", []))
    ]
    if not candidate_meetings:
        candidate_meetings = extract_conflict_meetings(calendar_snapshot) or calendar_snapshot
    occupied_times = {str(meeting.get("time", "")) for meeting in calendar_snapshot}
    free_times = [time for time in AVAILABLE_TIMES if time not in occupied_times]
    delegate_candidates = ["chief.of.staff@company.com"]

    for meeting in candidate_meetings:
        meeting_id = str(meeting.get("id", ""))
        meeting_time = str(meeting.get("time", ""))
        if meeting_id and meeting_id not in resolved_meeting_ids:
            for candidate_time in free_times:
                if candidate_time != meeting_time:
                    actions.append(
                        ExecutiveInboxAction(
                            action_type="move_meeting",
                            meeting_id=meeting_id,
                            new_time=candidate_time,
                        )
                    )
            for delegate_email in delegate_candidates:
                actions.append(
                    ExecutiveInboxAction(
                        action_type="delegate_meeting",
                        meeting_id=meeting_id,
                        delegate_email=delegate_email,
                    )
                )
    reply_candidate_ids = likely_crisis_email_ids or opened_email_ids
    for email_id in sorted(email_id for email_id in reply_candidate_ids if email_id):
        if email_id not in replied_email_ids:
            actions.append(
                ExecutiveInboxAction(
                    action_type="reply_to_email",
                    email_id=email_id,
                    body=DEFAULT_REPLY_BODY,
                )
            )

    if not actions:
        actions.extend(
            [
                ExecutiveInboxAction(action_type="read_inbox"),
                ExecutiveInboxAction(action_type="get_calendar"),
            ]
        )
    return dedupe_actions(actions)


def build_candidate_prompt(
    tokenizer,
    env: ExecutiveInboxEnvironment,
    history: list[str],
    candidate_actions: list[ExecutiveInboxAction],
) -> str:
    candidate_lines = "\n".join(
        f"{idx + 1}. {action.model_dump_json(exclude_none=True)}"
        for idx, action in enumerate(candidate_actions)
    )
    user_prompt = (
        format_state(env, history)
        + "\nValid candidate actions. Choose one of these exact JSON objects:\n"
        + candidate_lines
    )
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Choose exactly one valid candidate action JSON. Prefer actions backed by observed evidence.",
                },
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Choose exactly one valid candidate action JSON backed by observed evidence.\n\n{user_prompt}"


def score_candidate_actions(
    model,
    tokenizer,
    inputs,
    candidate_actions: list[ExecutiveInboxAction],
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    scores: list[torch.Tensor] = []
    action_texts: list[str] = []
    for action in candidate_actions:
        action_text = action.model_dump_json(exclude_none=True)
        action_texts.append(action_text)
        action_ids = tokenizer(
            action_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        full_ids = torch.cat([inputs.input_ids, action_ids], dim=1)
        full_mask = torch.ones_like(full_ids, device=device)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            out = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
        logits = out.logits[:, :-1, :]
        targets = full_ids[:, 1:]
        prompt_len = inputs.input_ids.shape[1]
        action_logits = logits[:, prompt_len - 1 :, :]
        action_targets = targets[:, prompt_len - 1 :]
        token_log_probs = torch.log_softmax(action_logits, dim=-1).gather(
            -1, action_targets.unsqueeze(-1)
        ).squeeze(-1)
        scores.append(token_log_probs.mean())
    return torch.stack(scores), action_texts


def infer_expert_action(
    inbox_summary: list[dict[str, object]] | None,
    calendar_snapshot: list[dict[str, object]] | None,
    known_crisis_email_ids: set[str],
    replied_email_ids: set[str],
    resolved_meeting_ids: set[str],
) -> ExecutiveInboxAction:
    if inbox_summary is None:
        return ExecutiveInboxAction(action_type="read_inbox")
    if calendar_snapshot is None:
        return ExecutiveInboxAction(action_type="get_calendar")

    meetings_by_time: dict[str, list[dict[str, object]]] = {}
    for meeting in calendar_snapshot:
        meeting_time = str(meeting.get("time", ""))
        meetings_by_time.setdefault(meeting_time, []).append(meeting)
    conflict_times = [time for time, meetings in meetings_by_time.items() if len(meetings) >= 2]
    occupied_times = {str(meeting.get("time", "")) for meeting in calendar_snapshot}
    free_times = [time for time in AVAILABLE_TIMES if time not in occupied_times]
    crisis_pairs = extract_crisis_pairs(inbox_summary, calendar_snapshot)

    for email, meeting in crisis_pairs:
        meeting_id = str(meeting.get("id", ""))
        meeting_time = str(meeting.get("time", ""))
        known_crisis_email_ids.add(str(email.get("id", "")))
        if meeting_id and meeting_id not in resolved_meeting_ids and meeting_time in conflict_times:
            candidate_time = next((time for time in free_times if time != meeting_time), None)
            if candidate_time:
                return ExecutiveInboxAction(
                    action_type="move_meeting",
                    meeting_id=meeting_id,
                    new_time=candidate_time,
                )
            delegate_email = str(email.get("sender", ""))
            if delegate_email:
                return ExecutiveInboxAction(
                    action_type="delegate_meeting",
                    meeting_id=meeting_id,
                    delegate_email=delegate_email,
                )

    for email_id in sorted(email_id for email_id in known_crisis_email_ids if email_id):
        if email_id and email_id not in replied_email_ids:
            return ExecutiveInboxAction(
                action_type="reply_to_email",
                email_id=email_id,
                body="I moved or delegated the meeting and resolved the conflict.",
            )

    return ExecutiveInboxAction(action_type="read_inbox")


class LocalRunLogger:
    def __init__(self, base_dir: Path, run_name: str, config: dict[str, object]):
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", run_name).strip("-") or "run"
        self.run_dir = base_dir / f"{timestamp}-{slug}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.steps_path = self.run_dir / "steps.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.status_path = self.run_dir / "status.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.steps_path.touch()
        self.episodes_path.touch()
        self.events_path.touch()
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        self.write_status({"phase": "initialized", "run_dir": str(self.run_dir)})

    def log_step(self, payload: dict[str, object]) -> None:
        with self.steps_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def log_episode(self, payload: dict[str, object]) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def write_summary(self, payload: dict[str, object]) -> None:
        self.summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def write_status(self, payload: dict[str, object]) -> None:
        self.status_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def log_event(self, payload: dict[str, object]) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def maybe_init_wandb(args: argparse.Namespace, model_name: str):
    if not args.wandb:
        return None
    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not set; disabling W&B logging.")
        return None
    try:
        import wandb

        wandb.login(relogin=False)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": model_name,
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "lr": args.learning_rate,
            },
            mode="online",
        )
        return wandb
    except Exception as exc:
        print(f"W&B disabled ({exc})")
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train REINFORCE on Executive Inbox with Unsloth")
    p.add_argument("--model-name", default="unsloth/smollm2-1.7b-instruct")
    p.add_argument(
        "--policy-mode",
        choices=["candidate", "freeform"],
        default="candidate",
        help="Use structured candidate-action RL or free-form JSON generation.",
    )
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=4)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=ExecutiveInboxEnvironment.MAX_STEPS)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--prompt-max-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--compute-oracle-loss", action="store_true")
    p.add_argument(
        "--fallback-penalty",
        type=float,
        default=0.10,
        help="Extra per-step penalty when the expert fallback is used.",
    )
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--no-offload-embedding", action="store_true")
    p.add_argument("--num-conflicts", type=int, default=2)
    p.add_argument(
        "--base-url",
        default=None,
        help="Optional deployed OpenEnv base URL, e.g. https://hoony-executive-inbox.hf.space",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="openenv-executive-inbox")
    p.add_argument("--wandb-run-name", default="unsloth-reinforce")
    p.add_argument("--save-dir", default=None, help="Optional directory to save model/tokenizer")
    p.add_argument("--log-dir", default="logs/executive_inbox_rl")
    p.add_argument("--run-name", default="executive-inbox-reinforce")
    p.add_argument("--greedy", action="store_true", help="Disable sampling during action generation")
    p.add_argument(
        "--no-json-prefill",
        action="store_true",
        help="Do not seed generation with an opening JSON brace.",
    )
    p.add_argument(
        "--no-expert-fallback",
        action="store_true",
        help="Disable heuristic fallback when the model emits an invalid action",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger = LocalRunLogger(
        REPO_ROOT / args.log_dir,
        args.run_name,
        {
            "model_name": args.model_name,
            "policy_mode": args.policy_mode,
            "max_seq_length": args.max_seq_length,
            "lora_rank": args.lora_rank,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "max_new_tokens": args.max_new_tokens,
            "prompt_max_tokens": args.prompt_max_tokens,
            "temperature": args.temperature,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "seed_offset": args.seed_offset,
            "eval_only": args.eval_only,
            "compute_oracle_loss": args.compute_oracle_loss,
            "fallback_penalty": args.fallback_penalty,
            "seed": args.seed,
            "no_expert_fallback": args.no_expert_fallback,
            "num_conflicts": args.num_conflicts,
            "base_url": args.base_url,
            "use_wandb": bool(args.wandb),
        },
    )
    print(f"Repo root: {REPO_ROOT}", flush=True)
    print(f"Local logs: {logger.run_dir}", flush=True)
    print(f"Loading model: {args.model_name}", flush=True)
    logger.log_event({"event": "startup", "model_name": args.model_name})
    logger.write_status(
        {
            "phase": "loading_model",
            "model_name": args.model_name,
            "run_dir": str(logger.run_dir),
        }
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        offload_embedding=not args.no_offload_embedding,
    )
    if args.lora_rank > 0:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_rank * 2,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    model.train()
    print(f"Model loaded on {device}")
    if device.type == "cuda":
        print(
            f"CUDA memory after load: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB allocated / "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GiB reserved"
        )

    env = EnvAdapter(num_conflicts=args.num_conflicts, base_url=args.base_url)
    try:
        if args.eval_only:
            honest_eval = args.no_expert_fallback and args.max_steps >= env.MAX_STEPS
            benchmark_msg = (
                "Primary benchmark mode enabled: eval-only with no fallback and full step budget."
                if honest_eval
                else "Warning: primary benchmark should use --eval-only --no-expert-fallback --max-steps 15."
            )
            print(benchmark_msg, flush=True)
            logger.log_event(
                {
                    "event": "benchmark_mode",
                    "honest_eval": honest_eval,
                    "max_steps": args.max_steps,
                    "no_expert_fallback": args.no_expert_fallback,
                }
            )
        optimizer = None if args.eval_only else AdamW(model.parameters(), lr=args.learning_rate)
        wandb = maybe_init_wandb(args, args.model_name)
        logger.write_status(
            {
                "phase": "evaluating" if args.eval_only else "training",
                "device": str(device),
                "model_name": args.model_name,
                "run_dir": str(logger.run_dir),
            }
        )

        all_rewards: list[float] = []
        completed_episodes = 0
        resolved_episodes = 0
        oracle_nlls: list[float] = []
        for episode in range(args.episodes):
            episode_seed = args.seed + args.seed_offset + episode
            random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
            obs = env.reset()
            saved_log_probs: list[torch.Tensor] = []
            rewards: list[float] = []
            episode_reward = 0.0
            prev_cumulative_reward = float(obs.reward)
            history: list[str] = [f"Reset -> {format_observation(obs.output, obs.error)}"]
            inbox_summary: list[dict[str, object]] | None = None
            calendar_snapshot: list[dict[str, object]] | None = None
            opened_emails: dict[str, dict[str, object]] = {}
            opened_email_ids: set[str] = set()
            known_crisis_email_ids: set[str] = set()
            replied_email_ids: set[str] = set()
            resolved_meeting_ids: set[str] = set()
            model_action_steps = 0
            fallback_steps = 0
            episode_oracle_nlls: list[float] = []

            for _step in range(args.max_steps):
                if args.policy_mode == "candidate":
                    candidate_actions = build_candidate_actions(
                        inbox_summary=inbox_summary,
                        calendar_snapshot=calendar_snapshot,
                        opened_emails=opened_emails,
                        opened_email_ids=opened_email_ids,
                        replied_email_ids=replied_email_ids,
                        resolved_meeting_ids=resolved_meeting_ids,
                    )
                    prompt = build_candidate_prompt(tokenizer, env, history, candidate_actions)
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.prompt_max_tokens,
                    ).to(device)
                    try:
                        scores, candidate_texts = score_candidate_actions(
                            model=model,
                            tokenizer=tokenizer,
                            inputs=inputs,
                            candidate_actions=candidate_actions,
                            device=device,
                        )
                    except (torch.OutOfMemoryError, RuntimeError) as exc:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        if isinstance(exc, RuntimeError) and "CUDA" not in str(exc):
                            raise
                        print(
                            f"Episode {episode + 1}: OOM during candidate scoring; skipping this step."
                        )
                        continue
                    selection_logits = scores
                    if not args.greedy and args.temperature > 0:
                        selection_logits = selection_logits / args.temperature
                    if args.greedy or len(candidate_actions) == 1:
                        selected_idx = int(torch.argmax(selection_logits).item())
                        log_prob = torch.log_softmax(selection_logits, dim=0)[selected_idx]
                    else:
                        dist = torch.distributions.Categorical(logits=selection_logits)
                        selected_idx = int(dist.sample().item())
                        log_prob = dist.log_prob(torch.tensor(selected_idx, device=device))
                    action = candidate_actions[selected_idx]
                    action_text = candidate_texts[selected_idx]
                    response_text = action_text
                    action_source = "model_candidate"
                    model_action_steps += 1
                    candidate_count = len(candidate_actions)
                    if args.compute_oracle_loss:
                        oracle_action = infer_expert_action(
                            inbox_summary=inbox_summary,
                            calendar_snapshot=calendar_snapshot,
                            known_crisis_email_ids=set(known_crisis_email_ids),
                            replied_email_ids=set(replied_email_ids),
                            resolved_meeting_ids=set(resolved_meeting_ids),
                        )
                        oracle_text = oracle_action.model_dump_json(exclude_none=True)
                        if oracle_text in candidate_texts:
                            oracle_idx = candidate_texts.index(oracle_text)
                            oracle_nll = float((-torch.log_softmax(selection_logits, dim=0)[oracle_idx]).item())
                            episode_oracle_nlls.append(oracle_nll)
                else:
                    prompt = build_prompt(tokenizer, env, history)
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.prompt_max_tokens,
                    ).to(device)
                    if not args.no_json_prefill:
                        prefix_ids = tokenizer(
                            "{",
                            return_tensors="pt",
                            add_special_tokens=False,
                        ).input_ids.to(device)
                        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=1)
                        if "attention_mask" in inputs:
                            prefix_mask = torch.ones_like(prefix_ids, device=device)
                            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], prefix_mask], dim=1)
                    generate_kwargs = {
                        "max_new_tokens": args.max_new_tokens,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    if args.greedy or args.temperature <= 0:
                        generate_kwargs["do_sample"] = False
                    else:
                        generate_kwargs["do_sample"] = True
                        generate_kwargs["temperature"] = args.temperature

                    try:
                        with torch.no_grad():
                            gen_tokens = model.generate(
                                **inputs,
                                **generate_kwargs,
                            )
                    except torch.OutOfMemoryError:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        print(
                            f"Episode {episode + 1}: OOM during generation; skipping this step "
                            f"(try smaller model or fewer tokens)."
                        )
                        continue
                    gen_only = gen_tokens[0][inputs.input_ids.shape[1] :]
                    response_text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
                    if not args.no_json_prefill:
                        response_text = "{" + response_text
                    parsed_action = parse_llm_action(response_text)
                    action_source = "model"
                    if parsed_action.action_type == "unknown" and not args.no_expert_fallback:
                        action = infer_expert_action(
                            inbox_summary=inbox_summary,
                            calendar_snapshot=calendar_snapshot,
                            known_crisis_email_ids=known_crisis_email_ids,
                            replied_email_ids=replied_email_ids,
                            resolved_meeting_ids=resolved_meeting_ids,
                        )
                        action_source = "expert_fallback"
                    else:
                        action = parsed_action
                    if action_source == "model":
                        model_action_steps += 1
                    else:
                        fallback_steps += 1
                    action_text = action.model_dump_json(exclude_none=True)
                    if action_source == "model" and parsed_action.action_type != "unknown":
                        score_ids = tokenizer(
                            action_text,
                            return_tensors="pt",
                            add_special_tokens=False,
                        ).input_ids.to(device)
                    else:
                        score_ids = gen_only.unsqueeze(0)
                        if score_ids.shape[1] == 0:
                            print(
                                f"Episode {episode + 1}: empty model generation; skipping this step."
                            )
                            continue

                    # Reinforce valid model actions directly; penalize fallback-triggering generations.
                    full_ids = torch.cat([inputs.input_ids, score_ids], dim=1)
                    full_mask = torch.ones_like(full_ids, device=device)
                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda"
                        else contextlib.nullcontext()
                    )
                    try:
                        with autocast_ctx:
                            out = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
                    except (torch.OutOfMemoryError, RuntimeError) as exc:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        if isinstance(exc, RuntimeError) and "CUDA" not in str(exc):
                            raise
                        print(
                            f"Episode {episode + 1}: OOM during log-prob forward; skipping this step."
                        )
                        continue
                    logits = out.logits[:, :-1, :]
                    targets = full_ids[:, 1:]
                    prompt_len = inputs.input_ids.shape[1]
                    gen_logits = logits[:, prompt_len - 1 :, :]
                    gen_targets = targets[:, prompt_len - 1 :]
                    token_log_probs = torch.log_softmax(gen_logits, dim=-1).gather(
                        -1, gen_targets.unsqueeze(-1)
                    ).squeeze(-1)
                    log_prob = token_log_probs.sum()
                    candidate_count = 0
                saved_log_probs.append(log_prob)
                obs = env.step(action)
                cumulative_reward = (
                    float(obs.reward.item()) if hasattr(obs.reward, "item") else float(obs.reward)
                )
                step_reward = cumulative_reward - prev_cumulative_reward
                if action_source == "expert_fallback":
                    step_reward -= args.fallback_penalty
                prev_cumulative_reward = cumulative_reward
                rewards.append(step_reward)
                episode_reward += step_reward
                if action.action_type == "read_inbox" and action.email_id is None and isinstance(obs.output, list):
                    if obs.output and isinstance(obs.output[0], dict) and "sender" in obs.output[0]:
                        inbox_summary = obs.output
                if action.action_type == "get_calendar" and isinstance(obs.output, list):
                    if obs.output and isinstance(obs.output[0], dict) and "time" in obs.output[0]:
                        calendar_snapshot = obs.output
                if action.action_type == "read_inbox" and action.email_id and not obs.error:
                    opened_email_ids.add(action.email_id)
                    if isinstance(obs.output, dict):
                        opened_emails[action.email_id] = obs.output
                if inbox_summary is not None and calendar_snapshot is not None:
                    known_crisis_email_ids.update(
                        str(email.get("id", ""))
                        for email, _meeting in extract_crisis_pairs(inbox_summary, calendar_snapshot)
                        if str(email.get("id", ""))
                    )
                if action.action_type == "reply_to_email" and not obs.error and action.email_id:
                    replied_email_ids.add(action.email_id)
                if action.action_type in {"move_meeting", "delegate_meeting"} and not obs.error and action.meeting_id:
                    resolved_meeting_ids.add(action.meeting_id)
                history.append(
                    f"Action: {action.model_dump_json(exclude_none=True)} | "
                    f"Observation: {format_observation(obs.output, obs.error)}"
                )
                logger.log_step(
                    {
                        "episode": episode + 1,
                        "step": len(rewards),
                        "response_text": response_text,
                        "action_source": action_source,
                        "action": action.model_dump(exclude_none=True),
                        "action_text": action_text,
                        "step_reward": step_reward,
                        "cumulative_reward": cumulative_reward,
                        "done": obs.done,
                        "error": obs.error,
                        "output_summary": summarize_output(obs.output, obs.error),
                        "invalid_actions": env.state.invalid_actions_taken,
                        "emails_sent": env.state.emails_sent,
                        "conflict_resolved": env.state.conflict_resolved,
                        "partial_conflicts_resolved": env.state.partial_conflicts_resolved,
                        "crisis_emails_opened": env.state.crisis_emails_opened,
                        "correct_meeting_moves": env.state.correct_meeting_moves,
                        "correct_replies": env.state.correct_replies,
                        "model_action_steps_so_far": model_action_steps,
                        "fallback_steps_so_far": fallback_steps,
                        "candidate_count": candidate_count,
                    }
                )
                if obs.done:
                    break

            if not saved_log_probs:
                print(f"Episode {episode + 1}: no sampled tokens, skipping optimization step.")
                continue

            returns = compute_discounted_returns(rewards, gamma=args.gamma, device=device)
            policy_loss = torch.stack([-lp * R for lp, R in zip(saved_log_probs, returns)]).sum()
            if not args.eval_only:
                try:
                    optimizer.zero_grad()
                    policy_loss.backward()
                    optimizer.step()
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if isinstance(exc, RuntimeError) and "CUDA" not in str(exc):
                        raise
                    print(f"Episode {episode + 1}: OOM during backward; skipping optimizer step.")
                    continue

            all_rewards.append(episode_reward)
            completed_episodes += 1
            if env.state.conflict_resolved:
                    resolved_episodes += 1
            if episode_oracle_nlls:
                oracle_nlls.append(sum(episode_oracle_nlls) / len(episode_oracle_nlls))
            recent_avg = sum(all_rewards[-10:]) / min(len(all_rewards), 10)
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"Steps: {len(rewards)} | Reward: {episode_reward:.2f} | "
                f"Loss: {policy_loss.item():.4f} | RecentAvg10: {recent_avg:.3f}"
            )
            logger.log_episode(
                {
                    "episode": episode + 1,
                    "reward": episode_reward,
                    "final_cumulative_reward": prev_cumulative_reward,
                    "loss": float(policy_loss.item()),
                    "steps": len(rewards),
                    "invalid_actions": env.state.invalid_actions_taken,
                    "emails_sent": env.state.emails_sent,
                    "conflict_resolved": env.state.conflict_resolved,
                    "partial_conflicts_resolved": env.state.partial_conflicts_resolved,
                    "crisis_emails_opened": env.state.crisis_emails_opened,
                    "correct_meeting_moves": env.state.correct_meeting_moves,
                    "correct_replies": env.state.correct_replies,
                    "timeout": env.state.is_timeout,
                    "model_action_steps": model_action_steps,
                    "fallback_steps": fallback_steps,
                    "fallback_rate": (fallback_steps / len(rewards)) if rewards else 0.0,
                    "oracle_nll": (sum(episode_oracle_nlls) / len(episode_oracle_nlls)) if episode_oracle_nlls else None,
                    "recent_avg_reward_10": recent_avg,
                }
            )
            if wandb is not None:
                wandb.log(
                    {
                        "episode": episode + 1,
                        "reward": episode_reward,
                        "final_cumulative_reward": prev_cumulative_reward,
                        "loss": float(policy_loss.item()),
                        "steps": len(rewards),
                        "invalid_actions": env.state.invalid_actions_taken,
                        "emails_sent": env.state.emails_sent,
                        "conflict_resolved": int(env.state.conflict_resolved),
                        "partial_conflicts_resolved": env.state.partial_conflicts_resolved,
                        "crisis_emails_opened": env.state.crisis_emails_opened,
                        "correct_meeting_moves": env.state.correct_meeting_moves,
                        "correct_replies": env.state.correct_replies,
                        "model_action_steps": model_action_steps,
                        "fallback_steps": fallback_steps,
                        "fallback_rate": (fallback_steps / len(rewards)) if rewards else 0.0,
                    }
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if wandb is not None:
            wandb.finish()
        print("Evaluation complete." if args.eval_only else "Training complete.", flush=True)

        if args.save_dir:
            out_dir = Path(args.save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"Saved model and tokenizer to: {out_dir}")

        if all_rewards:
            avg_reward = sum(all_rewards) / len(all_rewards)
            print(f"Average reward over {len(all_rewards)} episodes: {avg_reward:.3f}", flush=True)
            logger.write_summary(
                {
                    "episodes_requested": args.episodes,
                    "episodes_completed": completed_episodes,
                    "episodes_resolved": resolved_episodes,
                    "resolve_rate": resolved_episodes / completed_episodes if completed_episodes else 0.0,
                    "average_reward": avg_reward,
                    "best_reward": max(all_rewards),
                    "average_oracle_nll": (sum(oracle_nlls) / len(oracle_nlls)) if oracle_nlls else None,
                    "best_recent_avg_10": max(
                        (
                            sum(all_rewards[max(0, i - 9) : i + 1])
                            / len(all_rewards[max(0, i - 9) : i + 1])
                        )
                        for i in range(len(all_rewards))
                    ),
                    "log_dir": str(logger.run_dir),
                }
            )
            logger.write_status(
                {
                    "phase": "complete",
                    "average_reward": avg_reward,
                    "episodes_completed": completed_episodes,
                    "episodes_resolved": resolved_episodes,
                    "average_oracle_nll": (sum(oracle_nlls) / len(oracle_nlls)) if oracle_nlls else None,
                    "run_dir": str(logger.run_dir),
                }
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
