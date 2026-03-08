from typing import Any, Dict, List, Optional
import random
import string

from openenv.core.env_server import Environment

try:
    from models import ExecutiveInboxAction, ExecutiveInboxObservation, ExecutiveInboxState
except ModuleNotFoundError:
    from ..models import ExecutiveInboxAction, ExecutiveInboxObservation, ExecutiveInboxState

def semi_random_id(prefix: str) -> str:
    """Generate a stable but random-looking ID for scenarios."""
    chars = string.ascii_lowercase + string.digits
    return f"{prefix}_{''.join(random.choices(chars, k=4))}"


class ExecutiveInboxEnvironment(
    Environment[
        ExecutiveInboxAction, ExecutiveInboxObservation, ExecutiveInboxState
    ]
):
    """
    Simulates an executive inbox where an AI agent must handle calendar conflicts 
    and email threads. Built with procedurally generated state to prevent memorization.
    """
    MAX_STEPS = 15

    def __init__(self, num_conflicts: int = 2, dense_reward: bool = True):
        super().__init__()
        self._num_conflicts = max(1, min(num_conflicts, 2))
        self._dense_reward = dense_reward
        self._step_count = 0
        self._inbox = []
        self._sent_emails = []
        self._calendar = []
        self._done = False
        self._reward = 0.0
        self._state = ExecutiveInboxState()
        self._reset_environment()

    @property
    def name(self) -> str:
        return "ExecutiveInboxEnvironment"

    @property
    def state(self) -> ExecutiveInboxState:
        return self._state

    def reset(self) -> ExecutiveInboxObservation:
        self._reset_environment()
        return ExecutiveInboxObservation(
            output="Environment initialized.",
            done=self._done,
            reward=self._reward
        )

    def _reset_environment(self):
        """Internal reset to procedurally generate a fresh scenario."""
        self._step_count = 0
        self._state = ExecutiveInboxState()
        self._init_scenario()

    def _init_scenario(self):
        """Initialize procedural variables for the scenario."""
        from .data_pools import NOISE_EMAILS, NOISE_CALENDAR, CRISIS_TYPES, CRISIS_SUBJECTS, CRISIS_BODIES, PERSONAL_TYPES, PERSONAL_SENDERS, PERSONAL_SUBJECTS, PERSONAL_BODIES
        
        # 1. Base Variables
        times = ["9:00 AM", "10:00 AM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"]
        
        # Pick one or two distinct conflict times depending on the curriculum setting.
        self._conflict_times = random.sample(times, self._num_conflicts)
        
        domain = random.choice(["company.com", "corp.net", "enterprise.org"])
        vip_domain = random.choice(["bigcorp.com", "megacorp.com", "globaltech.io"])
        
        # 2. Pick the Scenarios
        crises = random.sample(CRISIS_TYPES, self._num_conflicts)
        conflicts = random.sample(PERSONAL_TYPES, self._num_conflicts)
        
        # Save validation targets
        self._targets = []
        for i in range(self._num_conflicts):
            self._targets.append({
                "time": self._conflict_times[i],
                "owner_email": f"{random.choice(crises[i]['owner_prefix'])}@{domain}",
                "meeting_id": semi_random_id("c"),
                "crisis_data": crises[i],
                "conflict_data": conflicts[i],
                "delegate_email": "chief.of.staff@company.com",
                "crisis_email_id": None,
                "crisis_subject": None,
                "personal_email_id": None,
                "personal_meeting_id": None,
                "delegated_to": None,
                "opened_crisis_email": False,
                "moved_correct_meeting": False,
                "replied_with_resolution": False,
                "resolved": False
            })
            
        # 3. Build Inbox
        self._inbox = []
        
        for target in self._targets:
            # Dynamic String Compilation
            c_subj = random.choice(CRISIS_SUBJECTS).format(time=target["time"])
            c_body = random.choice(CRISIS_BODIES).format(time=target["time"], vip_domain=vip_domain)
            c_body += (
                f" If you cannot attend, delegate the meeting to {target['delegate_email']}."
            )
            
            p_sender = random.choice(PERSONAL_SENDERS)
            p_subj = random.choice(PERSONAL_SUBJECTS).format(time=target["time"])
            p_body = random.choice(PERSONAL_BODIES).format(time=target["time"])

            # Schema Drift: 10% chance to lowercase subject
            if random.random() < 0.10: c_subj = c_subj.lower()
            if random.random() < 0.10: p_subj = p_subj.lower()

            # Crisis Email
            crisis_email = {
                "id": semi_random_id("e"),
                "sender": target["owner_email"],
                "subject": c_subj,
                "date": "Today 08:30 AM",
                "body": c_body
            }
            self._inbox.append(crisis_email)
            target["crisis_email_id"] = crisis_email["id"]
            target["crisis_subject"] = crisis_email["subject"]
            
            # Personal Conflict Email
            personal_email = {
                "id": semi_random_id("e"),
                "sender": p_sender,
                "subject": p_subj,
                "date": "Today 08:00 AM",
                "body": p_body
            }
            self._inbox.append(personal_email)
            target["personal_email_id"] = personal_email["id"]
        
        # Noise Emails (Pick 3-5 random ones)
        num_noise = random.randint(3, 5)
        for noise in random.sample(NOISE_EMAILS, num_noise):
            self._inbox.append({
                "id": semi_random_id("e"),
                "sender": noise["sender"],
                "subject": noise["subject"],
                "date": f"Today 0{random.randint(6,9)}:00 AM",
                "body": noise["body"]
            })
            
        random.shuffle(self._inbox)
        self._sent_emails = []
        
        # 4. Build Calendar
        self._calendar = []
        
        for target in self._targets:
            # Crisis Event
            self._calendar.append({
                "id": target["meeting_id"],
                "time": target["time"],
                "title": target["crisis_data"]["calendar_title"],
                "participants": [target["delegate_email"], f"vip@{vip_domain}"]
            })
            
            # Personal Event
            personal_meeting = {
                "id": semi_random_id("c"),
                "time": target["time"],
                "title": target["conflict_data"]["calendar_title"],
                "participants": [random.choice(PERSONAL_SENDERS)]
            }
            self._calendar.append(personal_meeting)
            target["personal_meeting_id"] = personal_meeting["id"]
        
        # Noise Events (Pick 2-3 randomly, assigned to non-conflict times)
        num_noise_events = random.randint(2, 3)
        available_times = [t for t in times if t not in self._conflict_times]
        
        if num_noise_events > len(available_times):
            num_noise_events = len(available_times)
            
        random_times_for_noise = random.sample(available_times, num_noise_events)
        
        for i, noise_cal in enumerate(random.sample(NOISE_CALENDAR, num_noise_events)):
            self._calendar.append({
                "id": semi_random_id("c"),
                "time": random_times_for_noise[i],
                "title": noise_cal["title"],
                "participants": noise_cal["participants"]
            })
            
        random.shuffle(self._calendar)
        
        self._done = False
        self._reward = 0.0

    def _add_reward(self, amount: float) -> None:
        if self._dense_reward:
            self._reward += amount

    def _find_target_for_email(self, email_id: str) -> Optional[Dict[str, Any]]:
        return next((target for target in self._targets if target["crisis_email_id"] == email_id), None)

    def _find_target_for_meeting(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        return next((target for target in self._targets if target["meeting_id"] == meeting_id), None)

    def _body_mentions_resolution(self, body: str) -> bool:
        body_text = body.lower()
        keywords = ["moved", "rescheduled", "delegate", "delegated", "updated the schedule"]
        return any(keyword in body_text for keyword in keywords)

    def _has_resolution_reply(self, target: Dict[str, Any]) -> bool:
        owner_email = str(target["owner_email"]).lower()
        crisis_subject = str(target.get("crisis_subject", "")).lower()
        for email in self._sent_emails:
            to = str(email.get("to", "")).lower()
            subject = str(email.get("subject", "")).lower()
            body = str(email.get("body", ""))
            if to != owner_email:
                continue
            if crisis_subject and crisis_subject not in subject:
                continue
            if self._body_mentions_resolution(body):
                return True
        return False

    def _validate_task(self):
        """Check if all overlapping conflicts are successfully resolved."""
        if self._done:
            return

        resolved_count = 0

        for target in self._targets:
            # Check if the crisis meeting was moved off the conflict time
            crisis_meeting = next((m for m in self._calendar if m["id"] == target["meeting_id"]), None)
            
            moved_off_conflict = bool(crisis_meeting and crisis_meeting["time"] != target["time"])
            delegated_correctly = (
                str(target.get("delegated_to", "")).lower()
                == str(target.get("delegate_email", "")).lower()
            )
            sent_resolution = self._has_resolution_reply(target)
            
            if (moved_off_conflict or delegated_correctly) and sent_resolution:
                resolved_count += 1
                if not target["resolved"]:
                    target["resolved"] = True
                    self._state.partial_conflicts_resolved += 1
                    self._add_reward(0.30)
                
        if resolved_count == len(self._targets):
            self._done = True
            self._state.conflict_resolved = True
            if self._dense_reward:
                self._reward += 0.70
            else:
                self._reward = 1.0

    def step(self, action: ExecutiveInboxAction) -> ExecutiveInboxObservation:
        """Process an action and return the observation and reward."""
        self._step_count += 1
        
        # Penalize for inefficient exploration
        self._reward -= 0.01

        # Check for Max Steps Limit Timeout
        if self._step_count >= self.MAX_STEPS:
            self._done = True
            self._state.is_timeout = True
            if self._dense_reward:
                self._reward = max(self._reward - 0.50, -1.0)
            else:
                self._reward = -1.0
            return ExecutiveInboxObservation(
                output=None,
                error=f"Timeout reached. Maximum of {self.MAX_STEPS} steps allowed.",
                done=self._done,
                reward=self._reward
            )

        output = None
        error = None
        
        # 1. read_inbox
        if action.action_type == "read_inbox":
            if action.email_id:
                email = next((e for e in self._inbox if e["id"] == action.email_id), None)
                if email:
                    output = email
                    target = self._find_target_for_email(action.email_id)
                    if target is not None and not target["opened_crisis_email"]:
                        target["opened_crisis_email"] = True
                        self._state.crisis_emails_opened += 1
                        self._add_reward(0.10)
                else:
                    self._reward -= 0.1
                    self._state.invalid_actions_taken += 1
                    error = f"Error: Email with ID {action.email_id} not found."
            else:
                # Summarize without bodies
                output = [{"id": e["id"], "sender": e["sender"], "subject": e["subject"]} for e in self._inbox]

        # 2. send_email
        elif action.action_type == "send_email":
            if not action.to or not action.subject or not action.body:
                self._reward -= 0.1
                self._state.invalid_actions_taken += 1
                error = "Error: Cannot send email without all fields (to, subject, body)."
            else:
                new_id = f"email_{len(self._sent_emails) + 1}"
                self._sent_emails.append({
                    "id": new_id, "to": action.to, "subject": action.subject, "body": action.body
                })
                self._state.emails_sent += 1
                self._validate_task()
                output = f"Email sent successfully to {action.to}"

        # 3. reply_to_email
        elif action.action_type == "reply_to_email":
            if not action.email_id or not action.body:
                self._reward -= 0.1
                self._state.invalid_actions_taken += 1
                error = "Error: Cannot reply without email_id and body."
            else:
                original_email = next((e for e in self._inbox if e["id"] == action.email_id), None)
                if not original_email:
                    self._reward -= 0.1
                    self._state.invalid_actions_taken += 1
                    error = f"Error: Email with ID {action.email_id} not found."
                else:
                    to = original_email["sender"]
                    subject = f"Re: {original_email['subject']}"
                    new_id = f"email_{len(self._sent_emails) + 1}"
                    self._sent_emails.append({
                        "id": new_id, "to": to, "subject": subject, "body": action.body
                    })
                    self._state.emails_sent += 1
                    target = self._find_target_for_email(action.email_id)
                    if (
                        target is not None
                        and self._body_mentions_resolution(action.body)
                        and not target["replied_with_resolution"]
                    ):
                        target["replied_with_resolution"] = True
                        self._state.correct_replies += 1
                        self._add_reward(0.15)
                    self._validate_task()
                    output = f"Replied to {to} successfully."

        # 4. get_calendar
        elif action.action_type == "get_calendar":
            output = self._calendar

        # 5. move_meeting
        elif action.action_type == "move_meeting":
            if not action.meeting_id or not action.new_time:
                self._reward -= 0.1
                self._state.invalid_actions_taken += 1
                error = "Error: Cannot move meeting without providing meeting_id and new_time."
            else:
                # Constraint Check
                occupied = next((m for m in self._calendar if m["time"] == action.new_time and m["id"] != action.meeting_id), None)
                if occupied:
                    self._reward -= 0.1
                    self._state.invalid_actions_taken += 1
                    error = f"Error: Cannot move meeting to {action.new_time}. That slot is already occupied by '{occupied['title']}'."
                else:
                    meeting_found = False
                    for meeting in self._calendar:
                        if meeting["id"] == action.meeting_id:
                            old_time = meeting["time"]
                            meeting["time"] = action.new_time
                            meeting_found = True
                            target = self._find_target_for_meeting(action.meeting_id)
                            if (
                                target is not None
                                and action.new_time != target["time"]
                                and not target["moved_correct_meeting"]
                            ):
                                target["moved_correct_meeting"] = True
                                self._state.correct_meeting_moves += 1
                                self._add_reward(0.20)
                            self._validate_task()
                            output = f"Meeting '{meeting['title']}' moved from {old_time} to {action.new_time}."
                            break
                    if not meeting_found:
                        self._reward -= 0.1
                        self._state.invalid_actions_taken += 1
                        error = f"Error: Meeting with ID {action.meeting_id} not found."

        # 6. delegate_meeting
        elif action.action_type == "delegate_meeting":
            if not action.meeting_id or not action.delegate_email:
                self._reward -= 0.1
                self._state.invalid_actions_taken += 1
                error = "Error: Cannot delegate meeting without meeting_id and delegate_email."
            else:
                target = next((t for t in self._targets if t["meeting_id"] == action.meeting_id), None)
                if target is None:
                    self._reward -= 0.1
                    self._state.invalid_actions_taken += 1
                    error = "Error: Only crisis meetings can be delegated."
                elif action.delegate_email.lower() != target["delegate_email"].lower():
                    self._reward -= 0.1
                    self._state.invalid_actions_taken += 1
                    error = (
                        "Error: Invalid delegate. Route crisis meetings to "
                        f"{target['delegate_email']}."
                    )
                else:
                    meeting_found = False
                    for i, meeting in enumerate(self._calendar):
                        if meeting["id"] == action.meeting_id:
                            title = meeting["title"]
                            self._calendar.pop(i)
                            target["delegated_to"] = action.delegate_email
                            if not target["moved_correct_meeting"]:
                                target["moved_correct_meeting"] = True
                                self._state.correct_meeting_moves += 1
                                self._add_reward(0.20)
                            meeting_found = True
                            self._validate_task()
                            output = f"Meeting '{title}' (ID {action.meeting_id}) successfully delegated to {action.delegate_email}."
                            break
                    if not meeting_found:
                        self._reward -= 0.1
                        self._state.invalid_actions_taken += 1
                        error = f"Error: Meeting with ID {action.meeting_id} not found on calendar."
                    
        else:
            self._reward -= 0.1
            self._state.invalid_actions_taken += 1
            error = f"Error: Unknown action type '{action.action_type}'"

        return ExecutiveInboxObservation(
            output=output,
            error=error,
            done=self._done,
            reward=self._reward
        )
