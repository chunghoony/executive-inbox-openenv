"""
Combinatorial Data Pools for the Executive Inbox Environment.
Provides modular templates to exponentially scale the scenario permutations during RL training.
"""

# ==============================================================================
# 1. CRISIS SCENARIOS (Work Overlaps)
# ==============================================================================

CRISIS_TYPES = [
    {"type": "Client Attrition", "calendar_title": "VIP Client Strategy Meeting", "owner_prefix": ["boss", "vp.sales", "head.revenue"]},
    {"type": "Server Outage", "calendar_title": "Prod Outage War Room", "owner_prefix": ["cto", "vp.eng", "head.infra"]},
    {"type": "PR Disaster", "calendar_title": "Comms & PR Review", "owner_prefix": ["head.of.pr", "cmo", "vp.comms"]},
    {"type": "Legal Threat", "calendar_title": "Legal Review (Feature Launch)", "owner_prefix": ["general.counsel", "head.legal", "vp.legal"]}
]

CRISIS_SUBJECTS = [
    "Important sync at {time}",
    "Required attendance: meeting at {time}",
    "Need everyone in the war room at {time}",
    "Strategy discussion - {time}",
    "Please join the {time} call",
    "Emergency Huddle - {time}",
    "All-hands crisis response ({time})",
    "Drop everything: sync at {time}",
    "Immediate attention required ({time})",
    "Briefing session at {time} sharp"
]

# Note: We omit explicit cheating keywords like "URGENT" or "EMERGENCY" where possible, 
# forcing the agent to rely on semantic context.
CRISIS_BODIES = [
    "We have a massive situation unfolding. I'm pulling the whole team into a room at {time}. Clear your calendar.",
    "The executive team is furious about the latest developments. We need a resolution plan drawn up by {time}. Be there.",
    "This is arguably the worst time for this to happen. I am assembling a response team at {time}. Do not miss this.",
    "I just got off the phone with the board. We need to do damage control immediately. Syncing at {time}.",
    "We are bleeding revenue by the minute. Everyone needs to be hands-on-keyboard in the conference room at {time}.",
    "The press is already starting to ask questions. We need to get our story straight. Meet me in my office at {time}.",
    "Our legal exposure here is massive. We are pulling everyone into a mandatory review session at {time}.",
    "The situation has escalated. We need an all-hands-on-deck response starting exactly at {time}."
]

# ==============================================================================
# 2. PERSONAL CONFLICTS (Immovable Personal Commitments)
# ==============================================================================

PERSONAL_TYPES = [
    {"type": "Recital", "calendar_title": "Daughter's Piano Recital"},
    {"type": "Flight", "calendar_title": "Flight Options (Do not schedule)"},
    {"type": "Doctor", "calendar_title": "Specialist Consultation"},
    {"type": "Contractor", "calendar_title": "Contractor Arrival Window"},
    {"type": "Dinner", "calendar_title": "Anniversary Dinner"},
    {"type": "Childcare", "calendar_title": "School Pickup (Coordination)"}
]

PERSONAL_SENDERS = [
    "school@academy.edu", "alerts@delta.com", "noreply@medical-portal.net", 
    "dispatch@plumbing.com", "reservations@opentable.com", "nanny@care.com",
    "frontdesk@pediatrics.org", "updates@united.com", "scheduling@hvac-pros.com"
]

PERSONAL_SUBJECTS = [
    "Reminder: Appointment at {time}",
    "Upcoming schedule: {time}",
    "Confirmation for {time}",
    "Your {time} itinerary",
    "Action Required: Coverage needed at {time}",
    "Arrival window confirmed ({time})",
    "Don't forget: today at {time}",
    "Reservation details - {time}"
]

PERSONAL_BODIES = [
    "This is a final confirmation that your appointment begins promptly at {time}. Please ensure you arrive 15 minutes early.",
    "We are writing to confirm your slot today at {time}. If you need to reschedule, you will be subject to a strict cancellation fee.",
    "Please remember that you need to be physically present at {time}. We cannot hold your spot if you are late.",
    "Your departure/arrival window is set for {time}. Let us know immediately if this changes, as we have no other availability this week.",
    "Just a quick reminder about the commitment at {time}. It's very important that you are on time today!",
    "I won't be able to handle things at {time} as planned. You will absolutely need to step in and cover this slot.",
    "The schedule is locked in for {time}. See you then!"
]


# ==============================================================================
# 3. NOISE EMAILS (Daily Office Chatter)
# ==============================================================================

NOISE_EMAILS = [
    {"subject": "Weekly Newsletter", "sender": "updates@producthunt.com", "body": "Here are the top 10 products of the week. Number 3 will shock you!"},
    {"subject": "Happy Hour on Friday?", "sender": "sarah.team@company.com", "body": "Hey everyone, are we still doing drinks at O'Malleys this Friday? Need a headcount."},
    {"subject": "Your Amazon.com order has shipped", "sender": "auto-confirm@amazon.com", "body": "Your order for 'Ergonomic Office Chair' has shipped and will arrive tomorrow."},
    {"subject": "Expense Report Not Yet Submitted", "sender": "finance-bot@company.com", "body": "You have 3 unsubmitted expenses from your trip to New York. Please submit by EOD."},
    {"subject": "Lunch in 10?", "sender": "mark.dev@company.com", "body": "Heading down to the cafeteria. Want me to grab you a sandwich?"},
    {"subject": "New login from Mac OS", "sender": "security@it-support.net", "body": "We detected a new login to your account from a Mac device. If this was you, ignore this email."},
    {"subject": "Re: Project Alpha timeline", "sender": "jessica.pm@company.com", "body": "I think we can hit the Q3 deadline if we cut the reporting feature. Thoughts?"},
    {"subject": "Can you review my PR?", "sender": "intern.bob@company.com", "body": "I just pushed the fix for the login bug. PR #4092. Let me know if it looks okay!"},
    {"subject": "Client prep notes for tomorrow", "sender": "vp.sales@company.com", "body": "Not urgent today. I just want your notes before tomorrow's client prep."},
    {"subject": "Draft talking points for next week's review", "sender": "cmo@company.com", "body": "Please send the draft deck when you have a chance. This is for next week, not today's fire drill."},
    {"subject": "Quick legal wording pass", "sender": "head.legal@company.com", "body": "Can you glance at the wording later this afternoon? No immediate action needed."},
    {"subject": "System Maintenance Tonight", "sender": "eng-leads@company.com", "body": "The staging databases will be down for maintenance between 1 AM and 3 AM tonight."},
    {"subject": "Did you leave your jacket in the conference room?", "sender": "office.manager@company.com", "body": "Someone left a blue Patagonia jacket in room 4B."}
]

NOISE_CALENDAR = [
    {"title": "Focus Time", "participants": []},
    {"title": "1:1 with Jane", "participants": ["jane.manager@company.com"]},
    {"title": "Weekly Team Sync", "participants": ["eng-team@company.com", "design-team@company.com"]},
    {"title": "Product Roadmap Review", "participants": ["execs@company.com", "jessica.pm@company.com"]},
    {"title": "Dentist Follow-up", "participants": []},
    {"title": "Interview: Frontend Engineer", "participants": ["recruiting@company.com", "candidate@gmail.com"]},
    {"title": "Vendor Pitch: CloudFlare", "participants": ["sales@cloudflare.com", "it-leads@company.com"]},
    {"title": "All-Hands Rehearsal", "participants": ["ceo@company.com", "comms@company.com"]},
    {"title": "Executive Check-in", "participants": ["vp.sales@company.com", "chief.of.staff@company.com"]},
    {"title": "Legal Draft Review", "participants": ["head.legal@company.com", "chief.of.staff@company.com"]},
    {"title": "Lunch / Run", "participants": []},
    {"title": "Architecture Brainstorm", "participants": ["mark.dev@company.com", "sarah.team@company.com"]}
]
