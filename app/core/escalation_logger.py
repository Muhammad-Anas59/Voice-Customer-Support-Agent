"""
escalation_logger.py

Job: Whenever the assistant escalates (policy conflict, low confidence,
sentiment/urgency flag, order not found, API error), log it as a real
"ticket" with full context - not just a printed message that disappears.

This is a simple file-based ticket log (JSON lines file) - enough to prove
the concept and be genuinely useful. A real production version would send
this to email/Slack/a helpdesk tool instead, but the structure here is
built so swapping the notification method later is a small change, not a
rewrite.
"""

import os
import json
from datetime import datetime

TICKETS_FILE = "app/data/escalation_tickets.jsonl"


def log_escalation(customer_message, reason, category, conversation_state=None):
    """Appends one escalation record to the tickets file. Each line is a
    separate JSON object (JSON Lines format) so the file can grow safely
    without needing to re-read/re-write the whole thing each time.

    category should be one of: "policy_conflict", "low_confidence",
    "sentiment_urgency", "order_not_found", "api_error" - this lets a
    dashboard later group tickets by type."""

    os.makedirs(os.path.dirname(TICKETS_FILE), exist_ok=True)

    ticket = {
        "timestamp": datetime.now().isoformat(),
        "customer_message": customer_message,
        "reason": reason,
        "category": category,
        "conversation_context": conversation_state or {},
        "status": "open"
    }

    with open(TICKETS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(ticket) + "\n")

    return ticket


def get_all_tickets(status_filter=None):
    """Reads all logged tickets back. Optionally filter by status
    ('open' or 'resolved'). Used later by the admin panel/dashboard."""
    if not os.path.exists(TICKETS_FILE):
        return []

    tickets = []
    with open(TICKETS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ticket = json.loads(line)
                if status_filter is None or ticket["status"] == status_filter:
                    tickets.append(ticket)
    return tickets


def get_ticket_summary():
    """Quick counts by category - useful for the dashboard later without
    needing to build that yet."""
    tickets = get_all_tickets()
    summary = {}
    for t in tickets:
        cat = t["category"]
        summary[cat] = summary.get(cat, 0) + 1
    summary["total"] = len(tickets)
    return summary


if __name__ == "__main__":
    # Quick manual test: log a few fake escalations, then read them back
    log_escalation(
        customer_message="This is unacceptable, I'm contacting my lawyer",
        reason="Sentiment/urgency flag: legal threat mentioned",
        category="sentiment_urgency"
    )
    log_escalation(
        customer_message="What's your return policy?",
        reason="Policy sources disagree: 30 days vs 14 days",
        category="policy_conflict"
    )

    print("All tickets:")
    for t in get_all_tickets():
        print(f"  [{t['timestamp']}] ({t['category']}) {t['customer_message']}")

    print("\nSummary:")
    print(get_ticket_summary())
