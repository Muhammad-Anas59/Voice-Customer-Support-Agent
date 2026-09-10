"""
interaction_logger.py

Job: Log EVERY interaction the assistant handles - not just escalations
(that's what escalation_logger.py does). This is the broader record that
answers "how many questions came in, and what fraction did the assistant
resolve on its own" - the core number any support-automation pitch needs.

Kept separate from escalation_logger.py on purpose: escalations need rich
detail for a human to act on, while this just needs a lightweight record
of outcome + category for aggregate stats. Combining them would mean every
resolved answer carries escalation-shaped fields it doesn't need.
"""

import os
import json
from datetime import datetime

INTERACTIONS_FILE = "app/data/interactions.jsonl"


def log_interaction(question, handler, resolved, confidence=None, category=None):
    """Appends one interaction record. Called for EVERY question the
    assistant processes, whether it resolved it or escalated.

    handler: "policy", "order", or "sentiment_block" - which part of the
      pipeline handled this
    resolved: True if the customer got a real answer, False if escalated
    confidence: the RAG confidence score, if applicable (None for order
      questions, which don't have this concept)
    category: for escalations, the specific reason category (matches
      escalation_logger's categories); None for resolved interactions
    """
    os.makedirs(os.path.dirname(INTERACTIONS_FILE), exist_ok=True)

    # Confidence may arrive as a numpy float32 (from FAISS) - json.dumps
    # can't serialize numpy types directly, so cast to a plain Python float
    if confidence is not None:
        confidence = float(confidence)

    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "handler": handler,
        "resolved": resolved,
        "confidence": confidence,
        "category": category
    }

    with open(INTERACTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def get_all_interactions():
    """Reads every logged interaction back."""
    if not os.path.exists(INTERACTIONS_FILE):
        return []

    interactions = []
    with open(INTERACTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                interactions.append(json.loads(line))
    return interactions


def get_analytics_summary():
    """Computes the aggregate stats a dashboard actually needs:
    total volume, resolution rate, breakdown by handler/category, and a
    simple estimated time saved (rough industry-standard assumption:
    each auto-resolved ticket saves ~4 minutes of agent time - this is a
    reasonable, clearly-labeled estimate, not a precise measurement)."""
    interactions = get_all_interactions()
    total = len(interactions)

    if total == 0:
        return {
            "total_interactions": 0,
            "resolved_count": 0,
            "escalated_count": 0,
            "resolution_rate": 0,
            "by_handler": {},
            "by_escalation_category": {},
            "avg_confidence": None,
            "estimated_minutes_saved": 0
        }

    resolved = [i for i in interactions if i["resolved"]]
    escalated = [i for i in interactions if not i["resolved"]]

    by_handler = {}
    for i in interactions:
        h = i["handler"]
        by_handler[h] = by_handler.get(h, 0) + 1

    by_escalation_category = {}
    for i in escalated:
        c = i.get("category") or "unknown"
        by_escalation_category[c] = by_escalation_category.get(c, 0) + 1

    confidences = [i["confidence"] for i in interactions if i.get("confidence") is not None]
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else None

    MINUTES_SAVED_PER_RESOLVED_TICKET = 4
    estimated_minutes_saved = len(resolved) * MINUTES_SAVED_PER_RESOLVED_TICKET

    return {
        "total_interactions": total,
        "resolved_count": len(resolved),
        "escalated_count": len(escalated),
        "resolution_rate": round(len(resolved) / total * 100, 1),
        "by_handler": by_handler,
        "by_escalation_category": by_escalation_category,
        "avg_confidence": avg_confidence,
        "estimated_minutes_saved": estimated_minutes_saved
    }


if __name__ == "__main__":
    # Quick manual test
    log_interaction("How long does shipping take?", "policy", resolved=True, confidence=0.64)
    log_interaction("Where's my order 1002?", "order", resolved=True)
    log_interaction("What's your return policy?", "policy", resolved=False, confidence=0.66, category="policy_conflict")
    log_interaction("This is unacceptable, contacting my lawyer", "sentiment_block", resolved=False, category="sentiment_urgency")

    print("Summary:")
    import json as j
    print(j.dumps(get_analytics_summary(), indent=2))
