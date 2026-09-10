"""
conflict_detector.py

Job: Given the chunks retrieved for a customer question, check whether any
of them state DIFFERENT facts about the SAME specific policy point (e.g.
one document says "30 days", another says "14 days"). If so, flag it
instead of letting the answer-generation step quietly blend both into one
response.

This is deliberately conservative: it does NOT try to guess which source
is "correct". It flags the conflict and hands both versions to a human.
Auto-resolving in favor of one document is a good way to confidently tell
a customer the wrong thing.
"""

import os
import json
from datetime import datetime, timezone
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CONFLICT_MODEL = "gemini-3.5-flash-lite"
CONFLICT_LOG_PATH = os.path.join("app", "data", "conflicts_log.json")


def _build_conflict_prompt(question, chunks):
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    return f"""You are checking customer-support policy text for genuine contradictions.

You will see several text chunks pulled from different company documents, all
retrieved because they seem relevant to a customer's question. Your job is to
decide whether any of them make DIFFERENT factual claims about the SAME
specific point — the kind of contradiction a customer would notice and be
confused or upset by.

Flag it as a conflict ONLY if BOTH of these are true:
1. Two chunks answer the exact same specific point with different facts
   (a different number, deadline, yes/no, or eligibility rule for the
   same situation).
2. That specific point is what the customer is actually asking about.
   A retrieved chunk can contain other unrelated facts that happen to
   conflict with another chunk elsewhere in its text - ignore those.
   Only judge conflicts on the part of each chunk that is relevant to
   answering THIS customer's question.

Do NOT flag it if:
- The chunks are about different scenarios that merely sound similar
  (e.g. a manufacturing defect vs. shipping damage are different situations,
  even though both involve a "damaged" item).
- The chunks agree, even if worded differently.
- One chunk is simply more detailed than the other, without contradicting it.
- The disagreement is real but about a DIFFERENT point than what the
  customer asked (e.g. the customer asked about refund timing, and the
  chunks also happen to disagree about sale-item eligibility - that's a
  real conflict, but not this question's conflict, so don't report it here).

CUSTOMER QUESTION:
{question}

RETRIEVED CHUNKS:
{context_text}

Respond with ONLY valid JSON, no markdown fences, no other text, in this
exact shape:

{{
  "has_conflict": true or false,
  "conflicts": [
    {{
      "topic": "short label for the specific point in conflict",
      "sources": ["source_file_1", "source_file_2"],
      "detail": "one sentence stating what each source says differently"
    }}
  ]
}}

If has_conflict is false, "conflicts" must be an empty list."""


def detect_conflicts(question, chunks):
    """Checks the retrieved chunks for genuine contradictions.
    Returns a dict: {"has_conflict": bool, "conflicts": [...]}.
    Fails safe: if the check itself errors out or returns something
    unparseable, we treat it as no conflict detected rather than blocking
    the whole pipeline - but this is logged so it doesn't fail silently."""
    if len(chunks) < 2:
        # Can't have a conflict with only one source
        return {"has_conflict": False, "conflicts": []}

    prompt = _build_conflict_prompt(question, chunks)

    try:
        response = client.models.generate_content(
            model=CONFLICT_MODEL,
            contents=prompt
        )
        raw = response.text.strip()
        # Strip markdown fences if the model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        if "has_conflict" not in parsed or "conflicts" not in parsed:
            raise ValueError("Malformed conflict-check response shape")
        return parsed

    except Exception as e:
        print(f"[conflict_detector] Check failed, defaulting to no-conflict: {e}")
        return {"has_conflict": False, "conflicts": [], "check_error": str(e)}


def log_conflict(question, conflict_result):
    """Appends a caught conflict to a running log file. This is the raw
    data source for the 'conflicts caught' dashboard stat planned for
    Phase 3 - logging it now means we don't have to retrofit it later."""
    os.makedirs(os.path.dirname(CONFLICT_LOG_PATH), exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "conflicts": conflict_result["conflicts"]
    }

    existing = []
    if os.path.exists(CONFLICT_LOG_PATH):
        with open(CONFLICT_LOG_PATH, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.append(entry)
    with open(CONFLICT_LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)