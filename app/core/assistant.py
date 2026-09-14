"""
assistant.py

Job: This is the single entry point a customer actually talks to. It ties
together the three pieces built so far:
- router.py       -> decides if the question is about an order or a policy
- order_lookup.py -> answers order-status questions with real Shopify data
- retriever.py     -> answers policy questions from the RAG pipeline

The customer just asks a question in plain language; this file figures out
internally which engine should handle it and returns one clean answer.
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "integrations"))

from router import classify_question, extract_order_details, parse_order_number, parse_amount
from retriever import load_index, answer_question
from order_lookup import find_order
from sentiment_detector import check_sentiment
from escalation_logger import log_escalation
from interaction_logger import log_interaction


# --- Chit-chat detection (greetings/thanks/goodbyes) ---
# Catches simple pleasantries BEFORE they ever reach the RAG pipeline.
# Without this, a short message like "thanks for your time" was handled
# inconsistently - sometimes retrieval happened to land near a similar-
# sounding FAQ chunk and got a real answer, sometimes it didn't and got
# escalated as a failed policy lookup. Neither outcome is wrong exactly,
# but relying on embedding luck for something this simple is fragile.
#
# Deliberately conservative: splits the message on commas and classifies
# each piece separately - only treats the WHOLE message as chit-chat if
# EVERY piece is a recognized pleasantry. This handles compound phrases
# like "Hi, how are you?" or "Okay, goodbye." correctly, while a message
# like "thanks, what's your return policy?" still falls through to real
# routing untouched, since "what's your return policy" isn't in the
# recognized pleasantry list.

_ATOMIC_PATTERNS = [
    (re.compile(r"^(hi|hello|hey)$", re.IGNORECASE), "greeting"),
    (re.compile(r"^good (morning|afternoon|evening)$", re.IGNORECASE), "greeting"),
    (re.compile(r"^how('?s| is| are) (it going|you( doing)?)$", re.IGNORECASE), "wellbeing"),
    (re.compile(r"^what'?s (up|going on|new)$", re.IGNORECASE), "casual_opener"),
    (re.compile(r"^(ok(ay)?|alright|well)$", re.IGNORECASE), "filler"),
    (re.compile(
        r"^(thanks?( you)?|thank you)( so much| very much)?"
        r"( for (your|the) (time|help|information|kindness))?$",
        re.IGNORECASE
    ), "thanks"),
    (re.compile(r"^i (just )?(said|say) thank(s| you)$", re.IGNORECASE), "thanks"),
    (re.compile(r"^(got it|sounds good|cool|great|perfect|no worries)$", re.IGNORECASE), "ack"),
    (re.compile(r"^(bye|goodbye|see you( later)?)$", re.IGNORECASE), "farewell"),
    (re.compile(r"^that('?s| is) all( for now)?$", re.IGNORECASE), "farewell"),
    (re.compile(r"^nothing else$", re.IGNORECASE), "farewell"),
    (re.compile(r"^i'?m good$", re.IGNORECASE), "farewell"),
    (re.compile(r"^no(,)? (that'?s all|thanks)$", re.IGNORECASE), "farewell"),
]

_CHITCHAT_REPLIES = {
    "wellbeing": "I'm doing great, thanks for asking! How can I help you today?",
    "casual_opener": "Not much on my end - ready to help whenever you need it! What can I do for you today?",
    "greeting": "Hi there! How can I help you today?",
    "thanks": "You're very welcome! Let us know if you need anything else.",
    "farewell": "Take care! Feel free to reach out anytime you need help.",
    "ack": "Sounds good! Let us know if you need anything else.",
    "filler": "Sounds good! Let us know if you need anything else.",
}
# Priority order when a message has multiple recognized pieces (e.g.
# "Hi, how are you?" -> greeting + wellbeing -> wellbeing's reply wins,
# since it's the more specific/substantive of the two).
_CATEGORY_PRIORITY = ["wellbeing", "casual_opener", "greeting", "thanks", "farewell", "ack", "filler"]


def _classify_segment(segment):
    segment = segment.strip().rstrip("!.?").strip()
    for pattern, category in _ATOMIC_PATTERNS:
        if pattern.match(segment):
            return category
    return None


def detect_chitchat(question):
    """Returns a canned reply if the ENTIRE message (every comma-separated
    piece of it) is a recognized pleasantry, otherwise None so the caller
    falls through to real routing."""
    text = question.strip()
    if not text or len(text.split()) > 8:
        return None

    pieces = [p for p in text.rstrip("!.?").split(",")]
    categories = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        category = _classify_segment(piece)
        if category is None:
            return None  # one unrecognized piece means this isn't pure chit-chat
        categories.append(category)

    if not categories:
        return None

    for preferred in _CATEGORY_PRIORITY:
        if preferred in categories:
            return _CHITCHAT_REPLIES[preferred]

    return None


def reset_flow_state(conversation_state):
    """Clears order-flow-specific fields (awaiting, order_number) WITHOUT
    touching the persistent conversation history - history should survive
    for the whole session even after an order lookup completes and its
    flow-specific fields get reset, since a human resolving an escalation
    later in the same session still needs the full back-and-forth."""
    conversation_state.pop("awaiting", None)
    conversation_state.pop("order_number", None)


def handle_order_question(question, conversation_state):
    """Handles an order-status question. If order number or total is
    missing, asks a follow-up instead of guessing or escalating - this
    mirrors how a real support agent would respond.

    Parsing depends on what we're currently waiting for:
    - Nothing yet (fresh message): use extract_order_details, which
      handles an open-ended message that might contain an order number,
      a total, both, or neither, and is deliberately strict about the
      total needing a decimal point to avoid confusing it with an order
      number when both appear together.
    - Specifically awaiting an order number or a total: use the lenient
      single-purpose parsers instead, since we already know the ENTIRE
      reply is answering one specific question - this is what fixes the
      bug where a plain reply like '80' (no decimal) would never be
      recognized as an answer to 'what was the total charged?' and would
      leave the customer stuck being asked the same question forever.
    - If a reply doesn't contain what we're looking for at all, we don't
      keep asking - the conversation state resets so the next message
      gets classified fresh, giving the customer an escape valve instead
      of a dead end.
    """
    awaiting = conversation_state.get("awaiting")

    if awaiting == "order_number":
        order_number = parse_order_number(question)
        if not order_number:
            reset_flow_state(conversation_state)
            return {
                "answer": "I couldn't find an order number in that - no worries, what else can I help with?",
                "escalated": False,
                "needs_followup": False
            }
        conversation_state["order_number"] = order_number
        conversation_state["awaiting"] = "total"
        return {
            "answer": "Thanks - and to verify it's your order, what was the total amount charged?",
            "escalated": False,
            "needs_followup": True
        }

    if awaiting == "total":
        total = parse_amount(question)
        if not total:
            reset_flow_state(conversation_state)
            return {
                "answer": "I couldn't find an amount in that - no worries, what else can I help with?",
                "escalated": False,
                "needs_followup": False
            }
        order_number = conversation_state.get("order_number")

    else:
        # Fresh message - not currently mid-flow
        order_number, total = extract_order_details(question)

        if not order_number:
            conversation_state["awaiting"] = "order_number"
            return {
                "answer": "Sure, I can check that for you - what's your order number?",
                "escalated": False,
                "needs_followup": True
            }

        if not total:
            conversation_state["order_number"] = order_number
            conversation_state["awaiting"] = "total"
            return {
                "answer": "Thanks - and to verify it's your order, what was the total amount charged?",
                "escalated": False,
                "needs_followup": True
            }

    result = find_order(order_number, total)
    reset_flow_state(conversation_state)  # verification done, reset flow fields (history preserved)

    if not result["found"]:
        log_escalation(
            customer_message=question,
            reason="Order lookup failed - no match found for provided order number and total",
            category="order_not_found",
            conversation_state=conversation_state
        )
        log_interaction(question, handler="order", resolved=False, category="order_not_found")
        return {
            "answer": "I couldn't find an order matching those details. Could you double check the order number and total? If it still doesn't match, I'll get a human to help.",
            "escalated": False,
            "needs_followup": False
        }

    reply = f"Your order {result['order_number']} is currently: {result['status'].replace('_', ' ')}."
    if result["tracking_number"]:
        reply += f" Tracking number: {result['tracking_number']} via {result['carrier']}. Track it here: {result['tracking_url']}"
    else:
        reply += " It hasn't shipped yet, so there's no tracking number available."

    log_interaction(question, handler="order", resolved=True)

    return {
        "answer": reply,
        "escalated": False,
        "needs_followup": False
    }


def handle_policy_question(question, index, chunks, conversation_state):
    """Handles a general policy question via the existing RAG pipeline."""
    result = answer_question(question, index, chunks)

    # A third escalation path: retrieval confidence was high enough to
    # proceed, but the LLM itself decided the retrieved text doesn't
    # actually answer the question, and returns this exact fallback
    # string as a normal (non-escalated) answer. Catch it here and treat
    # it the same as a real escalation instead of showing the raw
    # internal string to the customer.
    NO_ANSWER_FALLBACK = "I don't have enough information to answer this confidently."
    llm_declined = (not result["escalated"]) and result.get("answer", "").strip() == NO_ANSWER_FALLBACK

    if result["escalated"] or llm_declined:
        reason = result.get("reason", "") or "The retrieved policy text didn't actually address this question."
        if "disagree" in reason.lower() or "conflict" in reason.lower():
            answer = "I found conflicting information in our policies on this - I don't want to give you the wrong answer, so I'm flagging this for a team member to confirm and get back to you."
            category = "policy_conflict"
        else:
            answer = "I don't have a confident answer to that from our policies - I'm flagging this for a team member to follow up with you."
            category = "low_confidence"

        log_escalation(
            customer_message=question,
            reason=reason,
            category=category,
            conversation_state=conversation_state
        )
        log_interaction(question, handler="policy", resolved=False,
                         confidence=result.get("confidence"), category=category)

        return {
            "answer": answer,
            "escalated": True,
            "reason": reason,
            "needs_followup": False
        }

    log_interaction(question, handler="policy", resolved=True, confidence=result.get("confidence"))

    return {
        "answer": result["answer"],
        "escalated": False,
        "needs_followup": False
    }


def get_response(question, conversation_state, index, chunks):
    """Public entry point: runs the real routing logic, then always
    appends this turn to a persistent conversation history before
    returning - regardless of which internal path handled the question.
    Wrapping it this way means history capture can't be forgotten or
    skipped by any individual handler, and doesn't require threading
    extra bookkeeping through every branch of the routing logic below."""
    response = _get_response_inner(question, conversation_state, index, chunks)

    history = conversation_state.setdefault("history", [])
    history.append({"customer": question, "assistant": response.get("answer")})

    return response


def _get_response_inner(question, conversation_state, index, chunks):
    """Classifies the question, then routes it to the right handler.
    conversation_state persists across turns so follow-up answers (like
    'my order number is 1001') are understood in context, and so a full
    conversation history is available if this session later escalates."""

    # If we're mid-way through collecting order details, treat this
    # message as answering that, not as a brand new question to classify
    if conversation_state.get("awaiting"):
        return handle_order_question(question, conversation_state)

    # Sentiment/urgency check runs FIRST, before any routing - anger,
    # urgency, or legal/safety language always escalates immediately,
    # regardless of whether a confident answer could otherwise be given
    sentiment = check_sentiment(question)
    if sentiment["flagged"]:
        reason = f"Sentiment/urgency flag: {sentiment['reason']}"
        log_escalation(
            customer_message=question,
            reason=reason,
            category="sentiment_urgency",
            conversation_state=conversation_state
        )
        log_interaction(question, handler="sentiment_block", resolved=False, category="sentiment_urgency")
        return {
            "answer": "I can see this is important and want to make sure it's handled properly - I'm escalating this to a team member right away.",
            "escalated": True,
            "reason": reason,
            "needs_followup": False
        }

    # Chit-chat check runs BEFORE routing/RAG - a plain "thanks" or "hi"
    # should never be treated as a failed policy lookup.
    chitchat_reply = detect_chitchat(question)
    if chitchat_reply:
        log_interaction(question, handler="chitchat", resolved=True)
        return {
            "answer": chitchat_reply,
            "escalated": False,
            "needs_followup": False
        }

    label = classify_question(question)

    if label == "order":
        return handle_order_question(question, conversation_state)
    else:
        return handle_policy_question(question, index, chunks, conversation_state)


if __name__ == "__main__":
    index, chunks = load_index()
    conversation_state = {}

    print("Verve Athletics Support Assistant (type 'quit' to exit)\n")
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        try:
            response = get_response(question, conversation_state, index, chunks)
            print(f"Assistant: {response['answer']}\n")
        except Exception as e:
            log_escalation(
                customer_message=question,
                reason=f"API/system error: {e}",
                category="api_error"
            )
            print("Assistant: I'm having trouble processing that right now - I'm flagging this for a team member to follow up with you directly.\n")
            print(f"[internal error log: {e}]\n")
