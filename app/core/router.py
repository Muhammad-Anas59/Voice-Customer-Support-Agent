"""
router.py

Job: Look at an incoming customer message and decide where it should go:
- "order" -> the customer is asking about a specific order's status/tracking
  -> handled by order_lookup.py (live Shopify data)
- "policy" -> a general question about shipping, returns, warranty, etc.
  -> handled by retriever.py (RAG pipeline over policy documents)

This is the piece that makes the system feel like ONE assistant instead of
two separate scripts - the customer just asks a question, and the system
figures out internally which engine should answer it.
"""

import os
import re
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHAT_MODEL = "gemini-3.5-flash-lite"


def classify_question(question):
    """Asks Gemini to classify the question as 'order' or 'policy'.
    Kept to a single strict word output so it's cheap and easy to parse."""
    prompt = f"""Classify the customer message below into exactly one word:

"order" - the customer wants to CHECK THE CURRENT STATUS of a specific
  order they placed - a direct status/tracking lookup.
  Examples: "where's my order", "has my package shipped",
  "when will #1001 arrive", "track order 1002"

"policy" - the customer is asking what to do, what the rules are, or how
  a situation is handled - even if it mentions a package, order, or
  shipping. This includes asking for guidance about a problem (e.g. a
  package that seems delayed or lost), not just checking its status.
  Examples: "what's your return policy", "how long does shipping take",
  "can I get a refund", "my package hasn't moved in 10 days, what
  should I do", "what happens if my order never arrives"

The key distinction: "order" = "tell me the status of my order right now."
"policy" = "what's the process / what should I do / what are the rules" -
even when a specific order or package is mentioned.

Respond with ONLY the single word "order" or "policy" - nothing else.

Customer message: "{question}"

Classification:"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )
    label = response.text.strip().lower()

    # Safety net: if Gemini returns anything unexpected, default to
    # "policy" - the RAG pipeline safely escalates on no-match anyway,
    # while guessing "order" wrongly would ask for irrelevant details.
    return "order" if "order" in label else "policy"


def extract_order_details(question):
    """Tries to pull an order number and a dollar amount out of the
    message using simple pattern matching. Returns (order_number, total)
    - either can be None if not found, and the caller should ask for
    whichever is missing rather than guessing.

    Deliberately strict (total requires a decimal point) because this
    function parses an OPEN-ENDED message that might contain an order
    number, a total, both, or neither - without the decimal requirement,
    a bare number like '1002' could be wrongly captured as the total
    instead of the order number when both appear in one message."""
    order_match = re.search(r"#?\s?(\d{3,6})\b", question)
    total_match = re.search(r"\$?\s?(\d+\.\d{2})\b", question)

    order_number = order_match.group(1) if order_match else None
    total = total_match.group(1) if total_match else None

    return order_number, total


def parse_order_number(text):
    """Lenient order-number parser - use ONLY when we already know the
    entire message is answering 'what's your order number', so there's
    no ambiguity with a total to worry about. Accepts a bare number
    ('1001') or one with a leading #."""
    match = re.search(r"#?\s?(\d{3,6})\b", text)
    return match.group(1) if match else None


def parse_amount(text):
    """Lenient amount parser - use ONLY when we already know the entire
    message is answering 'what was the total charged', so there's no
    ambiguity with an order number to worry about. Unlike
    extract_order_details, this does NOT require a decimal point -
    accepts plain replies like '80' or '200', not just '80.00', since a
    real customer is likely to type it either way."""
    match = re.search(r"\$?\s?(\d+(?:\.\d{1,2})?)\b", text)
    return match.group(1) if match else None


if __name__ == "__main__":
    test_questions = [
        "Where's my order #1001?",
        "What's your return policy?",
        "Has order 1002 shipped yet, it cost me $80.00",
        "How long do refunds take?",
        "Do you sell yoga mats?",
        "My package hasn't moved in 10 days, what should I do?",
        "What happens if my order never arrives?"
    ]
    for q in test_questions:
        label = classify_question(q)
        order_num, total = extract_order_details(q)
        print(f"'{q}' -> {label} (order_number={order_num}, total={total})")
