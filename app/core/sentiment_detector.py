"""
sentiment_detector.py

Job: Check an incoming customer message for anger, urgency, or legal/safety
language BEFORE it goes to the policy or order pipeline. If detected, the
message gets escalated to a human immediately - regardless of whether a
confident policy or order answer exists. Some things should never be
fully automated, and this is where that judgment lives.
"""

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHAT_MODEL = "gemini-3.5-flash-lite"


def check_sentiment(message):
    """Asks Gemini to flag the message if it shows real anger, urgency,
    or legal/safety language. Returns a dict: {flagged: bool, reason: str}.
    Kept conservative on purpose - flags clear cases, not mild negativity,
    so it doesn't escalate every slightly unhappy customer."""

    prompt = f"""Read the customer message below and decide if it should be
immediately escalated to a human support agent, BEFORE any automated
answer is given.

Flag it as escalate=true ONLY if the message clearly shows:
- Genuine anger or frustration (not just a neutral complaint)
- Urgency suggesting real harm or a time-critical problem (e.g. safety issue, medical reaction, allergic reaction, being charged incorrectly or double-charged)
- Legal language (e.g. mentions of a lawyer, lawsuit, chargeback threat, "reporting this")
- Explicit threats or self-harm/safety concerns of any kind

Do NOT flag a message just because it's a normal question, even if slightly
annoyed (e.g. "this is taking forever" is NOT enough on its own to flag,
but "this is unacceptable, I'm contacting my lawyer" IS).

Respond in EXACTLY this format, nothing else:
escalate: true OR false
reason: <one short sentence explaining why, or "none" if not escalating>

Customer message: "{message}"
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )
    text = response.text.strip()

    flagged = False
    reason = "none"

    for line in text.splitlines():
        line = line.strip().lower()
        if line.startswith("escalate:"):
            flagged = "true" in line
        elif line.startswith("reason:"):
            reason = line.split("reason:", 1)[1].strip()

    return {"flagged": flagged, "reason": reason}


if __name__ == "__main__":
    test_messages = [
        "This is taking forever, when will my order arrive?",
        "This is absolutely unacceptable, I am contacting my lawyer and disputing this charge.",
        "What's your return policy?",
        "I had an allergic reaction to the fabric in your shirt and my skin is burning right now.",
        "Can I exchange my shoes for a different size?"
    ]
    for msg in test_messages:
        result = check_sentiment(msg)
        print(f"'{msg}'\n  -> flagged={result['flagged']}, reason={result['reason']}\n")
