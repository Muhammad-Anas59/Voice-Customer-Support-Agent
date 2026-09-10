"""
retriever.py

Job: Take a customer question, find the most relevant policy chunks using
the saved FAISS index, then ask Gemini to draft an answer using ONLY those
retrieved chunks - with a citation to the source document. If nothing
relevant enough is found, it says so honestly instead of guessing.

Before generating an answer, retrieved chunks are checked for genuine
contradictions (see conflict_detector.py). If the sources disagree on the
same specific point, we escalate instead of silently blending both
versions into one answer.
"""

import os
import json
import numpy as np
import faiss
from google import genai
from google.genai.types import EmbedContentConfig
from dotenv import load_dotenv

from conflict_detector import detect_conflicts, log_conflict

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
# Below this similarity score, we don't trust the match enough to answer
CONFIDENCE_THRESHOLD = 0.55


def load_index(folder="app/data"):
    """Loads the saved FAISS index and chunk metadata from disk."""
    index = faiss.read_index(os.path.join(folder, "policy_index.faiss"))
    with open(os.path.join(folder, "chunks_metadata.json"), "r") as f:
        chunks = json.load(f)
    return index, chunks


def embed_query(text):
    """Embeds the customer's QUESTION (not a document) - Gemini optimizes
    this differently than document embeddings for better matching."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


def l2_distance_to_confidence(distance):
    """Converts FAISS's raw distance number into a simple 0-1 confidence
    score that's easier to reason about and show on a dashboard later."""
    return 1 / (1 + distance)


def retrieve_relevant_chunks(question, index, chunks, top_k=5):
    """Finds the top_k chunks most similar in meaning to the question.
    Returns a list of dicts with the chunk text, source, and confidence score."""
    query_vector = np.array([embed_query(question)]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunks[idx]
        results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "confidence": l2_distance_to_confidence(dist)
        })
    return results


def generate_answer(question, retrieved_chunks):
    """Asks Gemini to answer the question using ONLY the retrieved chunks.
    Explicitly instructed to refuse rather than guess if the chunks don't
    actually answer the question - this is the core anti-hallucination step.

    STYLE: prompted to reply like a real support agent (short, direct,
    conversational) rather than compiling every adjacent policy detail
    into a bulleted summary - see STYLE RULES below."""
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )

    prompt = f"""You are a customer support agent for Verve Athletics,
replying directly to a customer over email or chat.

Answer the customer's question using ONLY the policy text provided below.
Do not use any outside knowledge.

STYLE RULES - IMPORTANT:
- Write like a real support agent replying to one customer, not a
  compiled summary of every related policy.
- Only answer what was actually asked. Do not list every adjacent rule
  from the policy text just because it's nearby.
- If the question is general (e.g. "how long does shipping take") and
  multiple relevant options exist (e.g. shipping speeds), briefly list
  just the DOMESTIC options by default - do not include international
  rates/timelines unless the customer's question mentions international
  shipping, a country, or "abroad".
- Default to 1-3 short sentences. Only go longer if the question
  genuinely has multiple parts the customer asked about.
- Do NOT use bullet points, bold headers, or a "here's an overview"
  structure unless the customer's question truly has several distinct
  parts that need separating.
- Mention the source naturally in plain language if it's useful (e.g.
  "our return policy allows..."), not as a bracketed citation tag.
- Sound warm and direct, like a helpful person - not a policy document.

If the provided text does NOT actually answer the question, respond
exactly with: "I don't have enough information to answer this confidently."
Do not guess or make up an answer.

POLICY TEXT:
{context_text}

CUSTOMER QUESTION:
{question}

ANSWER:"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )
    return response.text


def answer_question(question, index, chunks):
    """Main entry point: retrieves relevant chunks, checks confidence,
    checks for conflicting sources, and either generates a grounded
    answer or escalates to a human."""
    retrieved = retrieve_relevant_chunks(question, index, chunks)
    top_confidence = retrieved[0]["confidence"] if retrieved else 0

    if top_confidence < CONFIDENCE_THRESHOLD:
        return {
            "answer": None,
            "escalated": True,
            "escalation_reason": "low_confidence",
            "reason": "No policy found with high enough confidence for this question.",
            "confidence": top_confidence,
            "sources": [],
            "conflicts": []
        }

    conflict_result = detect_conflicts(question, retrieved)

    if conflict_result["has_conflict"]:
        log_conflict(question, conflict_result)
        return {
            "answer": None,
            "escalated": True,
            "escalation_reason": "policy_conflict",
            "reason": "Our policy sources disagree on this point and need a human to confirm the correct answer.",
            "confidence": top_confidence,
            "sources": list(set(c["source"] for c in retrieved)),
            "conflicts": conflict_result["conflicts"]
        }

    answer_text = generate_answer(question, retrieved)

    return {
        "answer": answer_text,
        "escalated": False,
        "escalation_reason": None,
        "confidence": top_confidence,
        "sources": list(set(c["source"] for c in retrieved)),
        "conflicts": []
    }


if __name__ == "__main__":
    index, chunks = load_index()
    debug_mode = input("Show raw retrieved chunk text for debugging? (y/n): ").strip().lower() == "y"
    while True:
        question = input("\nAsk a customer support question (or 'quit'): ")
        if question.lower() == "quit":
            break

        if debug_mode:
            retrieved = retrieve_relevant_chunks(question, index, chunks)
            print("\n--- RAW RETRIEVED CHUNKS ---")
            for i, c in enumerate(retrieved):
                print(f"[{i}] source={c['source']} confidence={c['confidence']:.2f}")
                print(f"    text: {c['text']}\n")

        result = answer_question(question, index, chunks)
        print("\n--- RESULT ---")
        print(f"Escalated: {result['escalated']}")
        print(f"Confidence: {result['confidence']:.2f}")
        if result["escalated"]:
            print(f"Reason: {result['reason']}")
            if result["conflicts"]:
                print("Conflicting sources found:")
                for c in result["conflicts"]:
                    print(f"  - {c['topic']}: {c['detail']} (sources: {', '.join(c['sources'])})")
        else:
            print(f"Sources: {result['sources']}")
            print(f"\nAnswer:\n{result['answer']}")
