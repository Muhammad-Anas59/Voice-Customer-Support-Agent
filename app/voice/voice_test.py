"""
voice_test.py

Speak a question -> AssemblyAI transcribes it -> your real RAG pipeline
(assistant.get_response) generates an answer -> answer prints on screen.

Still text-only output for now. TTS (speaking the answer back) comes next.
"""

import sys
import os
import uuid
import threading
import pyaudio

# app/voice/ -> app/core/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from assistant import get_response
from retriever import load_index

from assemblyai.streaming.v3 import (
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    TurnEvent,
)

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set ASSEMBLYAI_API_KEY in your environment")

SAMPLE_RATE = 16000
CHUNK = 800  # 50ms of audio at 16kHz

print("Loading RAG index...")
index, chunks = load_index()
print(f"Loaded {len(chunks)} chunks. Ready.")

# One conversation for this test session
session_id = str(uuid.uuid4())
conversation_state = {}


def handle_question(question):
    """Runs in a background thread so it never blocks the AssemblyAI
    connection's keepalive while Gemini is generating an answer."""
    try:
        result = get_response(question, conversation_state, index, chunks)
        answer = result["answer"]
    except Exception:
        import traceback
        traceback.print_exc()
        answer = "Sorry, I hit an error processing that."

    print(f"[AGENT ANSWER] {answer}", flush=True)
    print("Listening again...", flush=True)


def on_turn(_, event: TurnEvent):
    if not event.end_of_turn:
        return  # ignore partials for now, keep output clean

    question = event.transcript.strip()
    if not question:
        return

    print(f"[YOU SAID] {question}", flush=True)
    print("[thinking...]", flush=True)
    # Fire off the slow part in the background, return immediately so the
    # connection's keepalive keeps responding.
    threading.Thread(target=handle_question, args=(question,), daemon=True).start()


def main():
    client = RealTimeTranscriber(
        RealTimeTranscriberOptions(api_key=API_KEY)
    )
    client.on(RealTimeEvents.Turn, on_turn)
    client.connect(
        RealTimeParameters(
            sample_rate=SAMPLE_RATE,
            speech_model="universal-3-5-pro",
            mode="balanced",
        )
    )

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("Listening... ask a policy question or about an order. Ctrl+C to stop.")
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            client.stream(data)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        client.disconnect(terminate=True)


if __name__ == "__main__":
    main()
