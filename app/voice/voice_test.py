"""
voice_test.py

Speak a question -> AssemblyAI transcribes it -> your real RAG pipeline
(assistant.get_response) generates an answer -> answer is spoken back out
loud via ElevenLabs (and still printed to screen too).

Day 2: TTS is now wired in. Still a standalone script, not the full
Flask + SocketIO app - that's Day 3-4.
"""

import sys
import os
import uuid
import threading
import pyaudio

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

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

load_dotenv()

ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    raise RuntimeError("Set ASSEMBLYAI_API_KEY in your environment or .env file")

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise RuntimeError("Set ELEVENLABS_API_KEY in your .env file")

SAMPLE_RATE = 16000
CHUNK = 800  # 50ms of audio at 16kHz

# Same voice/model as the isolated tts_test.py - keep these in sync,
# or move both to a shared config module once Day 3-4 restructuring happens.
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George"
MODEL_ID = "eleven_flash_v2_5"     # low-latency, for real-time conversation

tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Set while the agent's own answer is playing through speakers, so the mic
# picking up that audio doesn't get transcribed and treated as a new
# question (feedback loop). Cleared after playback finishes, plus a short
# buffer to catch the tail end of the audio still in the air.
agent_speaking = threading.Event()

print("Loading RAG index...")
index, chunks = load_index()
print(f"Loaded {len(chunks)} chunks. Ready.")

# One conversation for this test session
session_id = str(uuid.uuid4())
conversation_state = {}


def speak(text):
    """Convert text to speech and play it through speakers. Runs inside
    the same background thread as the RAG call, so it never blocks the
    AssemblyAI connection's keepalive.

    Sets agent_speaking while audio is playing (plus a short buffer after)
    so on_turn ignores the agent hearing itself through the mic."""
    agent_speaking.set()
    try:
        audio = tts_client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format="mp3_44100_128",
        )
        play(audio)
    except Exception as e:
        print(f"[TTS] Failed to speak answer: {e}", flush=True)
    finally:
        # Small buffer so the tail of the audio still in the air (and any
        # in-flight partial transcript from it) doesn't sneak through.
        threading.Timer(0.5, agent_speaking.clear).start()


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
    speak(answer)
    print("Listening again...", flush=True)


def on_turn(_, event: TurnEvent):
    if not event.end_of_turn:
        return  # ignore partials for now, keep output clean

    if agent_speaking.is_set():
        return  # agent is currently talking - ignore the mic hearing itself

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
        RealTimeTranscriberOptions(api_key=ASSEMBLYAI_API_KEY)
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
