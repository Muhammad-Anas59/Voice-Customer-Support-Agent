"""
app/api/server.py

Day 3-4: real Flask + Flask-SocketIO app replacing the standalone
voice_test.py script. Browser sends mic audio over a websocket instead of
pyaudio reading a local mic, so this works as an actual web demo.

Per-connection (per browser tab) state:
    - its own AssemblyAI RealTimeTranscriber
    - its own conversation_state dict for the RAG pipeline
Keyed by Socket.IO's request.sid, same pattern as the original Flask app's
`sessions` dict keyed by session_id.

Run:
    python app/api/server.py
Then open http://localhost:5000 in a browser.
"""

import os
import sys
import base64
import threading
import time

from dotenv import load_dotenv
from flask import Flask, render_template
from flask_socketio import SocketIO
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play as _unused_play  # noqa: F401 (not used server-side; kept for parity)

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
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ASSEMBLYAI_API_KEY or not ELEVENLABS_API_KEY:
    raise RuntimeError("Set ASSEMBLYAI_API_KEY and ELEVENLABS_API_KEY in .env")

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_flash_v2_5"
SAMPLE_RATE = 16000

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "..", "templates"),
)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

print("Loading RAG index...")
index, chunks = load_index()
print(f"Loaded {len(chunks)} chunks. Ready.")

# sid -> {"transcriber": ..., "conversation_state": {...}, "agent_speaking": bool}
sessions = {}


@app.route("/")
def home():
    return render_template("index.html")


def make_transcriber(sid):
    """One AssemblyAI real-time connection per browser tab."""

    def on_turn(_, event: TurnEvent):
        sess = sessions.get(sid)
        if not sess:
            return

        if not event.end_of_turn:
            socketio.emit("partial_transcript", {"text": event.transcript}, to=sid)
            return

        if sess["agent_speaking"]:
            return  # ignore the agent hearing its own TTS output

        question = event.transcript.strip()
        if not question:
            return

        socketio.emit("final_transcript", {"text": question}, to=sid)
        threading.Thread(target=handle_question, args=(sid, question), daemon=True).start()

    client = RealTimeTranscriber(RealTimeTranscriberOptions(api_key=ASSEMBLYAI_API_KEY))
    client.on(RealTimeEvents.Turn, on_turn)

    params = RealTimeParameters(
        sample_rate=SAMPLE_RATE, speech_model="universal-3-5-pro", mode="balanced"
    )

    # AssemblyAI's connection occasionally hits a transient SSL handshake
    # timeout (network-level flakiness, not a code bug - it self-recovers
    # on a fresh attempt). Retry a few times with backoff before giving up,
    # so a one-off network blip doesn't kill the session.
    last_error = None
    for attempt in range(1, 4):
        try:
            client.connect(params)
            return client
        except Exception as e:
            last_error = e
            print(f"[session {sid}] AssemblyAI connect attempt {attempt} failed: {e}")
            if attempt < 3:
                socketio.emit(
                    "status", {"text": f"Connecting to speech service (retry {attempt})..."}, to=sid
                )
                time.sleep(attempt * 2)  # 2s, then 4s

    socketio.emit(
        "connection_error",
        {"text": "Couldn't connect to the speech service after several tries. Please refresh and try again."},
        to=sid,
    )
    raise last_error


def handle_question(sid, question):
    sess = sessions.get(sid)
    if not sess:
        return

    try:
        result = get_response(question, sess["conversation_state"], index, chunks)
        answer = result["answer"]
    except Exception:
        import traceback
        traceback.print_exc()
        answer = "Sorry, I hit an error processing that."

    socketio.emit("agent_text", {"text": answer}, to=sid)

    sess["agent_speaking"] = True
    try:
        audio = tts_client.text_to_speech.convert(
            text=answer, voice_id=VOICE_ID, model_id=MODEL_ID, output_format="mp3_44100_128"
        )
        audio_bytes = b"".join(audio)  # SDK returns a generator of chunks
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        socketio.emit("agent_audio", {"audio_b64": b64, "mime": "audio/mpeg"}, to=sid)
    except Exception as e:
        print(f"[TTS] Failed: {e}")
    finally:
        # Client tells us when playback actually finishes (see agent_audio_done
        # handler below) - that's more accurate than guessing a fixed delay
        # server-side, since MP3 duration varies with answer length.
        pass


@socketio.on("connect")
def on_connect():
    sid = request_sid()
    try:
        transcriber = make_transcriber(sid)
    except Exception:
        # connection_error was already emitted inside make_transcriber;
        # don't create a session entry for a transcriber that never connected.
        return
    sessions[sid] = {
        "transcriber": transcriber,
        "conversation_state": {},
        "agent_speaking": False,
    }
    print(f"[session {sid}] connected")


@socketio.on("disconnect")
def on_disconnect():
    sid = request_sid()
    sess = sessions.pop(sid, None)
    if sess:
        try:
            sess["transcriber"].disconnect(terminate=True)
        except Exception:
            pass
    print(f"[session {sid}] disconnected")


@socketio.on("audio_chunk")
def on_audio_chunk(data):
    """data is raw PCM16 mono 16kHz bytes sent from the browser."""
    sid = request_sid()
    sess = sessions.get(sid)
    if sess:
        sess["transcriber"].stream(data)


@socketio.on("agent_audio_done")
def on_agent_audio_done():
    """Browser tells us playback finished - safe to listen again."""
    sid = request_sid()
    sess = sessions.get(sid)
    if sess:
        sess["agent_speaking"] = False


def request_sid():
    from flask import request
    return request.sid


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)