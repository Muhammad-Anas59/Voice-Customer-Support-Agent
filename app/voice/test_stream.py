"""
Day 1 test: speak into your mic, see live text on screen.
No app integration yet — just proving AssemblyAI streaming works.
"""

import os
import pyaudio
from assemblyai.streaming.v3 import (
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    TurnEvent,
)

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set ASSEMBLYAI_API_KEY in your environment or .env file")

SAMPLE_RATE = 16000
CHUNK = 800  # 50ms of audio at 16kHz

def on_turn(_, event: TurnEvent):
    tag = "FINAL" if event.end_of_turn else "partial"
    print(f"[{tag}] {event.transcript}")

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

    print("Listening... speak into your mic. Press Ctrl+C to stop.")
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
