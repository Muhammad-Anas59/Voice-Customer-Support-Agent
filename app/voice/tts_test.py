"""
Day 2, step 1: standalone ElevenLabs TTS test.

Isolated on purpose (same pattern as Day 1's test_stream.py / voice_test.py):
just text -> spoken audio through speakers, no mic input and no RAG pipeline
involved yet. Once this works reliably, the next step wires the ElevenLabs
call into voice_test.py in place of the current print(f"[AGENT ANSWER] ...").

Requirements:
    pip install elevenlabs python-dotenv
Playback also needs mpv or ffmpeg installed and on PATH (the SDK's play()
helper shells out to one of them).

Run:
    python tts_test.py
"""

import os
import sys
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

load_dotenv()

API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    print("ERROR: ELEVENLABS_API_KEY not found in .env")
    sys.exit(1)

# "George" - a good general-purpose default voice. Swap for whatever
# voice_id you pick from the ElevenLabs voice library later.
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

# eleven_flash_v2_5 is the low-latency model - the right choice for a
# real-time conversational agent (Day 3-4 loop) vs. eleven_multilingual_v2
# or eleven_v3, which sound slightly better but are slower to generate.
# Worth A/B-ing both once you're listening to real output.
MODEL_ID = "eleven_flash_v2_5"

# Sample text standing in for a real [AGENT ANSWER] from get_response().
SAMPLE_ANSWER = (
    "Thanks for reaching out to Verve Athletics support. Orders typically "
    "ship within two to three business days, and you'll receive a tracking "
    "link by email as soon as it's on its way. Is there anything else I can "
    "help you with?"
)


def speak(text: str) -> None:
    client = ElevenLabs(api_key=API_KEY)

    print(f"[TTS] Sending to ElevenLabs ({MODEL_ID}, voice={VOICE_ID})...")
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format="mp3_44100_128",
    )

    print("[TTS] Playing audio...")
    play(audio)
    print("[TTS] Done.")


if __name__ == "__main__":
    print(f"[AGENT ANSWER] {SAMPLE_ANSWER}\n")
    try:
        speak(SAMPLE_ANSWER)
    except Exception as e:
        print(f"[TTS] Failed: {e}")
        print(
            "If this is a playback error, confirm mpv or ffmpeg is "
            "installed and on PATH. If this is an auth/permission error, "
            "confirm the restricted ELEVENLABS_API_KEY has 'Text to "
            "Speech = Access' enabled."
        )
        sys.exit(1)