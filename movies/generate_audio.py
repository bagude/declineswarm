#!/usr/bin/env python3
"""
generate_audio.py — Generate the epic-narrator voiceover + sound effects for the
three "visual essay" web movies using the ElevenLabs API.

WHY THIS EXISTS
---------------
The narration script for each movie is *already written*: it is the timed caption
track baked into each HTML file (the `CAPS` array). This script turns those lines
into audio and drops the files exactly where the movies expect them.

USAGE
-----
    export ELEVENLABS_API_KEY=sk_...            # your key
    python3 generate_audio.py                   # generate everything
    python3 generate_audio.py --only narration  # just the voiceover
    python3 generate_audio.py --only sfx        # just the sound effects
    python3 generate_audio.py --voice <id>      # override the narrator voice
    python3 generate_audio.py --dry-run         # print the plan, call nothing

Files already present are skipped, so a re-run after a network hiccup is cheap.
No third-party packages required (uses only the Python standard library).

OUTPUT LAYOUT (relative to this file)
    audio/m1/nar_00.mp3 ... nar_21.mp3     # movie 1 narration, one per caption
    audio/m2/nar_00.mp3 ...                # movie 2
    audio/m3/nar_00.mp3 ...                # movie 3
    audio/sfx/ambient_drone.mp3            # looping underscore bed
    audio/sfx/whoosh.mp3 | blip.mp3 | success.mp3 | error.mp3 | boot.mp3
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(HERE, "audio")
API_ROOT = "https://api.elevenlabs.io/v1"

# The narrator. Default is "Adam" — ElevenLabs' deep, measured American narration
# voice, the closest premade match to a Morgan-Freeman / Neil-deGrasse-Tyson
# documentary read. Swap for another premade voice id with --voice or the
# NARRATOR_VOICE_ID env var. A few good alternates (deep / cinematic):
#   Adam    pNInz6obpgDQGcFmaJgB   deep, calm, authoritative  (default)
#   Brian   nPczCjzI2devNBz1zQrb   resonant American baritone
#   George  JBFqnCBsd6RMkjVDRZzb   warm British storyteller
#   Daniel  onwK4e9ZLuTAKqWW03F9   deep British news-presenter
#   Bill    pqHfZKP75CvOlQylNhV4   older, gravelly American
DEFAULT_VOICE = os.environ.get("NARRATOR_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

# eleven_multilingual_v2 = highest-quality expressive model (best for narration).
TTS_MODEL = os.environ.get("ELEVEN_TTS_MODEL", "eleven_multilingual_v2")

# Tuned for a steady, expressive documentary read: enough stability to stay
# consistent clip-to-clip, enough style for gravitas.
VOICE_SETTINGS = {
    "stability": 0.45,
    "similarity_boost": 0.80,
    "style": 0.35,
    "use_speaker_boost": True,
}

# Spoken-form fixes so the narrator reads code-ish tokens naturally instead of
# spelling out underscores / dots.
REPLACEMENTS = {
    "stop_reason": "stop reason",
    "end_turn": "end turn",
    "get_weather": "get weather",
    "CLAUDE.md": "Claude dot M D",
    "/pick-task": "slash pick task",
    "config.yaml": "config dot yaml",
    "allocation.py": "allocation dot pie",
    "approved.csv": "approved dot C S V",
}


def speakable(text: str) -> str:
    for a, b in REPLACEMENTS.items():
        text = text.replace(a, b)
    return text


# --------------------------------------------------------------------------- #
# Narration scripts — one entry per caption, IN ORDER. These are copied verbatim
# from each movie's CAPS array; index i -> nar_{i:02d}.mp3, played at that
# caption's start time by the movie's audio engine.
# --------------------------------------------------------------------------- #
MOVIES = {
    "m1": [
        "Every AI agent — every one — begins with a single idea.",
        "An LLM call is just a function. Text goes in. Text comes out.",
        "But what if the text coming out could ask for something?",
        "“I’d like to call get_weather.” The model can’t run it. We can.",
        "So we arrange the pieces: a list of messages, the model, a decision, a tool.",
        "And we connect the tool’s output back to the messages. A cycle. A loop.",
        "Watch it run. A question enters the state, and flows to the model. — cycle one",
        "The model answers with a request. stop_reason routes it down to the tool.",
        "The result isn’t shown to anyone. It’s appended — it becomes state.",
        "Around again. — cycle two. Now the model sees its own request and the result.",
        "Nothing left to ask for. stop_reason says end_turn. The loop exits with an answer.",
        "Now a harder question. — cycle three. A city that doesn’t exist.",
        "The tool fails. But look — the error doesn’t crash anything. It’s appended too.",
        "— cycle four. The model reads its own failure... and recovers gracefully.",
        "Errors are not exceptions here. Errors are just edges.",
        "Now, look again at what we built — semantically this time.",
        "The list is STATE: the only truth the system carries. The cycle is EXECUTION.",
        "The decision and the registry are ROUTING: the model’s output picks the edge.",
        "Two lanes are still missing. CONTROL: the power to pause, persist, and resume...",
        "...and MEMORY: the window is finite, so old turns compress, and knowledge moves out.",
        "Five primitives. Watch them all run at once — cycles five, six, seven...",
        "Everything else — LangGraph, Claude Code, the harness — is composition.",
    ],
    "m2": [
        "Last time, we built the loop ourselves — thirty lines, five primitives.",
        "This time, notice something: someone already shipped it.",
        "Type one word — claude — and before you say anything at all...",
        "...the state arrives pre-seeded. CLAUDE.md loads first. Context before conversation.",
        "The shape is identical. But the tools aren’t weather and math anymore.",
        "They’re Read, Edit, Bash — a terminal and a filesystem. That choice is the thesis.",
        "— cycle one. “Fix the failing test.” Claude asks to Read the file. The source becomes state.",
        "— cycle two. Now it wants to Edit. But something new stands in the way.",
        "A hook. Deterministic code that inspects the request before it runs. Not asked. Enforced.",
        "The edit is in scope — the gate opens, the change applies, and it’s appended like anything else.",
        "— cycle three. Run the tests. And... failure.",
        "You know this move: the error doesn’t crash the loop. It becomes an edge.",
        "And quietly, in the margin — the context grew too long, so it compacts itself. Memory, working.",
        "— cycle four. Claude reads its own failure and edits again, sharper this time.",
        "— cycle five. Tests again... twelve passed.",
        "— cycle six. Nothing left to request. end_turn. “Done — tests green.”",
        "Six cycles. Now name what you watched.",
        "STATE is messages plus CLAUDE.md. ROUTING is the built-in registry and slash-command verbs.",
        "CONTROL is hooks and permission tiers. MEMORY is auto-compact. All five lanes — productized.",
        "And here is the move that matters: the whole harness folds into one directory.",
        "Clone it, and every seat in the organization boots this exact loop. Same primitives. Different altitude.",
    ],
    "m3": [
        "Intermission. You’ve built the loop, and watched it shipped.",
        "Now the practical question: how do you shape it — without building anything from scratch?",
        "Think of a franchise. The value was never any single storefront —",
        "— it’s the operating playbook. And this platform exposes seven extension points for yours.",
        "CLAUDE.md — the operating manual. Rules and context, loaded before every session begins.",
        "Slash commands — standard procedures. A full workflow bound to a single typed verb.",
        "Hooks — compliance controls. Enforced in code, whether the model agrees or not.",
        "Skills — domain expertise in folders, loaded exactly when a task calls for it.",
        "Subagents — delegated specialists. Fresh context, one narrow mandate, a typed report back.",
        "MCP servers — integrations. The loop reaches external systems: APIs, databases, applications.",
        "Headless mode — unattended operations. Scheduled runs, CI, scripts: no one at the keyboard.",
        "Seven extension points. One platform. Now — what does real life actually look like?",
        "Day one. An empty folder, a rulebook, and one verb: /pick-task.",
        "One hook — ten lines of bash — becomes law: touch only the files in the task packet.",
        "Then a coworker sits down. Nothing to invent. One command.",
        "The loop runs — and watch: it strays toward the wrong file...",
        "...the hook blocks it. Deterministically. And the model simply corrects course.",
        "Tests pass against the approved reference. “Done” is mechanical — not a feeling.",
        "Total infrastructure: seven files in a folder. Which means... it’s just git.",
        "A teammate clones. Types claude. The whole system boots on their machine, identically.",
        "Rules, verbs, laws, validation — nothing to install, no server, no platform, no permission slip.",
        "From primitive intuition to a prototype anyone can run — today.",
        "Configuration is composition. The playbook is the product.",
    ],
}

# --------------------------------------------------------------------------- #
# Sound effects. `duration` is a hint (seconds) or None for auto. Kept subtle
# and cinematic so they underscore rather than distract.
# --------------------------------------------------------------------------- #
SFX = {
    "ambient_drone": {
        "prompt": "deep cinematic ambient underscore, warm low drone pad, subtle "
                  "evolving space texture, calm documentary bed, no melody, seamless",
        "duration": 22.0,
        "influence": 0.3,
    },
    "whoosh": {
        "prompt": "smooth deep cinematic transition whoosh, soft airy sweep, low end",
        "duration": 2.0,
        "influence": 0.5,
    },
    "blip": {
        "prompt": "soft minimal digital UI blip, gentle high-tech data tick, short and clean",
        "duration": 0.7,
        "influence": 0.6,
    },
    "success": {
        "prompt": "warm positive confirmation chime, soft bell, gentle success tone",
        "duration": 1.4,
        "influence": 0.5,
    },
    "error": {
        "prompt": "low soft muted error tone, gentle negative thud, not harsh, subtle",
        "duration": 0.9,
        "influence": 0.5,
    },
    "boot": {
        "prompt": "futuristic terminal power-on hum, soft rising synth boot, quiet tech startup",
        "duration": 1.6,
        "influence": 0.5,
    },
}


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _post(url: str, payload: dict, api_key: str, retries: int = 4) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    delay = 2.0
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("xi-api-key", api_key)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "audio/mpeg")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            last_err = f"HTTP {e.code}: {body}"
            # 4xx that isn't rate-limiting won't fix itself — fail fast.
            if e.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(last_err)
        except urllib.error.URLError as e:
            last_err = f"network error: {e.reason}"
        if attempt < retries - 1:
            print(f"    retry {attempt + 1}/{retries - 1} in {delay:.0f}s ({last_err})")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(last_err or "unknown error")


def tts(text: str, voice_id: str, api_key: str) -> bytes:
    url = f"{API_ROOT}/text-to-speech/{voice_id}?output_format=mp3_44100_128"
    payload = {
        "text": speakable(text),
        "model_id": TTS_MODEL,
        "voice_settings": VOICE_SETTINGS,
    }
    return _post(url, payload, api_key)


def sound_effect(prompt: str, duration, influence: float, api_key: str) -> bytes:
    url = f"{API_ROOT}/sound-generation"
    payload = {"text": prompt, "prompt_influence": influence}
    if duration is not None:
        payload["duration_seconds"] = duration
    return _post(url, payload, api_key)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["narration", "sfx"], help="generate just one kind")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="narrator voice id")
    ap.add_argument("--force", action="store_true", help="regenerate even if file exists")
    ap.add_argument("--dry-run", action="store_true", help="print plan, make no API calls")
    args = ap.parse_args()

    api_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("XI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: set ELEVENLABS_API_KEY (or XI_API_KEY) in your environment.")
        print("       export ELEVENLABS_API_KEY=sk_...")
        return 1

    do_nar = args.only in (None, "narration")
    do_sfx = args.only in (None, "sfx")

    jobs = []  # (path, kind, callable-or-None)
    if do_nar:
        for mid, lines in MOVIES.items():
            for i, line in enumerate(lines):
                path = os.path.join(AUDIO_DIR, mid, f"nar_{i:02d}.mp3")
                jobs.append((path, f"{mid} nar {i:02d}", ("tts", line)))
    if do_sfx:
        for name, spec in SFX.items():
            path = os.path.join(AUDIO_DIR, "sfx", f"{name}.mp3")
            jobs.append((path, f"sfx {name}", ("sfx", spec)))

    total = len(jobs)
    print(f"Plan: {total} audio files  (voice={args.voice}  model={TTS_MODEL})")
    print(f"Output: {AUDIO_DIR}")
    if args.dry_run:
        for path, label, spec in jobs:
            exists = "  [exists]" if os.path.exists(path) else ""
            print(f"  - {label:16s} -> {os.path.relpath(path, HERE)}{exists}")
        print("\nDry run: nothing generated. Remove --dry-run to build.")
        return 0

    made = skipped = 0
    for n, (path, label, spec) in enumerate(jobs, 1):
        if os.path.exists(path) and not args.force:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kind, data = spec
        print(f"[{n}/{total}] {label} ...", flush=True)
        try:
            if kind == "tts":
                audio = tts(data, args.voice, api_key)
            else:
                audio = sound_effect(data["prompt"], data.get("duration"),
                                     data.get("influence", 0.3), api_key)
        except Exception as e:  # noqa: BLE001 — surface any failure, keep going
            print(f"    FAILED: {e}")
            continue
        with open(path, "wb") as f:
            f.write(audio)
        made += 1
        time.sleep(0.25)  # be gentle on rate limits

    print(f"\nDone. {made} generated, {skipped} already present, "
          f"{total - made - skipped} failed.")
    print("Open any movie in movies/ (via a local web server) to hear it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
