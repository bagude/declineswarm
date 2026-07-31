# Web Movies — epic-narrator audio

Three animated "visual essays" (HTML canvas, 2:30 each), now wired for a
documentary-style voiceover + cinematic sound effects.

| File | Title |
|------|-------|
| `agent-loop.html` | The Agent Loop — how thirty lines become a system |
| `claude-code-loop.html` | The Same Loop, Shipped — Claude Code as the harness |
| `extending-platform.html` | Extending the Platform — a prototype anyone can run |

## The audio is generated on your machine

The narration script is *already written* — it's the timed caption track baked
into each movie. `generate_audio.py` turns those lines into audio via the
ElevenLabs API and drops the files exactly where the movies look for them.

> **Why not pre-generated here?** This build environment's egress policy blocks
> `api.elevenlabs.io`, so the API can't be reached from CI. Run the one command
> below locally (your key, your machine) and everything fills in.

### 1. Generate the audio

```bash
cd movies
export ELEVENLABS_API_KEY=sk_...          # your key
python3 generate_audio.py                 # ~66 files: narration + SFX
```

No dependencies (standard library only). Files already present are skipped, so
re-running after a hiccup is cheap. Useful flags:

```bash
python3 generate_audio.py --dry-run       # show the plan, call nothing
python3 generate_audio.py --only narration
python3 generate_audio.py --only sfx
python3 generate_audio.py --voice <id>    # pick a different narrator
python3 generate_audio.py --force         # regenerate everything
```

**The narrator.** Default is ElevenLabs' *Adam* — deep, measured, American, the
closest premade match to a Morgan-Freeman / Neil-deGrasse-Tyson documentary
read. Alternates (deep / cinematic) are listed at the top of `generate_audio.py`;
pass one with `--voice` or set `NARRATOR_VOICE_ID`.

### 2. Watch

Open a movie **through a local web server** (browsers block audio on `file://`
in some cases, and relative paths resolve cleanly over http):

```bash
cd movies
python3 -m http.server 8000
# then open http://localhost:8000/agent-loop.html
```

Press **▶ Play**. Narration fires at each caption; SFX and a low ambient bed
underscore the animation. The **🔊 button** (added to the controls) mutes/unmutes.

## How the sync works

Each movie runs on a master clock `t` (seconds). An added **audio layer** watches
that clock every frame:

- **Narration** — one clip per caption (`audio/mN/nar_XX.mp3`), triggered when
  `t` crosses that caption's start time. Indices line up 1:1 with the on-screen
  captions.
- **Sound effects** — `audio/sfx/*.mp3` fired at animation events (scene
  transitions → `whoosh`, tool calls → `blip`, failures → `error`, passes →
  `success`, boots → `boot`).
- **Ambient** — `audio/sfx/ambient_drone.mp3` loops quietly under everything.

Seeking and restarting reset the audio cleanly (no bursts). If the audio files
aren't present yet, the movies still play — just silently. Nothing about the
original animation was changed; the audio layer is purely additive.

## Layout after generating

```
movies/
├── agent-loop.html            # audio-wired
├── claude-code-loop.html
├── extending-platform.html
├── generate_audio.py          # run this with your key
└── audio/                     # created by the script
    ├── m1/ nar_00.mp3 … nar_21.mp3
    ├── m2/ nar_00.mp3 … nar_20.mp3
    ├── m3/ nar_00.mp3 … nar_22.mp3
    └── sfx/ ambient_drone.mp3 whoosh.mp3 blip.mp3 success.mp3 error.mp3 boot.mp3
```
