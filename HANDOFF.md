# VoiceForge — session handoff

## State (2026-07-11, initial build)

Built end-to-end in one session: server, UI, Chatterbox + Qwen3-TTS engines,
launcher, CLI, Claude Code skill, gallery integration doc.

### Verified live (2026-07-11, RTX 3060 Ti)
- Chatterbox multilingual_v3: default voice, single-ref clone, **2-ref mix**
  (log: "blended 2 refs"), long-text chunking (467 chars → 23.8s audio),
  warm generation ≈ real-time (5s gen for 6s audio)
- Chatterbox turbo: works incl. `<laugh>` tag (quieter output, rms 0.05)
- OpenAI endpoint, say.cmd, /api/speak (audible), headless UI check PASSED
- Engine install pipeline (qwen3tts installed via POST /api/engines/../install)

### Architecture decisions (why)
- **Worker subprocesses, one venv per engine** — chatterbox pins
  transformers 5.2.0/torch 2.6.0; qwen-tts wants py 3.12 + its own stack.
  Also: killing the process is the only reliable VRAM release on the shared
  8 GB RTX 3060 Ti (gallery + games). Protocol: JSON lines stdin→stdout,
  logs on stderr (`server/engine_manager.py` docstring).
- **Voice = folder of refs + weights** — one mechanism covers multi-clip
  stability (same speaker) AND mixing (different speakers, weighted
  embedding blend in `engines/chatterbox/worker.py:build_conditionals`).
- **HF_HOME = ./models** — fully self-contained project, easy to relocate.
- Port **8484** (gallery backend is 8474).

### Gotchas burned (post-review fixes, all committed)
- **setuptools<81 required** in chatterbox env: resemble-perth imports
  pkg_resources (absent in uv venvs, deleted in setuptools>=82) and perth
  swallows the ImportError → cryptic "'NoneType' object is not callable"
  at model load. Fixed in requirements; a stale worker must be unloaded
  after the fix (it cached perth=None).
- **PS 5.1 advanced-function trap** (say.ps1): one [Parameter()] attribute
  makes ALL params positional-by-declaration-order → quoted text bound to
  -Voice. Fixed with explicit Position=0 on $Text.
- **Idle watchdog vs long jobs**: last_used only updates on response, so a
  >15-min generation would be killed mid-flight → added Worker.busy guard.
- **Orphaned workers**: killing the server (taskkill /F) leaves worker
  processes alive mid-download/load → added FastAPI lifespan shutdown that
  kills workers; don't /F the server if avoidable.
- PyPI torch wheels on Windows are CPU-only — the `+cu124` local version pin
  with `--extra-index-url download.pytorch.org/whl/cu124` is what forces CUDA.
- Chatterbox `prepare_conditionals` takes `exaggeration=` on 500M models but
  not Turbo → try/except TypeError in worker.
- Qwen3-TTS cloning needs ref transcript → auto-transcribed once with
  faster-whisper (CPU), cached as `<clip>.txt` sidecar next to the ref.

### Not yet done / next
- Qwen3-TTS engine written but **install + live test pending** (Install button
  in UI → Engines). VoiceDesign method name (`generate_voice_design`) is
  best-effort — verify against the installed qwen-tts version.
- Mythos Chat Lab wiring (docs/integration.md has the exact snippets).
- Possible engines to add later: F5-TTS (speed), Step-Audio-EditX if GPU ever
  ≥16 GB (best-in-class editing).
- No streaming yet — whole-clip generation only. Chatterbox has no streaming
  API; Qwen3-TTS does (97 ms) if latency ever matters for the assistant.

## Session log
- 2026-07-11: research (Chatterbox V3 family beat ElevenLabs 63.75% blind;
  Step-Audio too big for 8 GB; Fish non-commercial; IndexTTS-2 slow) → full
  build → chatterbox verified locally (default voice, clone, mix, say.cmd,
  OpenAI endpoint). See README for surface.
