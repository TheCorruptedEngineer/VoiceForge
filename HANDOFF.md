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
- **Pitch-shift algorithm saga** (both user-reported): librosa phase vocoder
  = robotic (phase smearing). pyworld WORLD = scratchy on neural-TTS audio
  (D4C aperiodicity misreads synthesized speech) and F0 lands ~0.7 st off.
  Final: **TD-PSOLA via praat-parselmouth** ("Change gender" call) — time-
  domain, no resynthesis, F0 dead-on (+4 st → 265.7 vs 265.1 Hz target),
  handles formant scaling in the same call. Chain: PSOLA → WORLD → librosa.
- postfx also has `clarity` 0..1 (rumble HPF + RBJ EQ stack: −mud 300 Hz,
  +presence 3.2 kHz, +air 9 kHz shelf; gains scale with slider). Measured
  at 0.6: presence band +1.9 dB, mud −1.2 dB.
- engine.json is read once at server start (EngineSpec cache) — param/schema
  edits need a server restart, worker code edits only need engine unload.
- PyPI torch wheels on Windows are CPU-only — the `+cu124` local version pin
  with `--extra-index-url download.pytorch.org/whl/cu124` is what forces CUDA.
- Chatterbox `prepare_conditionals` takes `exaggeration=` on 500M models but
  not Turbo → try/except TypeError in worker.
- Qwen3-TTS cloning needs ref transcript → auto-transcribed once with
  faster-whisper (CPU), cached as `<clip>.txt` sidecar next to the ref.

### Not yet done / next
- qwen-tts API verified by inspection: `generate_voice_clone(text, language,
  ref_audio, ref_text)` and `generate_voice_design(text, instruct, language)`
  match the worker exactly. Untapped extras: `x_vector_only_mode=True`
  (clone WITHOUT transcript), `generate_custom_voice(text, speaker)` preset
  voices, and sampling kwargs (temperature/top_p) via **kwargs.
- VoiceDesign model itself untested (separate ~4.5 GB checkpoint, downloads
  on first select of that model in the UI).
- Mythos Chat Lab wiring (docs/integration.md has the exact snippets).
- Possible engines to add later: F5-TTS (speed), Step-Audio-EditX if GPU ever
  ≥16 GB (best-in-class editing).
- **Voice-quality roadmap** (user asked for "make voices better"; shipped so
  far: ref preprocessing + postfx pitch/speed/LUFS in engines/_shared/postfx.py):
  1. **resemble-enhance** post-pass (denoiser+enhancer, MIT, ~2 GB VRAM) as an
     optional "Enhance" toggle — biggest bang for perceived quality.
  2. **RVC identity pass**: convert Chatterbox output through an RVC model
     trained on ~10 min of the target speaker — pushes similarity well above
     zero-shot. Heavier setup (per-voice training), design as third engine
     type "fx" so it can chain after any TTS engine.
  3. **Seed hunting**: generate N takes at different seeds, auto-rank by
     speaker-embedding cosine to the refs (resemblyzer), return best. Cheap,
     pure orchestration in the server.
  4. Better multilingual base: watch for Chatterbox V4 / Qwen3-TTS updates.
- No streaming yet — whole-clip generation only. Chatterbox has no streaming
  API; Qwen3-TTS does (97 ms) if latency ever matters for the assistant.

## Session log
- 2026-07-11: research (Chatterbox V3 family beat ElevenLabs 63.75% blind;
  Step-Audio too big for 8 GB; Fish non-commercial; IndexTTS-2 slow) → full
  build → chatterbox verified locally (default voice, clone, mix, say.cmd,
  OpenAI endpoint). See README for surface.
