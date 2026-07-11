# VoiceForge — local voice cloning & TTS hub

Fully local ElevenLabs replacement: clone a voice from a 10–30s clip, tune the
delivery, mix voices, and speak text — via a web UI, a CLI, or a plain HTTP API
that any app (Mythos gallery, a PC assistant, Claude Code) can call.

Nothing ever leaves this machine. Models download once into `models/`.

## Quick start

1. Double-click **`VoiceForge.bat`** → browser opens at http://127.0.0.1:8484
2. Open **Engines** → Install **Chatterbox** (first install downloads ~3 GB)
3. **+ New** voice → upload (or record) a 10–30s clean clip of the voice
4. Type text → **Generate** (Ctrl+Enter). First generation loads the model (~1 min)

## Engines (July 2026 research)

| Engine | Why | License | VRAM |
|---|---|---|---|
| **Chatterbox Multilingual V3** (default) | The open family that **beat ElevenLabs 63.75%** in blind A/B listening tests. 23 languages, expressiveness + pacing knobs, voice mixing. | MIT | ~4–6 GB |
| **Chatterbox Turbo** | 350M, English, fast — supports `<laugh>` etc. | MIT | ~3 GB |
| **Qwen3-TTS 1.7B/0.6B Base** | 3-second rapid cloning, different timbre flavor. | Apache-2.0 | ~4–5 GB |
| **Qwen3-TTS VoiceDesign** | Describe a voice in words ("deep calm narrator") → get it. | Apache-2.0 | ~5 GB |
| **F5-TTS v1** | Fast flow-matching cloning (near real-time warm at NFE 16), low hallucination on long text, auto-transcribes refs. | MIT | ~3 GB |
| **Kokoro-82M** | Faster than real time, excellent built-in voices — the instant engine for say.cmd/notifications. **No cloning.** | Apache-2.0 | <2 GB |
| **XTTS-v2** | Conditions on **all reference clips at once** — feed it minutes of audio for a steadier clone. 16 languages. | ⚠ CPML weights: **non-commercial** | ~4 GB |

Rejected: Step-Audio-EditX (needs 12–16 GB), Fish/OpenAudio S1 (non-commercial
license), IndexTTS-2 (too slow on 8 GB-class GPUs). GPT-SoVITS (1-minute
fine-tuning, the real "train on longer audio" option) is roadmapped — it isn't
pip-installable and training wants 12 GB VRAM.

Each engine lives in its own venv (`envs/<engine>`) and runs as a subprocess —
conflicting dependencies can't collide, and unloading an engine (or the idle
timeout) kills the process and frees VRAM completely.

## Voices

A voice = `voices/<slug>/` with reference clips + `voice.json`.

- Several clips of the **same person** → embeddings are averaged → steadier clone.
- Clips of **different people** with weight sliders → **voice mixing** (Chatterbox).
- Per-voice default params: tune sliders, then "Save as voice defaults".
- Deleting anything moves it to `voices/.trash/`, never destroys it.

### Output shaping (every engine)

Post-FX runs on the finished audio inside each worker (`engines/_shared/postfx.py`):
**Pitch** (±12 st, TD-PSOLA — natural, formant-preserving), **Formant /
character** (±4 st, vocal-tract size independent of pitch), **Speed**
(0.5–1.6×, pitch-preserving), **Clarity** (0–1 intelligibility EQ: less mud,
more presence/air), **Loudness normalize** (-18 LUFS, on by default).
Reference clips are also auto-cleaned before cloning (silence-trim + peak
normalize, cached as `<clip>.prep.wav`) — steadies clones from rough
recordings.

Chatterbox needs no transcript. Qwen3-TTS needs the clip's transcript and will
auto-transcribe once with faster-whisper (cached in a `<clip>.txt` sidecar).

## API (how other apps integrate)

```
POST /api/tts        {"text": "...", "voice": "my-voice"}          → WAV bytes
POST /api/speak      {"text": "...", "wait": false}                → plays on this PC's speakers
POST /v1/audio/speech {"model":"voiceforge","voice":"my-voice","input":"..."}   (OpenAI-compatible)
GET  /api/voices     list voices
```

Optional `/api/tts` fields: `engine`, `model`, `params` (see `/api/engines` for
each engine's schema), `language`.

## CLI

```
say.cmd "build finished, all tests green"
say.cmd -Voice narrator -NoWait "hello there"
```

## Layout

```
server/     FastAPI hub (thin, no ML deps)
engines/<id>/   engine.json + requirements.txt + worker.py  ← add engines here
ui/         single-file web UI
voices/     your voice library (gitignored)
models/     HF_HOME — all downloaded weights (gitignored)
envs/       per-engine venvs, uv-managed (gitignored)
outputs/    generated audio + metadata sidecars (gitignored)
```

**Adding a new engine** = new folder in `engines/` with those three files.
No server changes needed.

## Notes

- Chatterbox output carries Resemble's inaudible Perth watermark (responsible-AI
  marker baked into the upstream model wrapper).
- Serbian isn't in Chatterbox's 23 languages; closest are Russian/Polish.
  English is the primary use case.
- GPU is serialized: one generation at a time, `exclusive_gpu` keeps a single
  engine resident (both configurable in Settings).
