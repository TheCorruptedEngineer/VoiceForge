"""VoiceForge server — local voice-cloning TTS hub.

Run:  uvicorn app:app --host 127.0.0.1 --port 8484   (from server/, inside envs/server)

Surface:
  UI            GET  /
  Engines       GET  /api/engines            catalog + install state + param schemas
                POST /api/engines/{id}/install     (then poll /install/log?pos=N)
                POST /api/engines/{id}/unload
                GET  /api/engines/{id}/log         live worker stderr
  Voices        GET/POST /api/voices, GET/PATCH/DELETE /api/voices/{slug}
                POST /api/voices/{slug}/refs       multipart clips
  Speech        POST /api/tts     {text, voice, engine?, model?, params?, language?} → wav
                POST /api/postfx  {file, params} → wav  re-shape the cached raw
                                  take (pitch/speed/…) without regenerating
                POST /api/speak   same body → plays on the server's speakers (CLI/agents)
                POST /v1/audio/speech              OpenAI-compatible
  Misc          GET /api/status, GET/PATCH /api/config, GET /api/history
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import voices as V
from engine_manager import EngineError, EngineManager

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
RAW_DIR = OUTPUTS / ".raw"  # pre-FX takes; what /api/postfx re-shapes
RAW_KEEP = 40               # newest raw takes kept (a raw wav is a few MB)
CONFIG_PATH = ROOT / "config.json"

# The shared postfx module runs here too (CPU-only DSP — parselmouth/scipy,
# no torch), so output shaping never needs a worker or the GPU.
import sys as _sys
_sys.path.insert(0, str(ROOT / "engines" / "_shared"))
import postfx as _postfx

DEFAULT_CONFIG = {
    "port": 8484,
    "exclusive_gpu": True,
    "idle_unload_minutes": 15,
    "default_engine": "chatterbox",
    "default_model": None,
    "default_voice": None,
    # written by the server after every successful generation; requests that
    # omit voice/engine/model/params inherit it ("the default is whatever I
    # last used"). Settings-page defaults are the fallback beneath it.
    "last_used": None,
}

_config_lock = threading.Lock()


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return cfg


def save_config(cfg: dict):
    with _config_lock:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict):
    """Never leave a half-written history/config sidecar after interruption."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


CONFIG = load_config()
manager = EngineManager(lambda: CONFIG)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    manager.unload()  # kill workers on shutdown — orphans would hog VRAM


app = FastAPI(title="VoiceForge", version="0.1.0", lifespan=lifespan)


@app.exception_handler(EngineError)
async def engine_error_handler(_: Request, exc: EngineError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(V.VoiceError)
async def voice_error_handler(_: Request, exc: V.VoiceError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------------------------------------------------------- engines --

@app.get("/api/engines")
def api_engines():
    return [spec.public() for spec in manager.specs.values()]


@app.post("/api/engines/{engine_id}/install")
def api_install(engine_id: str):
    manager.install(engine_id)
    return {"ok": True}


@app.get("/api/engines/{engine_id}/install/log")
def api_install_log(engine_id: str, pos: int = 0):
    return manager.install_log(engine_id, pos)


@app.post("/api/engines/{engine_id}/unload")
def api_unload(engine_id: str):
    manager.unload(engine_id)
    return {"ok": True}


@app.get("/api/engines/{engine_id}/log")
def api_worker_log(engine_id: str):
    return {"lines": manager.worker_log(engine_id)}


@app.get("/api/status")
def api_status():
    return {"engines": manager.status(), "config": CONFIG}


# ----------------------------------------------------------------- config --

@app.get("/api/config")
def api_get_config():
    return CONFIG


@app.patch("/api/config")
def api_patch_config(patch: dict = Body(...)):
    for k, v in patch.items():
        if k in DEFAULT_CONFIG:
            CONFIG[k] = v
    # explicitly chosen Settings defaults must take effect now, not stay
    # shadowed behind the last-used snapshot
    if ("default_voice" in patch or "default_engine" in patch) and CONFIG.get("last_used"):
        CONFIG["last_used"] = None
    save_config(CONFIG)
    return CONFIG


# ----------------------------------------------------------------- voices --

@app.get("/api/voices")
def api_voices():
    return V.list_voices()


@app.post("/api/voices")
def api_create_voice(body: dict = Body(...)):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "voice name required")
    return V.create_voice(name, body.get("language", ""), body.get("notes", ""))


@app.get("/api/voices/{slug}")
def api_get_voice(slug: str):
    return V.get_voice(slug)


@app.patch("/api/voices/{slug}")
def api_update_voice(slug: str, patch: dict = Body(...)):
    return V.update_voice(slug, patch)


@app.delete("/api/voices/{slug}")
def api_delete_voice(slug: str):
    V.delete_voice(slug)
    if CONFIG.get("default_voice") == slug:
        CONFIG["default_voice"] = None
        save_config(CONFIG)
    return {"ok": True}


@app.post("/api/voices/{slug}/refs")
async def api_add_refs(slug: str, files: list[UploadFile]):
    meta = None
    for f in files:
        meta = V.add_ref(slug, f.filename or "clip.wav", await f.read())
    if meta is None:
        raise HTTPException(400, "no files uploaded")
    return meta


# ----------------------------------------------------------------- speech --

def _voice_exists(slug: str | None) -> bool:
    if not slug:
        return False
    try:
        V.get_voice(slug)
        return True
    except V.VoiceError:
        return False


def _synthesize(text: str, voice_slug: str | None, engine_id: str | None,
                model_id: str | None, params: dict | None,
                language: str | None) -> Path:
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text is empty")

    last = CONFIG.get("last_used") or {}

    engine_id = (engine_id or last.get("engine")
                 or CONFIG.get("default_engine") or "chatterbox")
    if engine_id not in manager.specs:
        raise HTTPException(400, f"unknown engine {engine_id!r}")
    if not model_id:
        model_id = (last.get("model") if last.get("engine") == engine_id else None) \
            or CONFIG.get("default_model") or None
    if not params and last.get("engine") == engine_id:
        params = last.get("params") or None
    if not language and last.get("engine") == engine_id:
        language = last.get("language") or None

    if not voice_slug:
        for candidate in (last.get("voice"), CONFIG.get("default_voice")):
            if _voice_exists(candidate):
                voice_slug = candidate
                break
    refs, voice_params = [], {}
    if voice_slug:
        voice = V.get_voice(voice_slug)
        refs = V.resolve_refs(voice_slug)
        voice_params = (voice.get("params") or {}).get(engine_id, {})
        language = language or voice.get("language") or None

    merged = {**voice_params, **(params or {})}

    out_name = time.strftime("%Y%m%d-%H%M%S") + f"-{voice_slug or 'default'}-{uuid.uuid4().hex[:6]}.wav"
    out_path = OUTPUTS / out_name
    OUTPUTS.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    resp = manager.tts(engine_id, model_id, {
        "text": text,
        "refs": refs,
        "params": merged,
        "language": language,
        "out": str(out_path),
        "raw_out": str(RAW_DIR / out_name),
    })
    _prune_raw()

    sidecar = {
        "text": text, "voice": voice_slug, "engine": engine_id,
        "model": resp.get("model"), "params": merged, "language": language,
        "seconds": resp.get("seconds"), "gen_time": round(time.time() - t0, 2),
        "actual_seed": resp.get("seed"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json_atomic(out_path.with_suffix(".json"), sidecar)

    CONFIG["last_used"] = {
        "voice": voice_slug, "engine": engine_id, "model": resp.get("model"),
        "params": merged, "language": language,
    }
    save_config(CONFIG)
    return out_path


@app.post("/api/tts/cancel")
def api_tts_cancel():
    return {"cancelled": manager.cancel_current()}


@app.post("/api/tts")
def api_tts(body: dict = Body(...)):
    out = _synthesize(body.get("text", ""), body.get("voice"), body.get("engine"),
                      body.get("model"), body.get("params"), body.get("language"))
    return FileResponse(out, media_type="audio/wav", filename=out.name,
                        headers={"X-Output-File": out.name})


# ------------------------------------------------------------ output shaping --

_postfx_lock = threading.Lock()  # serialize rewrites of the same output file


def _prune_raw():
    if not RAW_DIR.exists():
        return
    raws = sorted(RAW_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in raws[RAW_KEEP:]:
        try:
            p.unlink()
        except OSError:
            pass


@app.post("/api/postfx")
def api_postfx(body: dict = Body(...)):
    """Re-apply output shaping (pitch/formant/intonation/speed/clarity/
    normalize) to a take's cached raw waveform — CPU-only, no worker, no
    regeneration. Rewrites the output wav + sidecar so history and downloads
    match what you hear."""
    name = Path(body.get("file") or "").name  # basename only — no traversal
    if not name.endswith(".wav"):
        raise HTTPException(400, "file must be a .wav output name")
    raw = RAW_DIR / name
    out = OUTPUTS / name
    if not raw.exists():
        raise HTTPException(404, "no raw take cached for this output — regenerate once")

    fx = {k: v for k, v in (body.get("params") or {}).items() if k in _postfx.POSTFX_KEYS}

    import soundfile as sf
    with _postfx_lock:
        y, sr = sf.read(raw, dtype="float32")
        try:
            y = _postfx.apply(y, sr, fx, lambda m: print(f"[postfx] {m}", flush=True))
        except Exception as e:
            raise HTTPException(500, f"postfx failed: {e}")
        sf.write(out, y, sr)

        side = out.with_suffix(".json")
        if side.exists():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
                meta["params"] = {**(meta.get("params") or {}), **fx}
                meta["seconds"] = round(len(y) / sr, 2)
                _write_json_atomic(side, meta)
            except (json.JSONDecodeError, OSError):
                pass

    return Response(content=out.read_bytes(), media_type="audio/wav",
                    headers={"X-Output-File": name})


_play_lock = threading.Lock()


@app.post("/api/speak")
def api_speak(body: dict = Body(...)):
    """Generate and play on this machine's speakers. For say.cmd / agents."""
    out = _synthesize(body.get("text", ""), body.get("voice"), body.get("engine"),
                      body.get("model"), body.get("params"), body.get("language"))
    import winsound

    def play():
        with _play_lock:  # queue overlapping speaks instead of garbling them
            winsound.PlaySound(str(out), winsound.SND_FILENAME)

    if body.get("wait", True):
        play()
    else:
        threading.Thread(target=play, daemon=True).start()
    return {"ok": True, "file": out.name}


@app.post("/v1/audio/speech")
def openai_speech(body: dict = Body(...)):
    """OpenAI-compatible endpoint: {model, input, voice, response_format?}.
    model may be "voiceforge" (defaults), an engine id, or "engine:model_id".
    """
    model = body.get("model") or ""
    engine_id = model_id = None
    if model and model != "voiceforge":
        engine_id, _, model_id = model.partition(":")
        model_id = model_id or None
        if engine_id not in manager.specs:
            engine_id = None
    fmt = body.get("response_format", "wav")
    if fmt != "wav":
        raise HTTPException(400, "only response_format=wav is supported")
    voice = body.get("voice") or None
    if voice in ("alloy", "default"):  # OpenAI client defaults → our default voice
        voice = None
    out = _synthesize(body.get("input", ""), voice, engine_id, model_id,
                      body.get("params"), None)
    return Response(content=out.read_bytes(), media_type="audio/wav")


# ---------------------------------------------------------------- history --

@app.get("/api/history")
def api_history(limit: int = 40, offset: int = 0, q: str = "",
                engine: str = "", voice: str = ""):
    """Durable, searchable history. Disk is the source of truth."""
    items = []
    total = 0
    total_all = 0
    if OUTPUTS.exists():
        wavs = sorted(OUTPUTS.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        total_all = len(wavs)
        query = q.strip().casefold()
        filtered = []
        for p in wavs:
            meta = {}
            side = p.with_suffix(".json")
            if side.exists():
                try:
                    meta = json.loads(side.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError, UnicodeError):
                    pass
            if engine and meta.get("engine") != engine:
                continue
            if voice == "__default__" and meta.get("voice"):
                continue
            if voice and voice != "__default__" and (meta.get("voice") or "") != voice:
                continue
            haystack = " ".join(str(meta.get(k) or "") for k in
                                ("text", "voice", "engine", "model", "created")).casefold()
            if query and query not in haystack:
                continue
            filtered.append({
                "file": p.name,
                "url": f"/outputs/{p.name}",
                "saved": True,
                "modified": p.stat().st_mtime,
                **meta,
            })
        total = len(filtered)
        items = filtered[offset:] if limit <= 0 else filtered[offset:offset + limit]
    return JSONResponse(
        {"total": total, "total_all": total_all, "items": items},
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.delete("/api/history/{filename}")
def api_delete_history(filename: str):
    """Delete one generated take and its metadata/raw cache, basename-only."""
    name = Path(filename).name
    if name != filename or not name.lower().endswith(".wav"):
        raise HTTPException(400, "invalid history filename")
    wav = OUTPUTS / name
    if not wav.exists():
        raise HTTPException(404, "history item not found")
    for path in (wav, wav.with_suffix(".json"), RAW_DIR / name):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise HTTPException(500, f"could not remove {path.name}: {exc}")
    return {"ok": True, "file": name}


# ----------------------------------------------------------------- static --

OUTPUTS.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")
app.mount("/voicefiles", StaticFiles(directory=str(ROOT / "voices")), name="voicefiles")
app.mount("/", StaticFiles(directory=str(ROOT / "ui"), html=True), name="ui")
