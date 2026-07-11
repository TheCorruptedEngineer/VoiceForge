"""Kokoro-82M worker — envs/kokoro (py 3.12). Same stdio protocol.

No cloning: `refs` are ignored; the voice comes from the kokoro_voice param
(54 built-ins). Tiny model, loads in seconds, generates faster than real time
— the engine for notifications and say.cmd.

Language codes are kokoro pipeline codes: a=US-English, b=UK-English,
e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=PT-BR, z=Mandarin.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import postfx


def log(msg):
    print(f"[kokoro] {msg}", file=sys.stderr, flush=True)


def reply(obj):
    print(json.dumps(obj), file=sys.stdout, flush=True)


PIPELINES: dict[str, object] = {}  # lang_code -> KPipeline
MODEL_ID = None
SR = 24000

LANG_ALIASES = {"en": "a", "en-us": "a", "en-gb": "b", "es": "e", "fr": "f",
                "hi": "h", "it": "i", "ja": "j", "pt": "p", "zh": "z"}


def load_model(model_id: str):
    global MODEL_ID
    if model_id != "v1_0":
        raise ValueError(f"unknown kokoro model {model_id!r}")
    get_pipeline("a")  # warm the default-language pipeline
    MODEL_ID = model_id
    log("kokoro v1.0 ready")


def get_pipeline(lang: str):
    from kokoro import KPipeline
    lang = LANG_ALIASES.get((lang or "a").lower(), (lang or "a").lower())
    if lang not in "abefhijpz":
        lang = "a"
    if lang not in PIPELINES:
        log(f"creating pipeline for lang_code={lang!r}")
        PIPELINES[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    return PIPELINES[lang]


def unload():
    global PIPELINES, MODEL_ID
    import gc
    import torch
    PIPELINES = {}
    MODEL_ID = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def do_tts(msg: dict) -> dict:
    import numpy as np
    import soundfile as sf

    if MODEL_ID is None:
        raise RuntimeError("no model loaded")
    text = msg["text"]
    params = msg.get("params") or {}
    out_path = msg["out"]
    voice = (params.get("kokoro_voice") or "af_heart").strip()
    speed = float(params.get("native_speed") or 1.0)

    pipe = get_pipeline(msg.get("language") or "a")
    pieces = []
    for _gs, _ps, audio in pipe(text, voice=voice, speed=speed):
        a = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        pieces.append(a.astype(np.float32))
    if not pieces:
        raise RuntimeError("kokoro produced no audio (check voice name)")
    wav = np.concatenate(pieces)

    fx = {k: v for k, v in params.items() if k != "speed"}
    wav = postfx.apply(wav, SR, fx, log)
    sf.write(out_path, wav, SR)
    return {"out": out_path, "seconds": round(len(wav) / SR, 2),
            "model": MODEL_ID, "voice": voice}


def main():
    log(f"worker up (python {sys.version.split()[0]})")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = msg.get("id")
        try:
            cmd = msg.get("cmd")
            if cmd == "load":
                load_model(msg["model"])
                reply({"id": rid, "ok": True, "loaded": MODEL_ID})
            elif cmd == "tts":
                reply({"id": rid, "ok": True, **do_tts(msg)})
            elif cmd == "ping":
                reply({"id": rid, "ok": True, "loaded": MODEL_ID})
            elif cmd == "unload":
                unload()
                reply({"id": rid, "ok": True})
            else:
                reply({"id": rid, "ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as e:
            log(traceback.format_exc())
            reply({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
