"""XTTS-v2 worker — envs/xtts (py 3.11, idiap coqui-tts fork). Same protocol.

The distinguishing feature: conditions on EVERY reference clip of the voice
at once (speaker_wav accepts a list) — more/longer audio genuinely improves
the clone, unlike single-ref engines. Weights license is CPML
(non-commercial) — surfaced in the UI blurb.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # CPML prompt would hang stdin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import postfx


def log(msg):
    print(f"[xtts] {msg}", file=sys.stderr, flush=True)


def reply(obj):
    print(json.dumps(obj), file=sys.stdout, flush=True)


MODEL = None
MODEL_ID = None

LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs",
         "ar", "zh-cn", "ja", "hu", "ko", "hi"}


def load_model(model_id: str):
    global MODEL, MODEL_ID
    import torch
    from TTS.api import TTS

    if model_id != "xtts_v2":
        raise ValueError(f"unknown xtts model {model_id!r}")
    if MODEL is not None:
        unload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("loading XTTS-v2 … (first run downloads ~1.9 GB)")
    MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    MODEL_ID = model_id
    log("xtts_v2 ready")


def unload():
    global MODEL, MODEL_ID
    import gc
    import torch
    MODEL = None
    MODEL_ID = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def norm_lang(lang: str | None) -> str:
    lang = (lang or "en").lower()
    if lang in LANGS:
        return lang
    if lang.startswith("zh"):
        return "zh-cn"
    return lang.split("-")[0] if lang.split("-")[0] in LANGS else "en"


def do_tts(msg: dict) -> dict:
    import soundfile as sf

    if MODEL is None:
        raise RuntimeError("no model loaded")
    text = msg["text"]
    refs = msg.get("refs") or []
    params = msg.get("params") or {}
    out_path = msg["out"]

    if not refs:
        raise RuntimeError("xtts needs at least one reference clip")
    # every clip conditions the clone — longer total audio = steadier voice
    ref_paths = [postfx.preprocess_ref(r["path"], log) for r in refs]
    log(f"conditioning on {len(ref_paths)} clip(s)")

    wav = MODEL.tts(
        text=text,
        speaker_wav=ref_paths,
        language=norm_lang(msg.get("language")),
        temperature=float(params.get("temperature") or 0.65),
        repetition_penalty=float(params.get("repetition_penalty") or 2.0),
        speed=float(params.get("native_speed") or 1.0),
    )
    import numpy as np
    wav = np.asarray(wav, dtype=np.float32)
    sr = 24000  # xtts_v2 output rate

    if msg.get("raw_out"):  # pre-FX take, cached so /api/postfx can re-shape live
        sf.write(msg["raw_out"], wav, sr)
    wav = postfx.apply(wav, sr, params, log)
    sf.write(out_path, wav, sr)
    return {"out": out_path, "seconds": round(len(wav) / sr, 2), "model": MODEL_ID,
            "refs_used": len(ref_paths)}


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
