"""F5-TTS worker — envs/f5tts (py 3.11, torch 2.6 cu124). Same stdio protocol.

Cloning: F5 auto-transcribes the reference when ref_text is empty (built-in
ASR pass, cached by us in a sidecar like the qwen worker). Native speed knob
(native_speed) is preferred over postfx time-stretch; postfx handles pitch/
formant/clarity/normalize.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import postfx


def log(msg):
    print(f"[f5tts] {msg}", file=sys.stderr, flush=True)


def reply(obj):
    print(json.dumps(obj), file=sys.stdout, flush=True)


MODEL = None
MODEL_ID = None

MODEL_NAMES = {"v1_base": "F5TTS_v1_Base"}


def load_model(model_id: str):
    global MODEL, MODEL_ID
    import torch
    from f5_tts.api import F5TTS

    if model_id not in MODEL_NAMES:
        raise ValueError(f"unknown f5tts model {model_id!r}")
    if MODEL is not None:
        unload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {MODEL_NAMES[model_id]} on {device} … (first run downloads weights)")
    MODEL = F5TTS(model=MODEL_NAMES[model_id], device=device)
    MODEL_ID = model_id
    log(f"{model_id} ready")


def unload():
    global MODEL, MODEL_ID
    import gc
    import torch
    MODEL = None
    MODEL_ID = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ref_transcript(path: str, given: str | None) -> str:
    """Empty string lets F5 auto-transcribe; we cache its result via sidecar
    so repeat generations skip the ASR pass."""
    if given:
        return given
    sidecar = Path(path + ".txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    return ""  # F5 transcribes internally on empty ref_text


def do_tts(msg: dict) -> dict:
    import soundfile as sf

    if MODEL is None:
        raise RuntimeError("no model loaded")
    text = msg["text"]
    refs = msg.get("refs") or []
    params = msg.get("params") or {}
    out_path = msg["out"]

    if not refs:
        raise RuntimeError("f5tts needs a reference clip — add one to the voice")
    if len(refs) > 1:
        refs = sorted(refs, key=lambda r: -float(r.get("weight", 1.0)))
        log("f5tts uses one reference; taking the highest-weight clip")
    ref = refs[0]
    ref_path = postfx.preprocess_ref(ref["path"], log)

    seed = int(params.get("seed") or 0)
    wav, sr, _spec = MODEL.infer(
        ref_file=ref_path,
        ref_text=ref_transcript(ref_path, ref.get("transcript")),
        gen_text=text,
        nfe_step=int(params.get("nfe_step") or 32),
        cfg_strength=float(params.get("cfg_strength") or 2.0),
        speed=float(params.get("native_speed") or 1.0),
        seed=seed if seed > 0 else None,
        remove_silence=False,
    )

    fx = {k: v for k, v in params.items() if k != "speed"}  # native rate used
    wav = postfx.apply(wav, sr, fx, log)
    sf.write(out_path, wav, sr)
    return {"out": out_path, "seconds": round(len(wav) / sr, 2), "model": MODEL_ID,
            "seed": MODEL.seed}


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
