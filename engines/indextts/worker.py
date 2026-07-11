"""IndexTTS-2 worker — envs/indextts (py 3.11). Same stdio protocol.

Emotional zero-shot cloning. The reference clip is the speaker; emotion comes
from the same clip by default, or from emo_text (a described mood) at strength
emo_alpha. Heavy model — fp16 on an 8 GB card, short passages recommended.

IndexTTS2.infer() writes a wav directly, so we render to a temp file, then load
it and run the shared postfx (pitch/speed/clarity/…) before writing out+raw.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR.parent / "_shared"))
import postfx


def log(msg):
    print(f"[indextts] {msg}", file=sys.stderr, flush=True)


def reply(obj):
    print(json.dumps(obj), file=sys.stdout, flush=True)


MODEL = None
MODEL_ID = None
CKPT = ENGINE_DIR / "checkpoints"


def load_model(model_id: str):
    global MODEL, MODEL_ID
    import torch
    from indextts.infer_v2 import IndexTTS2

    if model_id != "indextts2":
        raise ValueError(f"unknown indextts model {model_id!r}")
    if MODEL is not None:
        unload()
    fp16 = torch.cuda.is_available()  # fp16 on GPU to fit 8 GB; fp32 on CPU
    log(f"loading IndexTTS-2 (fp16={fp16}) from {CKPT} …")
    MODEL = IndexTTS2(
        cfg_path=str(CKPT / "config.yaml"),
        model_dir=str(CKPT),
        use_fp16=fp16,
        use_cuda_kernel=False,
    )
    MODEL_ID = model_id
    log("indextts2 ready")


def unload():
    global MODEL, MODEL_ID
    import gc
    import torch
    MODEL = None
    MODEL_ID = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def do_tts(msg: dict) -> dict:
    import soundfile as sf

    if MODEL is None:
        raise RuntimeError("no model loaded")
    text = msg["text"]
    refs = msg.get("refs") or []
    params = msg.get("params") or {}
    out_path = msg["out"]

    if not refs:
        raise RuntimeError("indextts needs a reference clip — add one to the voice")
    if len(refs) > 1:
        refs = sorted(refs, key=lambda r: -float(r.get("weight", 1.0)))
        log("indextts uses one reference; taking the highest-weight clip")
    ref_path = postfx.preprocess_ref(refs[0]["path"], log)

    emo_text = (params.get("emo_text") or "").strip() or None
    kw = dict(
        spk_audio_prompt=ref_path,
        text=text,
        output_path=str(Path(out_path).with_suffix(".idx.wav")),
        emo_alpha=float(params.get("emo_alpha", 1.0)),
        use_random=bool(int(params.get("use_random") or 0)),
        verbose=False,
    )
    if emo_text:
        kw["use_emo_text"] = True
        kw["emo_text"] = emo_text

    log(f"generating ({len(text)} chars)" + (f", emotion={emo_text!r}" if emo_text else ""))
    MODEL.infer(**kw)

    tmp = Path(kw["output_path"])
    wav, sr = sf.read(str(tmp), dtype="float32")
    try:
        tmp.unlink()
    except OSError:
        pass

    if msg.get("raw_out"):  # pre-FX take, cached so /api/postfx can re-shape live
        sf.write(msg["raw_out"], wav, sr)
    wav = postfx.apply(wav, sr, params, log)
    sf.write(out_path, wav, sr)
    return {"out": out_path, "seconds": round(len(wav) / sr, 2), "model": MODEL_ID}


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
