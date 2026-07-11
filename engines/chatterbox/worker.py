"""Chatterbox worker — runs inside envs/chatterbox (Python 3.11, torch 2.6 cu124).

Protocol (line-delimited JSON, stdin → stdout, logs on stderr):
  {"id":1,"cmd":"load","model":"multilingual_v3"}
  {"id":2,"cmd":"tts","text":"...","refs":[{"path":"...","weight":0.7}],
   "params":{"exaggeration":0.5},"language":"en","out":"C:/.../x.wav"}
  {"id":3,"cmd":"ping"}

Voice mixing: speaker embeddings of all refs are blended by weight (the
dominant ref donates its prompt tokens/features, which cannot be averaged).
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import postfx


def log(msg):
    print(f"[chatterbox] {msg}", file=sys.stderr, flush=True)


def reply(obj):
    print(json.dumps(obj), file=sys.stdout, flush=True)


MODEL = None
MODEL_ID = None

MODEL_CLASSES = {
    "multilingual_v3": ("chatterbox.mtl_tts", "ChatterboxMultilingualTTS"),
    "turbo": ("chatterbox.tts_turbo", "ChatterboxTurboTTS"),
    "english_v1": ("chatterbox.tts", "ChatterboxTTS"),
}


def load_model(model_id: str):
    global MODEL, MODEL_ID
    import importlib
    import torch

    if model_id not in MODEL_CLASSES:
        raise ValueError(f"unknown chatterbox model {model_id!r}")
    if MODEL is not None:
        del_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {model_id} on {device} … (first run downloads weights)")
    mod_name, cls_name = MODEL_CLASSES[model_id]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    MODEL = cls.from_pretrained(device=device)
    MODEL_ID = model_id
    log(f"{model_id} ready (sr={MODEL.sr})")


def del_model():
    global MODEL, MODEL_ID
    import gc
    import torch
    MODEL = None
    MODEL_ID = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_conds(path: str, exaggeration: float):
    path = postfx.preprocess_ref(path, log)  # trim silence + normalize (cached)
    try:
        MODEL.prepare_conditionals(path, exaggeration=exaggeration)
    except TypeError:  # turbo takes no exaggeration
        MODEL.prepare_conditionals(path)


def build_conditionals(refs: list[dict], exaggeration: float):
    """Single ref → plain conditioning. Multiple → weighted embedding blend."""
    import torch

    if len(refs) == 1:
        prepare_conds(refs[0]["path"], exaggeration)
        return

    conds = []
    for r in refs:
        prepare_conds(r["path"], exaggeration)
        conds.append(MODEL.conds)
    weights = [float(r.get("weight", 1.0)) for r in refs]
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]
    dom = max(range(len(weights)), key=lambda i: weights[i])
    base = conds[dom]

    try:
        with torch.no_grad():
            t3_embs = [c.t3.speaker_emb for c in conds]
            base.t3.speaker_emb = sum(w * e for w, e in zip(weights, t3_embs))
            if isinstance(base.gen, dict) and "embedding" in base.gen:
                gen_embs = [c.gen["embedding"] for c in conds]
                base.gen["embedding"] = sum(w * e for w, e in zip(weights, gen_embs))
        log(f"blended {len(refs)} refs, weights={['%.2f' % w for w in weights]}")
    except Exception as e:
        log(f"embedding blend failed ({e}); falling back to dominant ref")
        base = conds[dom]
    MODEL.conds = base


SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")
MAX_CHUNK = 260


def split_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, cur = [], ""
    for sent in SENTENCE_SPLIT.split(text):
        while len(sent) > MAX_CHUNK:  # pathological run-on: hard split on space
            cut = sent.rfind(" ", 0, MAX_CHUNK)
            cut = cut if cut > 40 else MAX_CHUNK
            piece, sent = sent[:cut], sent[cut:].lstrip()
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(piece)
        if len(cur) + len(sent) + 1 <= MAX_CHUNK:
            cur = f"{cur} {sent}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return chunks or [text]


def filtered_kwargs(fn, **kw):
    sig = inspect.signature(fn)
    return {k: v for k, v in kw.items() if k in sig.parameters and v is not None}


def do_tts(msg: dict) -> dict:
    import torch

    if MODEL is None:
        raise RuntimeError("no model loaded")
    text = msg["text"]
    refs = msg.get("refs") or []
    params = msg.get("params") or {}
    language = msg.get("language") or "en"
    out_path = msg["out"]

    seed = int(params.get("seed") or 0)
    if seed > 0:
        torch.manual_seed(seed)

    exaggeration = float(params.get("exaggeration", 0.5))
    if refs:
        build_conditionals(refs, exaggeration)
    elif getattr(MODEL, "conds", None) is None:
        raise RuntimeError("this model has no built-in voice — add a reference clip")

    gen_kw = filtered_kwargs(
        MODEL.generate,
        exaggeration=exaggeration,
        cfg_weight=float(params["cfg_weight"]) if "cfg_weight" in params else None,
        temperature=float(params["temperature"]) if "temperature" in params else None,
        language_id=language,
    )

    pieces = []
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        log(f"generating chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        wav = MODEL.generate(chunk, **gen_kw)  # (1, N) tensor
        pieces.append(wav.squeeze(0).cpu())

    silence = torch.zeros(int(MODEL.sr * 0.15))
    joined = pieces[0]
    for p in pieces[1:]:
        joined = torch.cat([joined, silence, p])

    y = postfx.apply(joined.numpy(), MODEL.sr, params, log)

    import soundfile as sf
    sf.write(out_path, y, MODEL.sr)
    return {"out": out_path, "seconds": round(len(y) / MODEL.sr, 2),
            "model": MODEL_ID, "chunks": len(chunks)}


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
                del_model()
                reply({"id": rid, "ok": True})
            else:
                reply({"id": rid, "ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as e:
            log(traceback.format_exc())
            reply({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
