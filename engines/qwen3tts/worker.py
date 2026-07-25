"""Qwen3-TTS worker — runs inside envs/qwen3tts (Python 3.12, torch 2.7 cu126).

Same stdio JSON protocol as the chatterbox worker (see that file).

Cloning here needs the reference transcript (ref_text). If a ref has none,
we auto-transcribe once with faster-whisper (CPU) and cache the result in a
sidecar file next to the clip (<clip>.txt), so it costs a few seconds only
the first time a clip is used.
"""

from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import postfx


def log(msg):
    print(f"[qwen3tts] {msg}", file=sys.stderr, flush=True)


def reply(obj):
    print(json.dumps(obj), file=sys.stdout, flush=True)


MODEL = None
MODEL_ID = None
WHISPER = None

MODEL_REPOS = {
    "base_1_7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "base_0_6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "customvoice_1_7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "customvoice_0_6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "voicedesign_1_7b": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# server / voices may hand us ISO codes; qwen wants plain names
LANG_BY_CODE = {
    "en": "English", "zh": "Chinese", "de": "German", "it": "Italian",
    "pt": "Portuguese", "es": "Spanish", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "ru": "Russian",
}


def norm_language(lang: str | None) -> str:
    if not lang:
        return "Auto"
    return LANG_BY_CODE.get(lang.lower(), lang.capitalize() if len(lang) > 3 else "Auto")


def load_model(model_id: str):
    global MODEL, MODEL_ID
    import torch
    from qwen_tts import Qwen3TTSModel

    if model_id not in MODEL_REPOS:
        raise ValueError(f"unknown qwen3tts model {model_id!r}")
    if MODEL is not None:
        unload()
    repo = MODEL_REPOS[model_id]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"loading {repo} on {device} … (first run downloads weights)")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        MODEL = Qwen3TTSModel.from_pretrained(
            repo, device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa")
    except (TypeError, ValueError):
        log("SDPA load option unsupported by this qwen-tts build; using compatibility path")
        try:
            MODEL = Qwen3TTSModel.from_pretrained(repo, device_map=device, dtype=torch.bfloat16)
        except TypeError:
            MODEL = Qwen3TTSModel.from_pretrained(repo, device_map=device)
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
    if given:
        return given
    sidecar = Path(path + ".txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    global WHISPER
    if WHISPER is None:
        from faster_whisper import WhisperModel
        log("loading faster-whisper (base, cpu) for one-time ref transcription")
        WHISPER = WhisperModel("base", device="cpu", compute_type="int8")
    log(f"transcribing {Path(path).name} …")
    segments, _info = WHISPER.transcribe(path)
    text = "".join(s.text for s in segments).strip()
    sidecar.write_text(text, encoding="utf-8")
    log(f"transcript cached: {text[:80]!r}")
    return text


def do_tts(msg: dict) -> dict:
    import soundfile as sf
    import torch

    if MODEL is None:
        raise RuntimeError("no model loaded")
    text = msg["text"]
    refs = msg.get("refs") or []
    params = msg.get("params") or {}
    language = norm_language(msg.get("language"))
    out_path = msg["out"]

    seed = int(params.get("seed") or random.randint(1, 999999))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # sampling knobs are forwarded as **kwargs; retry without them if this
    # qwen-tts build doesn't take them
    sample_kw = {}
    if params.get("temperature") not in (None, "", 0):
        sample_kw["temperature"] = float(params["temperature"])
    if params.get("top_p") not in (None, "", 0):
        sample_kw["top_p"] = float(params["top_p"])
    if params.get("top_k") not in (None, "", 0):
        sample_kw["top_k"] = int(params["top_k"])
    if params.get("repetition_penalty") not in (None, "", 0):
        sample_kw["repetition_penalty"] = float(params["repetition_penalty"])

    def gen(fn, **kw):
        try:
            return fn(**kw, **sample_kw)
        except TypeError:
            if not sample_kw:
                raise
            log("this qwen-tts build rejects sampling kwargs; retrying without")
            return fn(**kw)

    if MODEL_ID == "voicedesign_1_7b":
        desc = (params.get("voice_description") or "").strip()
        if not desc:
            raise RuntimeError("VoiceDesign needs a 'voice description' — set it in Advanced params")
        fn = getattr(MODEL, "generate_voice_design", None)
        if fn is None:
            raise RuntimeError("this qwen-tts version has no generate_voice_design(); "
                               "use a Base model or update qwen-tts")
        wavs, sr = gen(fn, text=text, language=language, instruct=desc)
    elif MODEL_ID.startswith("customvoice_"):
        fn = getattr(MODEL, "generate_custom_voice", None)
        if fn is None:
            raise RuntimeError("this qwen-tts version has no generate_custom_voice(); update qwen-tts")
        speaker = (params.get("speaker") or "Ryan").strip()
        custom_kw = {"text": text, "language": language, "speaker": speaker}
        if MODEL_ID == "customvoice_1_7b":
            custom_kw["instruct"] = (params.get("style_instruction") or "").strip()
        wavs, sr = gen(fn, **custom_kw)
    else:
        if not refs:
            raise RuntimeError("qwen3tts cloning needs a reference clip — add one to the voice")
        if len(refs) > 1:
            refs = sorted(refs, key=lambda r: -float(r.get("weight", 1.0)))
            log("qwen3tts does not support mixing; using highest-weight ref only")
        ref = refs[0]
        ref_path = postfx.preprocess_ref(ref["path"], log)  # trim + normalize (cached)
        x_vector_only = bool(params.get("x_vector_only"))
        clone_kw = {
            "text": text,
            "language": language,
            "ref_audio": ref_path,
            "x_vector_only_mode": x_vector_only,
        }
        if not x_vector_only:
            clone_kw["ref_text"] = ref_transcript(ref["path"], ref.get("transcript"))
        wavs, sr = gen(MODEL.generate_voice_clone, **clone_kw)

    if msg.get("raw_out"):  # pre-FX take, cached so /api/postfx can re-shape live
        sf.write(msg["raw_out"], wavs[0], sr)
    wav = postfx.apply(wavs[0], sr, params, log)
    sf.write(out_path, wav, sr)
    return {"out": out_path, "seconds": round(len(wav) / sr, 2),
            "model": MODEL_ID, "seed": seed}


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
