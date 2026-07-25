"""VoiceForge worker for Meituan LongCat-AudioDiT."""

from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_upstream"))
sys.path.insert(0, str(HERE.parent / "_shared"))
import postfx


def log(message):
    print(f"[longcat] {message}", file=sys.stderr, flush=True)


def reply(payload):
    print(json.dumps(payload), file=sys.stdout, flush=True)


MODEL = None
TOKENIZER = None
MODEL_ID = None
WHISPER = None
REPOS = {
    "1b": "meituan-longcat/LongCat-AudioDiT-1B",
    "3_5b": "meituan-longcat/LongCat-AudioDiT-3.5B",
}


def load_model(model_id: str):
    global MODEL, TOKENIZER, MODEL_ID
    if model_id not in REPOS:
        raise ValueError(f"unknown LongCat model {model_id!r}")
    import torch
    import audiodit  # noqa: F401 — registers the HF model type
    from audiodit import AudioDiTModel
    from transformers import AutoTokenizer

    unload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = REPOS[model_id]
    log(f"loading {repo} on {device}; first use downloads weights")
    MODEL = AudioDiTModel.from_pretrained(repo).to(device)
    MODEL.vae.to_half()
    MODEL.eval()
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL.config.text_encoder_model)
    MODEL_ID = model_id
    log(f"{model_id} ready")


def unload():
    global MODEL, TOKENIZER, MODEL_ID
    import gc
    MODEL = TOKENIZER = None
    MODEL_ID = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def ref_transcript(path: str, given: str | None) -> str:
    if given:
        return given.strip()
    sidecar = Path(path + ".txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    global WHISPER
    if WHISPER is None:
        from faster_whisper import WhisperModel
        log("loading faster-whisper base on CPU for one-time reference transcript")
        WHISPER = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = WHISPER.transcribe(path)
    text = "".join(segment.text for segment in segments).strip()
    if not text:
        raise RuntimeError("could not transcribe the reference clip")
    sidecar.write_text(text, encoding="utf-8")
    return text


def do_tts(msg: dict) -> dict:
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("no model loaded")
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from utils import approx_duration_from_text, load_audio, normalize_text

    refs = msg.get("refs") or []
    if not refs:
        raise RuntimeError("LongCat cloning needs a reference clip")
    if len(refs) > 1:
        refs = sorted(refs, key=lambda r: -float(r.get("weight", 1.0)))
        log("LongCat uses the highest-weight reference clip")
    ref = refs[0]
    prompt_source = ref["path"]
    prompt_path = postfx.preprocess_ref(prompt_source, log)
    prompt_text = normalize_text(ref_transcript(prompt_source, ref.get("transcript")))
    text = normalize_text(msg["text"])
    params = msg.get("params") or {}

    seed = int(params.get("seed") or random.randint(1, 999999))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    inputs = TOKENIZER([f"{prompt_text} {text}"], padding="longest", return_tensors="pt")
    sr = MODEL.config.sampling_rate
    hop = MODEL.config.latent_hop
    max_duration = MODEL.config.max_wav_duration
    prompt_wav = load_audio(prompt_path, sr).unsqueeze(0)

    off = 3
    encoded = load_audio(prompt_path, sr)
    if encoded.shape[-1] % hop:
        encoded = F.pad(encoded, (0, hop - encoded.shape[-1] % hop))
    encoded = F.pad(encoded, (0, hop * off))
    device = next(MODEL.parameters()).device
    with torch.no_grad():
        prompt_latent = MODEL.vae.encode(encoded.unsqueeze(0).to(device))
    prompt_latent = prompt_latent[..., :-off]
    prompt_frames = prompt_latent.shape[-1]
    prompt_seconds = prompt_frames * hop / sr
    target_seconds = approx_duration_from_text(text, max_duration=max_duration - prompt_seconds)
    ref_estimate = approx_duration_from_text(prompt_text, max_duration=max_duration)
    target_seconds *= float(np.clip(prompt_seconds / max(ref_estimate, 0.1), 1.0, 1.5))
    duration = min(int(target_seconds * sr // hop) + prompt_frames,
                   int(max_duration * sr // hop))

    with torch.inference_mode():
        output = MODEL(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_audio=prompt_wav,
            duration=duration,
            steps=int(params.get("steps", 16)),
            cfg_strength=float(params.get("guidance_strength", 4.0)),
            guidance_method="apg",
        )
    wav = output.waveform.squeeze().detach().float().cpu().numpy()
    if msg.get("raw_out"):
        sf.write(msg["raw_out"], wav, sr)
    shaped = postfx.apply(wav, sr, params, log)
    sf.write(msg["out"], shaped, sr)
    return {"out": msg["out"], "seconds": round(len(shaped) / sr, 2),
            "model": MODEL_ID, "seed": seed}


def main():
    log(f"worker up (python {sys.version.split()[0]})")
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            rid, cmd = msg.get("id"), msg.get("cmd")
            if cmd == "load":
                load_model(msg["model"])
                reply({"id": rid, "ok": True, "loaded": MODEL_ID})
            elif cmd == "tts":
                reply({"id": rid, "ok": True, **do_tts(msg)})
            elif cmd == "ping":
                reply({"id": rid, "ok": True, "loaded": MODEL_ID})
            elif cmd == "unload":
                unload(); reply({"id": rid, "ok": True})
            else:
                reply({"id": rid, "ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:
            log(traceback.format_exc())
            reply({"id": locals().get("rid"), "ok": False,
                   "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
