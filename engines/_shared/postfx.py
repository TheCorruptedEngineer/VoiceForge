"""Post-generation audio FX, shared by all engine workers.

Applied to the finished waveform (numpy float32 mono), so it works identically
for every engine. Order: pitch -> speed -> loudness normalize.

Params (all optional, engine.json exposes them as sliders):
  pitch_semitones  float, +/-12. WORLD-vocoder F0 shift, formant-preserving —
                   the phase-vocoder (librosa) version sounded robotic on
                   speech; WORLD moves only the pitch track and keeps the
                   speaker's timbre. librosa remains as a fallback.
  formant_shift    float, +/-4 "semitones" of spectral-envelope warp.
                   Changes the apparent vocal-tract size (character voices)
                   independently of pitch. 0 = untouched.
  speed            float, 0.5..1.6. time-stretch without changing pitch.
  normalize        0/1 (default 1). loudness-match to TARGET_LUFS, with a
                   peak-normalize fallback if pyloudnorm is missing.
"""

from __future__ import annotations

import numpy as np

TARGET_LUFS = -18.0
PEAK_CEIL = 0.94


def apply(wav: np.ndarray, sr: int, params: dict, log=lambda m: None) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).squeeze()

    pitch = float(params.get("pitch_semitones") or 0.0)
    formant = float(params.get("formant_shift") or 0.0)
    if abs(pitch) >= 0.05 or abs(formant) >= 0.05:
        wav = _pitch_world(wav, sr, pitch, formant, log)

    speed = float(params.get("speed") or 1.0)
    if abs(speed - 1.0) >= 0.02:
        import librosa
        log(f"postfx: speed x{speed:.2f}")
        wav = librosa.effects.time_stretch(wav, rate=speed)

    if int(params.get("normalize", 1)):
        wav = _normalize(wav, sr, log)

    # hard safety ceiling either way
    peak = float(np.abs(wav).max() or 0.0)
    if peak > 1.0:
        wav = wav / peak * PEAK_CEIL
    return wav


def _pitch_world(wav: np.ndarray, sr: int, pitch: float, formant: float, log) -> np.ndarray:
    """Formant-preserving pitch shift via the WORLD vocoder.

    WORLD splits speech into F0 (pitch track), spectral envelope (timbre /
    formants) and aperiodicity (breath). Scaling F0 alone re-pitches the voice
    without the phase-vocoder robotics; warping the envelope's frequency axis
    shifts formants (apparent head size) independently of pitch."""
    try:
        import pyworld as pw

        x = wav.astype(np.float64)
        f0, t = pw.harvest(x, sr)
        sp = pw.cheaptrick(x, f0, t, sr)
        ap = pw.d4c(x, f0, t, sr)

        if abs(pitch) >= 0.05:
            f0 = f0 * (2.0 ** (pitch / 12.0))
            log(f"postfx: WORLD pitch {pitch:+.1f} st")

        if abs(formant) >= 0.05:
            ratio = 2.0 ** (formant / 12.0)
            bins = sp.shape[1]
            src_idx = np.clip(np.arange(bins) / ratio, 0, bins - 1)
            lo = np.floor(src_idx).astype(int)
            hi = np.minimum(lo + 1, bins - 1)
            frac = src_idx - lo
            sp = sp[:, lo] * (1 - frac) + sp[:, hi] * frac
            ap = ap[:, lo] * (1 - frac) + ap[:, hi] * frac
            sp = np.ascontiguousarray(sp)
            ap = np.ascontiguousarray(ap)
            log(f"postfx: formant shift {formant:+.1f} st (x{ratio:.3f})")

        y = pw.synthesize(f0, sp, ap, sr)
        return y.astype(np.float32)
    except Exception as e:
        log(f"postfx: WORLD unavailable ({e}); falling back to librosa (lower quality)")
        import librosa
        if abs(pitch) >= 0.05:
            wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=pitch)
        return wav


def _normalize(wav: np.ndarray, sr: int, log) -> np.ndarray:
    try:
        import pyloudnorm
        meter = pyloudnorm.Meter(sr)
        loudness = meter.integrated_loudness(wav.astype(np.float64))
        if np.isfinite(loudness):
            gain_db = TARGET_LUFS - loudness
            gain_db = max(min(gain_db, 20.0), -20.0)  # don't explode near-silence
            wav = wav * (10.0 ** (gain_db / 20.0))
            log(f"postfx: loudness {loudness:.1f} LUFS -> {TARGET_LUFS} ({gain_db:+.1f} dB)")
    except Exception as e:  # pyloudnorm missing or too-short clip
        log(f"postfx: LUFS normalize unavailable ({e}); peak normalize")
        peak = float(np.abs(wav).max() or 0.0)
        if peak > 0:
            wav = wav / peak * 0.7
    peak = float(np.abs(wav).max() or 0.0)
    if peak > PEAK_CEIL:
        wav = wav / peak * PEAK_CEIL
    return wav


def preprocess_ref(path: str, log=lambda m: None) -> str:
    """Clean a reference clip for cloning: mono, silence-trimmed, peak-normalized.
    Cached as <path>.prep.wav next to the original; returns the path to use.
    Falls back to the original on any failure or if trimming leaves <1s."""
    from pathlib import Path

    src = Path(path)
    prep = Path(str(src) + ".prep.wav")
    try:
        if prep.exists() and prep.stat().st_mtime >= src.stat().st_mtime:
            return str(prep)
        import librosa
        import soundfile as sf
        y, sr = librosa.load(str(src), sr=None, mono=True)
        yt, _ = librosa.effects.trim(y, top_db=35)
        if len(yt) < sr:  # over-trimmed — clip is quiet or short; keep original
            return str(src)
        peak = float(np.abs(yt).max() or 0.0)
        if peak > 0:
            yt = yt / peak * 0.9
        sf.write(str(prep), yt, sr)
        cut = (len(y) - len(yt)) / sr
        log(f"ref prep: {src.name} trimmed {cut:.1f}s silence, normalized")
        return str(prep)
    except Exception as e:
        log(f"ref prep failed for {src.name} ({e}); using raw clip")
        return str(src)
