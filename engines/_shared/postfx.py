"""Post-generation audio FX, shared by all engine workers.

Applied to the finished waveform (numpy float32 mono), so it works identically
for every engine — and re-applied by the server (POST /api/postfx) on the
cached raw take, which is what makes the sliders live: none of this needs the
model or the GPU. Order: PSOLA (pitch+formant+intonation+speed in one pass)
-> clarity EQ -> loudness normalize.

Params (all optional, engine.json exposes them as sliders; the ones below are
POSTFX_KEYS — everything else is generation-side):
  pitch_semitones  float, +/-12. TD-PSOLA pitch shift (Praat via parselmouth),
                   formant-preserving and time-domain, so no vocoder rasp.
                   History: librosa phase vocoder sounded robotic; pyworld
                   WORLD resynthesis sounded scratchy on neural-TTS audio
                   (its aperiodicity model misreads synthesized speech).
                   Fallback chain: PSOLA -> WORLD -> librosa.
  formant_shift    float, +/-4 "semitones" of formant scaling (same Praat
                   call). Changes apparent vocal-tract size (character
                   voices) independently of pitch. 0 = untouched.
  pitch_range      float, 0..2.5 (default 1). Intonation scale: how far the
                   pitch swings around its median. 0 = monotone/robotic,
                   1 = untouched, >1 = extra expressive. Same Praat call.
  speed            float, 0.5..1.6. time-stretch without changing pitch.
                   Done inside the same TD-PSOLA pass (duration factor) —
                   cleaner than the old librosa phase-vocoder stretch, which
                   remains only as the fallback.
  clarity          float 0..1. speech-intelligibility EQ stack, scaled by the
                   slider: rumble high-pass, -mud (300 Hz), +presence (3.2 kHz),
                   +air (9 kHz shelf). 0 = bypass.
  normalize        0/1 (default 1). loudness-match to TARGET_LUFS, with a
                   peak-normalize fallback if pyloudnorm is missing.
"""

from __future__ import annotations

import numpy as np

TARGET_LUFS = -18.0
PEAK_CEIL = 0.94

# The output-shaping params — what /api/postfx accepts and what the UI treats
# as "live" (re-rendered from the cached raw take, no regeneration).
POSTFX_KEYS = ("pitch_semitones", "formant_shift", "pitch_range", "speed",
               "clarity", "normalize")


def apply(wav: np.ndarray, sr: int, params: dict, log=lambda m: None) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).squeeze()

    pitch = float(params.get("pitch_semitones") or 0.0)
    formant = float(params.get("formant_shift") or 0.0)
    rng = params.get("pitch_range")
    rng = 1.0 if rng is None or rng == "" else max(0.0, min(float(rng), 3.0))
    speed = float(params.get("speed") or 1.0)

    want_pitch = abs(pitch) >= 0.05 or abs(formant) >= 0.05 or abs(rng - 1.0) >= 0.02
    want_speed = abs(speed - 1.0) >= 0.02
    if want_pitch or want_speed:
        try:
            wav = _pitch_psola(wav, sr, pitch, formant, rng, speed, log)
            want_speed = False  # PSOLA handled the stretch too
        except Exception as e:
            log(f"postfx: PSOLA failed ({e}); falling back")
            if want_pitch:
                wav = _pitch_world(wav, sr, pitch, formant, rng, log)
    if want_speed:  # only on PSOLA failure
        import librosa
        log(f"postfx: speed x{speed:.2f} (librosa fallback)")
        wav = librosa.effects.time_stretch(wav, rate=speed)

    clarity = float(params.get("clarity") or 0.0)
    if clarity >= 0.02:
        wav = _clarity(wav, sr, min(clarity, 1.0), log)

    if int(params.get("normalize", 1)):
        wav = _normalize(wav, sr, log)

    # hard safety ceiling either way
    peak = float(np.abs(wav).max() or 0.0)
    if peak > 1.0:
        wav = wav / peak * PEAK_CEIL
    return wav


def _pitch_psola(wav: np.ndarray, sr: int, pitch: float, formant: float,
                 rng: float, speed: float, log) -> np.ndarray:
    """Natural speech re-pitching via Praat's TD-PSOLA ("Change gender").

    Time-domain pitch-synchronous overlap-add: shifts pitch by moving glottal
    periods, never re-synthesizing the waveform from a spectral model — which
    is why it avoids both phase-vocoder robotics and WORLD's rasp on TTS
    audio. The same Praat command scales formants, the pitch range around the
    median (intonation) and the duration (pitch-preserving speed) — all four
    shapers in a single overlap-add pass."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)

    # Change gender re-analyzes pitch with the floor/ceiling it is given, and
    # scales THAT median to the requested one — so measure our median with the
    # exact same range, or the shift lands off by the median mismatch.
    floor, ceiling = 60, 500
    new_median = 0.0  # Praat: 0 = keep the current pitch median
    if abs(pitch) >= 0.05:
        p = snd.to_pitch(pitch_floor=floor, pitch_ceiling=ceiling)
        median = call(p, "Get quantile", 0, 0, 0.5, "Hertz")
        if not (median and np.isfinite(median)):
            raise RuntimeError("no voiced frames found for PSOLA")
        new_median = median * (2.0 ** (pitch / 12.0))
        if not 40.0 <= new_median <= 600.0:
            raise RuntimeError(f"target median {new_median:.0f} Hz outside PSOLA-safe range")

    formant_ratio = 2.0 ** (formant / 12.0)
    duration_factor = 1.0 / speed  # speed 1.6x = 0.625x the duration
    shifted = call(snd, "Change gender", floor, ceiling, formant_ratio, new_median,
                   rng, duration_factor)
    y = shifted.values.squeeze().astype(np.float32)
    log(f"postfx: PSOLA pitch {pitch:+.1f} st"
        + (f" (median -> {new_median:.0f} Hz)" if new_median else "")
        + (f", formants x{formant_ratio:.3f}" if abs(formant) >= 0.05 else "")
        + (f", intonation x{rng:.2f}" if abs(rng - 1.0) >= 0.02 else "")
        + (f", speed x{speed:.2f}" if abs(speed - 1.0) >= 0.02 else ""))
    return y


def _pitch_world(wav: np.ndarray, sr: int, pitch: float, formant: float,
                 rng: float, log) -> np.ndarray:
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

        if abs(rng - 1.0) >= 0.02:
            voiced = f0 > 0
            if voiced.any():
                median = float(np.median(f0[voiced]))
                f0 = np.where(voiced, np.maximum(median + (f0 - median) * rng, 40.0), f0)
                log(f"postfx: WORLD intonation x{rng:.2f}")

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


def _rbj_peaking(fc: float, sr: int, gain_db: float, q: float):
    """RBJ audio-EQ-cookbook peaking biquad, returns (b, a)."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w = 2.0 * np.pi * fc / sr
    alpha = np.sin(w) / (2.0 * q)
    cw = np.cos(w)
    b = np.array([1 + alpha * a_lin, -2 * cw, 1 - alpha * a_lin])
    a = np.array([1 + alpha / a_lin, -2 * cw, 1 - alpha / a_lin])
    return b / a[0], a / a[0]


def _rbj_highshelf(fc: float, sr: int, gain_db: float, s: float = 0.9):
    """RBJ high-shelf biquad, returns (b, a)."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w = 2.0 * np.pi * fc / sr
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / 2.0 * np.sqrt((a_lin + 1 / a_lin) * (1 / s - 1) + 2)
    sq = 2.0 * np.sqrt(a_lin) * alpha
    b = np.array([
        a_lin * ((a_lin + 1) + (a_lin - 1) * cw + sq),
        -2 * a_lin * ((a_lin - 1) + (a_lin + 1) * cw),
        a_lin * ((a_lin + 1) + (a_lin - 1) * cw - sq),
    ])
    a = np.array([
        (a_lin + 1) - (a_lin - 1) * cw + sq,
        2 * ((a_lin - 1) - (a_lin + 1) * cw),
        (a_lin + 1) - (a_lin - 1) * cw - sq,
    ])
    return b / a[0], a / a[0]


def _clarity(wav: np.ndarray, sr: int, amount: float, log) -> np.ndarray:
    """Speech-clarity EQ: rumble HPF, de-mud dip, presence bell, air shelf.
    All gains scale linearly with `amount` so the slider stays subtle."""
    from scipy import signal

    sos = signal.butter(2, 70, "highpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, wav.astype(np.float64))

    nyq_guard = sr * 0.45
    stack = [
        _rbj_peaking(300, sr, -2.5 * amount, 1.0),
        _rbj_peaking(min(3200, nyq_guard), sr, 4.5 * amount, 0.9),
    ]
    if 9000 < nyq_guard:
        stack.append(_rbj_highshelf(9000, sr, 3.0 * amount))
    for b, a in stack:
        y = signal.lfilter(b, a, y)
    log(f"postfx: clarity {amount:.2f} (presence {4.5 * amount:+.1f} dB @3.2k)")
    return y.astype(np.float32)


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
