"""Voice library — a voice is a folder of reference clips plus voice.json.

voices/<slug>/
    voice.json          { name, language, notes, refs: [{file, weight, transcript}], params: {engine_id: {...}} }
    refs/<clip files>

Multiple refs of the same speaker → engines average them (more stable clone).
Refs of *different* speakers with weights → voice mixing (engine permitting).
Deletion moves the folder to voices/.trash/ — nothing is ever destroyed.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "voices"
TRASH_DIR = VOICES_DIR / ".trash"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


class VoiceError(RuntimeError):
    pass


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "voice"


def _meta_path(slug: str) -> Path:
    return VOICES_DIR / slug / "voice.json"


def _load(slug: str) -> dict:
    p = _meta_path(slug)
    if not p.exists():
        raise VoiceError(f"voice {slug!r} not found")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(slug: str, meta: dict):
    with open(_meta_path(slug), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def list_voices() -> list[dict]:
    out = []
    if not VOICES_DIR.exists():
        return out
    for d in sorted(VOICES_DIR.iterdir()):
        if d.name.startswith(".") or not (d / "voice.json").exists():
            continue
        meta = _load(d.name)
        out.append({"slug": d.name, **meta})
    return out


def get_voice(slug: str) -> dict:
    return {"slug": slug, **_load(slug)}


def create_voice(name: str, language: str = "", notes: str = "") -> dict:
    slug = slugify(name)
    base, n = slug, 2
    while _meta_path(slug).exists():
        slug = f"{base}-{n}"
        n += 1
    (VOICES_DIR / slug / "refs").mkdir(parents=True)
    meta = {
        "name": name.strip() or slug,
        "language": language,
        "notes": notes,
        "refs": [],
        "params": {},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save(slug, meta)
    return {"slug": slug, **meta}


def add_ref(slug: str, filename: str, data: bytes, weight: float = 1.0,
            transcript: str = "") -> dict:
    meta = _load(slug)
    ext = Path(filename).suffix.lower()
    if ext not in AUDIO_EXTS:
        raise VoiceError(f"unsupported audio type {ext!r} (use wav/mp3/flac/ogg/m4a/opus)")
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(filename).name)
    dest = VOICES_DIR / slug / "refs" / safe
    n = 2
    while dest.exists():
        dest = dest.with_name(f"{dest.stem}-{n}{dest.suffix}")
        n += 1
    dest.write_bytes(data)
    meta["refs"].append({"file": dest.name, "weight": weight, "transcript": transcript})
    _save(slug, meta)
    return {"slug": slug, **meta}


def update_voice(slug: str, patch: dict) -> dict:
    meta = _load(slug)
    for key in ("name", "language", "notes", "params"):
        if key in patch:
            meta[key] = patch[key]
    if "refs" in patch:  # weight / transcript edits and removals (by file name)
        keep = {r["file"]: r for r in patch["refs"]}
        refs_dir = VOICES_DIR / slug / "refs"
        for r in meta["refs"]:
            if r["file"] not in keep:
                TRASH_DIR.mkdir(parents=True, exist_ok=True)
                src = refs_dir / r["file"]
                if src.exists():
                    shutil.move(str(src), str(TRASH_DIR / f"{slug}-{int(time.time())}-{r['file']}"))
        meta["refs"] = [
            {"file": f, "weight": float(r.get("weight", 1.0)),
             "transcript": r.get("transcript", "")}
            for f, r in keep.items()
            if (refs_dir / f).exists()
        ]
    _save(slug, meta)
    return {"slug": slug, **meta}


def delete_voice(slug: str):
    if not _meta_path(slug).exists():
        raise VoiceError(f"voice {slug!r} not found")
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(VOICES_DIR / slug), str(TRASH_DIR / f"{slug}-{int(time.time())}"))


def resolve_refs(slug: str) -> list[dict]:
    """Absolute ref paths + normalized weights, ready to hand to a worker."""
    meta = _load(slug)
    refs = []
    for r in meta["refs"]:
        p = VOICES_DIR / slug / "refs" / r["file"]
        if p.exists():
            refs.append({
                "path": str(p),
                "weight": float(r.get("weight", 1.0)),
                "transcript": r.get("transcript", "") or None,
            })
    if not refs:
        raise VoiceError(f"voice {slug!r} has no reference clips — upload at least one")
    total = sum(r["weight"] for r in refs) or 1.0
    for r in refs:
        r["weight"] = r["weight"] / total
    return refs
