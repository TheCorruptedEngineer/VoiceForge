"""Post-install: download the IndexTTS-2 checkpoints into engines/indextts/checkpoints.
Runs inside envs/indextts (so huggingface_hub is available). Idempotent —
snapshot_download skips files already present.
"""

import sys
from pathlib import Path

CKPT = Path(__file__).resolve().parent / "checkpoints"


def main():
    from huggingface_hub import snapshot_download

    CKPT.mkdir(exist_ok=True)
    print(f"downloading IndexTeam/IndexTTS-2 checkpoints -> {CKPT} (multi-GB, one time)")
    # Grab the whole repo except docs/git metadata — the model ships .pt stat
    # files (feat1/feat2/wav2vec2bert_stats) that a narrow allow-list misses.
    snapshot_download(
        repo_id="IndexTeam/IndexTTS-2",
        local_dir=str(CKPT),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    cfg = CKPT / "config.yaml"
    if not cfg.exists():
        print("ERROR: config.yaml missing after download", file=sys.stderr)
        return 1
    print("IndexTTS-2 checkpoints ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
