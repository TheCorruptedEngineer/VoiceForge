"""Fetch the official LongCat-AudioDiT inference package after venv install."""

from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "_upstream"
ARCHIVE = HERE / ".longcat-main.zip"
URL = "https://github.com/meituan-longcat/LongCat-AudioDiT/archive/refs/heads/main.zip"


def main() -> int:
    if (UPSTREAM / "audiodit" / "__init__.py").exists():
        print("LongCat upstream inference package already present")
        return 0
    print("downloading official LongCat-AudioDiT inference code")
    urllib.request.urlretrieve(URL, ARCHIVE)
    unpack = HERE / ".longcat-unpack"
    shutil.rmtree(unpack, ignore_errors=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(unpack)
    roots = [p for p in unpack.iterdir() if p.is_dir()]
    if len(roots) != 1 or not (roots[0] / "audiodit").exists():
        print("ERROR: downloaded archive does not contain audiodit", file=sys.stderr)
        return 1
    shutil.move(str(roots[0]), str(UPSTREAM))
    shutil.rmtree(unpack, ignore_errors=True)
    ARCHIVE.unlink(missing_ok=True)
    print("LongCat upstream ready; model weights download on first use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
