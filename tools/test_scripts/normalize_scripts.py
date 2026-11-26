"""Normalize encodings / line endings of the Windows scripts.

* ``*.ps1``  -> UTF-8 **with** BOM, CRLF (Windows PowerShell 5.1 reads BOM-less files
  as ANSI and would garble non-ASCII text).
* ``*.bat``  -> ASCII, CRLF (cmd.exe parses batch files in the OEM code page).

Usage: python tools/test_scripts/normalize_scripts.py [--check] [files...]
Without files, the repo's install/uninstall scripts are processed.  ``--check`` only
reports files that would change (exit code 1 if any).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ["install.ps1", "uninstall.ps1", "install.bat", "uninstall.bat"]
BOM = b"\xef\xbb\xbf"


def normalized(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw.startswith(BOM):
        raw = raw[len(BOM):]
    text = raw.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    text = text.replace("\n", "\r\n")
    if path.suffix.lower() == ".bat":
        text.encode("ascii")  # raises if not ASCII
        return text.encode("ascii")
    return BOM + text.encode("utf-8")


def main(argv: list[str]) -> int:
    check = "--check" in argv
    files = [a for a in argv if not a.startswith("--")] or [str(ROOT / f) for f in DEFAULT]
    changed = 0
    for f in files:
        p = Path(f)
        new = normalized(p)
        if p.read_bytes() != new:
            changed += 1
            if check:
                print(f"needs normalizing: {p}")
            else:
                p.write_bytes(new)
                print(f"normalized: {p}")
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
