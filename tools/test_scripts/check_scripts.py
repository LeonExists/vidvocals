"""Static checks for install.ps1 / uninstall.ps1 / *.bat that need only Python.

The thorough checks (real PowerShell parser, AST lint, unit tests, mocked end-to-end
runs) live in Test-InstallScripts.ps1 and need pwsh. This script is the fallback for
machines without PowerShell:

* .ps1: UTF-8 with BOM, CRLF; .bat: ASCII, CRLF
* balanced (), {}, [] outside strings/comments, terminated strings / here-strings
* no PowerShell-7-only operators (?? ??= ?. ?[ && || ternary) in code
* no "$name?" inside double-quoted strings (PowerShell reads the ? as part of the name)
* every script parameter is used
* everything install.ps1 creates is removed by uninstall.ps1 (artifact symmetry)
* the .bat wrappers call their .ps1 with all arguments

Usage: python tools/test_scripts/check_scripts.py   (exit code 0 = OK)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOM = b"\xef\xbb\xbf"
errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)
    print("FAIL:", msg)


def split_code(text: str, name: str) -> tuple[str, list[str]]:
    """Returns (code with strings/comments blanked, list of double-quoted string bodies)."""
    out: list[str] = []
    dq: list[str] = []
    i, n = 0, len(text)
    line = 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
        if text.startswith("<#", i):
            j = text.find("#>", i + 2)
            if j < 0:
                err(f"{name}:{line}: unterminated block comment")
                break
            line += text.count("\n", i, j)
            out.append("\n" * text.count("\n", i, j))
            i = j + 2
            continue
        if c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("@'", i) or text.startswith('@"', i):
            q = text[i + 1]
            m = re.compile(r"\r?\n" + re.escape(q) + "@").search(text, i + 2)
            if not m:
                err(f"{name}:{line}: unterminated here-string")
                break
            body = text[i + 2:m.start()]
            if q == '"':
                dq.append(body)
            line += body.count("\n")
            out.append(" S " + "\n" * body.count("\n"))
            i = m.end()
            continue
        if c in "'‘’‚‛":
            j = i + 1
            while j < n:
                if text[j] in "'‘’‚‛":
                    if j + 1 < n and text[j + 1] in "'‘’‚‛":
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                err(f"{name}:{line}: unterminated single-quoted string")
                break
            line += text.count("\n", i, j)
            out.append(" S " + "\n" * text.count("\n", i, j))
            i = j + 1
            continue
        if c == '"':
            j = i + 1
            depth = 0
            while j < n:
                ch = text[j]
                if ch == "`":
                    j += 2
                    continue
                if text.startswith("$(", j):
                    depth += 1
                    j += 2
                    continue
                if ch == ")" and depth:
                    depth -= 1
                elif ch == '"' and not depth:
                    if j + 1 < n and text[j + 1] == '"':
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                err(f"{name}:{line}: unterminated double-quoted string")
                break
            dq.append(text[i + 1:j])
            line += text.count("\n", i, j)
            out.append(" S " + "\n" * text.count("\n", i, j))
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out), dq


def check_ps1(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.startswith(BOM):
        err(f"{path.name}: no UTF-8 BOM")
    text = raw[len(BOM):].decode("utf-8") if raw.startswith(BOM) else raw.decode("utf-8")
    if re.search(r"(?<!\r)\n", text):
        err(f"{path.name}: bare LF line endings")
    code, dq = split_code(text, path.name)
    pairs = {")": "(", "}": "{", "]": "["}
    stack: list[tuple[str, int]] = []
    for ln, l in enumerate(code.split("\n"), 1):
        for ch in l:
            if ch in "({[":
                stack.append((ch, ln))
            elif ch in ")}]":
                if not stack or stack[-1][0] != pairs[ch]:
                    err(f"{path.name}:{ln}: unbalanced '{ch}'")
                    return text
                stack.pop()
    if stack:
        err(f"{path.name}:{stack[-1][1]}: unclosed '{stack[-1][0]}'")
    for ln, l in enumerate(code.split("\n"), 1):
        for pat, what in ((r"\?\?=?", "??"), (r"\?\.", "?."), (r"\?\[", "?["), (r"&&", "&&"), (r"\|\|", "||"),
                          (r"\s\?\s", "ternary ?:")):
            if re.search(pat, l):
                err(f"{path.name}:{ln}: PowerShell 7-only operator {what}: {l.strip()}")
    for body in dq:
        for m in re.finditer(r"(?<![`{])\$([A-Za-z_][A-Za-z0-9_]*)\?", body):
            err(f"{path.name}: '\"...${m.group(1)}?...\"' - PowerShell reads '{m.group(1)}?' as the variable name; use ${{{m.group(1)}}}?")
    # parameters used
    pm = re.search(r"\nparam\s*\((.*?)\n\)", code, re.S | re.I)
    if pm:
        names = re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", pm.group(1))
        rest = text[:pm.start()] + text[pm.end():]
        for nme in names:
            if not re.search(r"\$" + nme + r"\b", rest):
                err(f"{path.name}: parameter -{nme} is never used")
    else:
        err(f"{path.name}: no script param() block found")
    return text


def check_bat(path: Path, ps1: str) -> None:
    raw = path.read_bytes()
    if any(b > 127 for b in raw):
        err(f"{path.name}: not ASCII")
    text = raw.decode("ascii", "replace")
    if re.search(r"(?<!\r)\n", text):
        err(f"{path.name}: bare LF line endings")
    if f'-NoProfile -ExecutionPolicy Bypass -File "%~dp0{ps1}" %*' not in text:
        err(f"{path.name}: does not call {ps1} with all arguments")
    if 'if "%~1"=="" set "VF_PAUSE=1"' not in text:
        err(f"{path.name}: should pause only when started without arguments")


def check_symmetry(inst: str, uninst: str) -> None:
    # Get-AppPaths table: Key = (& $j 'rel')
    table = dict(re.findall(r"(\w+)\s*=\s*\(& \$j '([^']+)'\)", inst))
    created_keys = set(re.findall(r"Add-Created \$P\.(\w+)", inst))
    for k in ("StartBat", "StartPs1", "Config", "Manifest"):
        created_keys.add(k)
    removed_top = {m.lower() for m in re.findall(r"& \$addInternal '([^']+)'", uninst)}
    removed_top.add("install-manifest.json")
    if "__pycache__" not in uninst:
        err("uninstall.ps1 does not remove __pycache__ folders")
    for k in sorted(created_keys):
        rel = table.get(k)
        if rel is None:
            err(f"install.ps1: Add-Created uses unknown path key {k}")
            continue
        top = rel.split("\\")[0].lower()
        if top not in removed_top:
            err(f"install.ps1 creates '{rel}' (top level '{top}') but uninstall.ps1 does not remove it")
    # Everything else under the folder that install writes must be below a removed top-level folder.
    for k, rel in table.items():
        top = rel.split("\\")[0].lower()
        if k in ("Requirements", "Package"):
            continue
        if top not in removed_top:
            err(f"install.ps1 path {k}='{rel}' is not covered by uninstall.ps1")
    pat_i = re.search(r"FirewallRulePatterns = (@\([^)]*\))", inst)
    pat_u = re.search(r"FirewallRulePatterns = (@\([^)]*\))", uninst)
    if not pat_i or not pat_u or pat_i.group(1) != pat_u.group(1):
        err("firewall rule patterns differ between install.ps1 and uninstall.ps1")
    if "Voice Filter (TCP $Port)" not in inst:
        err("install.ps1: firewall rule display name is not 'Voice Filter (TCP <port>)'")
    an_i = re.search(r"\$script:AppName = '([^']+)'", inst)
    an_u = re.search(r"\$script:AppName = '([^']+)'", uninst)
    if not an_i or not an_u or an_i.group(1) != an_u.group(1):
        err("shortcut name ($script:AppName) differs between install.ps1 and uninstall.ps1")
    for envname in ("Desktop", "Programs"):
        if f"GetFolderPath('{envname}')" not in inst or f"'{envname}'" not in uninst:
            err(f"shortcut folder {envname} not handled symmetrically")


def main() -> int:
    inst = check_ps1(ROOT / "install.ps1")
    uninst = check_ps1(ROOT / "uninstall.ps1")
    check_bat(ROOT / "install.bat", "install.ps1")
    check_bat(ROOT / "uninstall.bat", "uninstall.ps1")
    check_symmetry(inst, uninst)
    if errors:
        print(f"{len(errors)} problem(s)")
        return 1
    print("check_scripts: all static checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
