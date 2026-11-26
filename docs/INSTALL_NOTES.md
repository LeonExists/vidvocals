# Install / uninstall scripts: notes, verification, test checklist

Files: `install.bat` → `install.ps1`, `uninstall.bat` → `uninstall.ps1`. The installer
generates `start.bat`, `start.ps1` and `config.json`. Tests live in `tools/test_scripts/`.

## What the installer does

| Step | What | Where |
|---|---|---|
| 1 | Checks Windows build, 64-bit, free disk space (warns below 15 GB), long-path support, OneDrive folders, `nvidia-smi` (driver version and highest CUDA version), a Voice Filter that is already running, the port | – |
| 2 | Downloads `uv` (latest release zip) unless `runtime\uv\uv.exe` runs | `runtime\uv` |
| 3 | `uv venv --python 3.12 --managed-python --no-project .venv`. A working `.venv` is reused. A broken one, or one built on a Python outside the folder, is recreated | `runtime\python`, `.venv` |
| 4 | PyTorch: index chosen from the driver's CUDA version (see below), `torch torchaudio` from `https://download.pytorch.org/whl/<tag>`, next lower tag on failure. Then `requirements.txt` with `-c runtime\torch-constraints.txt` (pins the installed torch/torchaudio), then a check that `torch.cuda.is_available()` | `.venv` |
| 5 | ffmpeg: `ffmpeg.exe` + `ffprobe.exe` on PATH (recorded as `pre-existing`, never touched), else the existing portable copy, else a download (gyan.dev essentials, fallback BtbN) | `runtime\ffmpeg\bin` |
| 6 | `config.json` (keys `port`, `device`, `ffmpeg`, `ffprobe`, `ffmpeg_source`, `torch_index`; other keys the user added are kept), `start.bat`, `start.ps1` | app folder |
| 7 | `python -m voicefilter.download --presets <list>` with all cache variables pointing into the folder (unless `-SkipModels`) | `models\` |
| 8 | Firewall rule `Voice Filter (TCP <port>)`: inbound, TCP, Allow, **Profile Private**. Old `Voice Filter (TCP *)` / `VoiceFilter (TCP *)` rules are removed first. If the active network is Public, the user is asked (default No) whether to switch it to Private. Both changes run in **one** elevated PowerShell (one UAC prompt). If the rule is already correct and nothing needs switching, no UAC prompt appears | Windows Firewall (outside) |
| 9 | `Voice Filter.lnk` on the Desktop and in Start Menu > Programs, target `start.bat`, working folder = app folder (WScript.Shell). An icon is used if `voicefilter\assets\voicefilter.ico` or `voicefilter\web\favicon.ico` exists | outside |
| 10 | `python -m voicefilter.selfcheck` (JSON report is printed and logged), summary with phone URL(s), `install-manifest.json` (merged with an existing one) | app folder |

Everything is logged to `data\logs\install-<timestamp>.log`. Steps whose output is shown
"live" (the PyTorch download and the model download, so that progress bars work) only log
the command and its exit code.

Exit codes: `0` OK; `1` a required step failed (uv, Python, PyTorch, requirements, ffmpeg,
config); `2` installed, but an optional step failed (models, firewall, shortcuts, self check).
Re-running `install.bat` reuses everything that is finished.

Never touched: the user PATH, the PowerShell profile, system Python, system ffmpeg, the
registry (except what Windows itself writes for the firewall rule and the network profile).
uv is isolated with `UV_PYTHON_INSTALL_DIR`, `UV_PYTHON_BIN_DIR`, `UV_CACHE_DIR`,
`UV_TOOL_DIR`, `UV_TOOL_BIN_DIR`, `UV_PYTHON_INSTALL_REGISTRY=0` (no PEP 514 registry
entries), `UV_MANAGED_PYTHON=1`, `UV_NO_CONFIG=1`, `UV_LINK_MODE=copy` and
`UV_NO_MODIFY_PATH=1`, set for the installer process only. User-level variables that would
redirect uv or pip (`UV_INDEX_URL`, `PIP_INDEX_URL`, `UV_PYTHON`, ...) are cleared for the process.

### PyTorch index choice

`nvidia-smi` reports the highest CUDA version the driver supports. The installer tries, from
newest to oldest, every tag the driver can run (at most 4): `cu132` (13.2), `cu130`, `cu129`,
`cu128`, `cu126`, `cu124`. No NVIDIA driver, or a driver below CUDA 12.4 → `cpu`, with a
warning. `-Cpu` forces `cpu`. `-TorchIndex cu126` forces one tag (for troubleshooting).
The brief named cu128 → cu126 → cu124. The newer tags were added because the dev
environment already uses torch 2.15 + cu132, and recent torch releases may only be published
for the newest CUDA lines. On an RTX 3080 with a current driver the first tag tried is cu130 or
cu132. To go back to the brief's behaviour, delete the first three entries of `$script:CudaTags`.

Whether a build is already installed is decided from `torch.__version__` /
`torchaudio.__version__` (these always end in `+cu128`, `+cpu` etc.). The wheel metadata
version is not used for this because it has no tag on some wheels. A torch from another
index is replaced with `--reinstall-package`. The constraints file uses the metadata
versions, because those are what uv resolves against. After `requirements.txt` the
installer checks that torch was not replaced.

### Generated environment (start.bat / start.ps1)

`VF_APP_DIR`, `VF_DATA_DIR`, `VF_MODELS_DIR`, `HF_HOME=models\hf`, `TORCH_HOME=models\torch`,
`XDG_CACHE_HOME=models\cache`, `NUMBA_CACHE_DIR=data\cache\numba`, `MPLCONFIGDIR=data\cache\mpl`,
`CUDA_CACHE_PATH=data\cache\cuda`, `TORCH_EXTENSIONS_DIR=data\cache\torch_extensions`,
`TEMP`/`TMP=data\tmp`, `VF_PORT`, `VF_DEVICE`, `VF_FFMPEG`, `VF_FFPROBE`, `PYTHONUTF8=1`,
`PYTHONIOENCODING=utf-8`, `PYTHONNOUSERSITE=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
`HF_HUB_DISABLE_SYMLINKS_WARNING=1`, `TOKENIZERS_PARALLELISM=false`, and clears `PYTHONPATH`,
`PYTHONHOME`, `PYTHONSTARTUP`, `VIRTUAL_ENV`, `CONDA_PREFIX`. With a portable ffmpeg,
`runtime\ffmpeg\bin` is put first on PATH (for that window only).

`start.bat` is pure ASCII. Every path inside the folder is built from `%~dp0`, so it works
with spaces, umlauts, `&`, `(`, `)` and `%` in the folder name, and it uses no
`( ... )` blocks. Paths outside the folder are written literally only if they are ASCII. A
non-ASCII path (for example a pre-existing ffmpeg under `C:\Users\Jörg\scoop\...`) is left
out of `start.bat`; the backend reads it from `config.json`, and `start.ps1` sets it. The
browser is opened by a hidden PowerShell that polls `http://127.0.0.1:<port>/api/info` (up
to about 2 minutes) and then opens `http://localhost:<port>/`. Use `start.bat --no-browser`
to skip it.

`HF_TOKEN` hook: a commented line in `Get-AppEnvironment` in `install.ps1` (not needed today).

### Generated files to git-ignore

`start.bat`, `start.ps1`, `config.json`, `install-manifest.json`, `.venv/`, `runtime/`,
`models/`, `data/` (`.write-test-*` exists only for a moment during the install).

## What the uninstaller does

1. Stops Voice Filter: processes whose executable is inside the app folder, `cmd.exe`
   windows whose command line runs this folder's `start.bat`, and a process listening on the
   configured port only if its executable is inside the folder. Other processes are reported,
   never killed.
2. Before deleting `models\`, collects the Hugging Face repo folder names (`models--org--name`)
   and torch hub checkpoint names, from the manifest and from `models\`. These are used to
   find stray copies outside the folder.
3. Removes `.venv`, `runtime`, `models` (unless `-KeepModels`), `data`, `start.bat`,
   `start.ps1`, `config.json`, `voicefilter.env`, `__pycache__` folders (not inside `.git`),
   then `install-manifest.json`. Each removal is retried, with a `rd /s /q \\?\...` fallback
   for paths longer than 260 characters.
4. Shortcuts: `Voice Filter.lnk` from the manifest and from Desktop / Start Menu (user and
   all-users). A shortcut is removed only if its target is inside this folder; one that points
   to another copy of Voice Filter is left alone.
5. Firewall rules matching `Voice Filter (TCP *)` / `VoiceFilter (TCP *)`. If the manifest
   says the installer switched a network to Private, the user is asked (default No) whether
   to switch it back. Both changes run in one elevated PowerShell, which only starts if
   there is something to do.
6. Stray caches outside the folder: `%USERPROFILE%\.cache\huggingface\hub\models--<ids we
   use>` (also under a user-level `HF_HOME` / `HF_HUB_CACHE`), `%USERPROFILE%\.cache\torch\hub\checkpoints\<our files>`,
   the uv cache only if the manifest says it was outside the folder (never, normally),
   and `%TEMP%\voicefilter*`. These are listed with their sizes and removed only after a yes
   (default No).
7. A pre-existing ffmpeg is reported as left alone. No system Python or ffmpeg is ever installed.
8. `-RemoveAppFolder` (or a yes to the final question, default No): starts a hidden
   PowerShell that waits until the uninstaller and the `cmd.exe` running `uninstall.bat`
   have exited, then deletes the folder (Remove-Item, then `rd /s /q` as a fallback, with
   retries). This replaces the suggested `ping & rmdir`: it does not depend on timing, and it
   handles a folder name with quotes. Refused for drive roots, the user profile, Desktop,
   Documents, Downloads, Windows and Program Files, and for folders without `uninstall.ps1`.
   Top-level items that are not part of Voice Filter are listed before the question.
   `uninstall.bat` changes to `%SystemRoot%` first and runs inside a `( ... )` block, so cmd
   never has to read the deleted batch file again.
9. Prints what was removed, what was left (with reasons) and what failed. Exit code 0 also
   when everything was already gone; 1 only if something could not be removed. If a step
   outside the folder failed (UAC declined), the manifest is kept so that a second run can
   finish it.

`-Yes` means "accept the default answer": everything that belongs to Voice Filter is
removed, but shared things (stray caches, the network profile, the app folder unless
`-RemoveAppFolder`) are left alone and listed.

## What was verified here (Linux build box)

PowerShell 7.7.0 was installed from Microsoft's apt repository
(`packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb`, then
`apt-get install powershell`). The GitHub release tarball was not needed.
PSScriptAnalyzer could **not** be installed: powershellgallery.com is unreachable from
this box, it is not on nuget.org, and building it from source was not allowed in this
session. Its usual rules (aliases, `-LiteralPath`, unused parameters) are covered by our
own AST lint instead.

`pwsh -NoProfile -File tools/test_scripts/Test-InstallScripts.ps1` (about 7300 checks, about 15 s, all pass):

* **Parser**: both scripts (and the test scripts) parse with
  `[System.Management.Automation.Language.Parser]::ParseFile` with no errors.
* **AST lint**:
  * no PowerShell 7-only syntax (ternary, `??`, `??=`, `?.`, `?[]`, `&&`, `||`), commands
    (`Join-String`, `Test-Json`, ...), parameters (`-AsHashtable`, `-AdditionalChildPath`,
    `-SkipCertificateCheck`, `-Parallel`, ...) or variables (`$IsWindows`, ...);
  * no aliases;
  * file cmdlets only with `-LiteralPath`;
  * no `Out-File` / `Set-Content` (their encodings differ between 5.1 and 7);
  * no variable names containing `?` (this check found a real bug: `"... $from?"`);
  * every parameter and every function is used;
  * BOM + CRLF for the .ps1 files, ASCII + CRLF for the .bat files, and both .bat files
    pass `%*` through.
* **Unit tests** of the pure functions:
  * nvidia-smi parsing (three header variants), driver → CUDA fallback, the tag choice for
    CUDA 12.2 / 12.6 / 12.8 / 12.9 / 13.2 / none / `-Cpu` / `-TorchIndex`, and the preset
    list;
  * PowerShell literal quoting, including typographic quotes, round-tripped through
    `Invoke-Expression`;
  * command-line quoting round-tripped through an MSVCRT argv parser (spaces, umlauts,
    trailing backslash, embedded quotes, empty string);
  * path containment, JSON extraction from program output, and the native-command runner
    (stderr does not throw under `ErrorActionPreference=Stop`, exit codes);
  * the generated `start.bat` (ASCII, CRLF, no blocks, non-ASCII paths left out) and
    `start.ps1` (parses), the browser poller and the elevated helper scripts (parse);
  * manifest merge (union of created paths, shortcuts and rules; first install time and
    network change kept), and JSON written without BOM with one-element arrays kept;
  * soft vs. fatal steps, uninstall path-safety guards, and a real run of the folder-removal
    helper on a scratch folder.
* **Mocked end-to-end runs**: install.ps1 and uninstall.ps1 really run (dot-sourced,
  `Invoke-Main`) in a folder named `Jörg Müller/Voice Filter (test)`. Fake `uv.exe`,
  `nvidia-smi.exe` and `ffmpeg.exe` are shell scripts. The fake venv `python.exe` forwards to
  the dev Python with real torch, so the installer's real torch and CUDA probes and
  `-m voicefilter.download` / `-m voicefilter.selfcheck` (stub modules) are executed.
  Firewall, network profile, UAC and shortcuts are mocked with a shared state file; the
  generated elevated scripts are actually executed against these mocks. Scenarios:
  * **S1** fresh install with `-Yes`: the primary ffmpeg URL fails and BtbN is used; the
    network is Public and stays Public. Checked: every file, `config.json`, the manifest,
    the firewall rule, the uv command lines, the constraints file, and the environment the
    model download sees (HF_HOME, TEMP and the uv cache inside the folder).
  * **S2** re-run with `-Port 9000 -SkipModels` and ffmpeg on PATH: the rule is replaced,
    `.venv` is reused, ffmpeg is detected as `pre-existing`, the first install time is kept.
  * **S3** interactive re-run, answering "y" to Private: the network is switched and
    recorded, and a current rule is not touched. This scenario found a real bug: the rule
    was deleted when only the profile changed.
  * **S4** interactive uninstall with stray caches present: the folder is back to its exact
    original content, rules and shortcuts are gone, the network is switched back after
    consent, strays are listed but kept, and the pre-existing ffmpeg is reported.
  * **S5** second uninstall: exit 0, "nothing needed".
  * **S6** model download fails: exit code 2, the rest is installed, the manifest says
    `models_downloaded: false`.
  * **S7** uninstall removes the strays after consent. This scenario found a real bug: the
    same temp file was listed twice through TEMP and TMP.
  * **S8** driver reports CUDA 12.8 and the index never delivers: cu128, cu126 and cu124 are
    tried, nothing newer, no CPU fallback; exit 1 with the step name and a re-run hint; the
    manifest is still written.
  * **S9** uninstall after the failed install leaves a clean folder.
* `python3 tools/test_scripts/check_scripts.py` is a fallback for machines without pwsh. It
  checks:
  * encodings and line endings;
  * balanced brackets and terminated strings / here-strings;
  * PS7 operators and `"$name?"` in strings;
  * that every parameter is used and the .bat wrappers pass `%*`;
  * **artifact symmetry**: every path in `Get-AppPaths` / `Add-Created` must be under
    something uninstall.ps1 removes, and the firewall patterns and shortcut names must
    match. Mutation-tested: it catches an injected `??`, `"$from?"`, and a removed
    `data` entry.
* `python3 tools/test_scripts/normalize_scripts.py [--check]` rewrites the .ps1 files as
  UTF-8 with BOM + CRLF and the .bat files as ASCII + CRLF. Run it after editing.

## Not verified (needs the Windows PC)

* **Windows PowerShell 5.1 itself.** The scripts avoid every 5.1 incompatibility we know of,
  but they were only executed on PowerShell 7.7 on Linux. Areas not covered:
  * StrictMode differences in 5.1;
  * 5.1's legacy native-argument quoting: the `python -c` snippets contain no double quotes
    for this reason;
  * `ConvertTo-Json` output in 5.1.
* Real `Get-NetFirewallRule` / `New-NetFirewallRule` / `Get-NetConnectionProfile` /
  `Set-NetConnectionProfile`, the UAC prompt through `Start-Process -Verb RunAs`,
  `Get-CimInstance Win32_Process`, `Get-NetTCPConnection`, the `WScript.Shell` shortcuts,
  `[Environment]::GetFolderPath`, and the `LongPathsEnabled` registry read.
* Real downloads: uv's latest zip layout, uv's own Python download (from GitHub), the
  PyTorch CUDA wheels (`download.pytorch.org` is blocked here), and the gyan.dev / BtbN zip
  layouts.
* That the chosen CUDA index actually has wheels for the latest torch on Python 3.12
  Windows (assumption: at least one of the four newest tags does).
* `cmd.exe` running the generated `start.bat` (quoting, `%~dp0` with umlauts, the
  `start "" /min powershell ...` browser helper), and the `exit /b` errorlevel behaviour
  inside the block in `uninstall.bat`.
* SmartScreen / Mark-of-the-Web prompts for files from a downloaded zip, Defender
  "Controlled folder access", and antivirus quarantining `uv.exe`.

## Manual test checklist (user's PC: Windows 11, RTX 3080)

Use a test folder whose name has a space and an umlaut, e.g. `C:\Users\<you>\Jörg Test\Voice Filter`.
If the folder came from a downloaded zip: right-click the zip, Properties, tick **Unblock**, then extract.

1. **Fresh install**: double-click `install.bat`.
   - [ ] The plan is printed and "Continue? [Y/n]" appears.
   - [ ] The NVIDIA driver and the CUDA version are shown.
   - [ ] PyTorch downloads with a progress bar.
   - [ ] "CUDA works: NVIDIA GeForce RTX 3080" appears (not the CUDA warning box).
   - [ ] ffmpeg is reported as pre-existing or portable.
   - [ ] The models download.
   - [ ] **Exactly one** UAC prompt, for the firewall.
   - [ ] If the WiFi is Public, you are asked to switch it, default No.
   - [ ] The summary shows the phone URL.
   - [ ] The window stays open ("Press any key").
2. Check:
   - [ ] `wf.msc` shows "Voice Filter (TCP 8765)": inbound, Allow, Private only, TCP 8765.
   - [ ] Desktop and Start Menu have "Voice Filter" shortcuts.
   - [ ] `config.json` has `"device": "cuda"`.
   - [ ] `data\logs\install-*.log` exists.
   - [ ] Nothing new in `%APPDATA%\uv`, `%LOCALAPPDATA%\uv`, `%USERPROFILE%\.cache` or
     `%USERPROFILE%\.local\bin`.
   - [ ] No new "Python 3.12" under Settings > Apps, and `py -0p` does not list the new Python.
   - [ ] The user PATH is unchanged (`[Environment]::GetEnvironmentVariable('Path','User')`
     before and after).
3. **Start**:
   - [ ] Double-click the Desktop shortcut: a console opens, the LAN URLs are printed, and
     the browser opens `http://localhost:8765` by itself.
   - [ ] From the phone, on the same WiFi, open the URL: the page loads.
   - [ ] Process one short video.
4. **Re-run**:
   - [ ] Run `install.bat` again: it is fast, `.venv` / uv / ffmpeg / PyTorch are reported
     as already present, and there is **no** UAC prompt.
   - [ ] Run `install.bat -Port 9000 -SkipModels -Yes`: one UAC prompt, the rule is now
     9000 and the 8765 rule is gone, start.bat uses 9000.
5. **Repair**:
   - [ ] Delete `.venv\Scripts\python.exe`, run `install.bat -Yes`: the venv is recreated.
   - [ ] Delete `runtime\ffmpeg`, run `install.bat -Yes`: ffmpeg is downloaded again (if it
     is not on PATH).
6. **Partial failure**:
   - [ ] Disconnect the network during the PyTorch download: a clear "PyTorch failed ...
     re-run" message and exit 1.
   - [ ] Reconnect, run `install.bat`: the install finishes.
   - [ ] Decline the UAC prompt: the firewall warning appears, exit code 2, everything else
     works on this PC.
   - [ ] Re-run and accept the prompt: the rule is created.
7. **Install while the app runs**: with the start.bat window open, run `install.bat`. It
   offers to stop the server, and the install succeeds.
8. **CPU path** (optional): `install.bat -Cpu -Yes -SkipModels` installs the `+cpu` torch
   with a warning and `"device": "cpu"`. Run the normal install again afterwards.
9. **PowerShell 7** (optional): `pwsh -File install.ps1 -Yes -SkipModels` works as well.
10. **Uninstall**: start the app, then double-click `uninstall.bat`.
    - [ ] The server window is closed.
    - [ ] One UAC prompt, for the firewall.
    - [ ] You are asked about the network only if the installer had switched it.
    - [ ] Stray caches are listed, if any.
    - [ ] "Also delete the app folder?" appears, default No.
    - [ ] The final list is shown.
    - [ ] The folder only contains the original files again; shortcuts and the firewall
      rule are gone.
11. Run `uninstall.bat -Yes` again: exit 0, "Nothing needed to be removed".
12. **Remove folder**: install again with `-SkipModels`, then run `uninstall.bat -Yes -RemoveAppFolder`.
    - [ ] After the window closes, the whole folder disappears within a few seconds.
13. **Keep models**: `uninstall.bat -Yes -KeepModels` keeps only `models\`. A following
    `install.bat` does not download the models again.

## Running the tests

```
# Linux (Ubuntu 24.04): sudo dpkg -i packages-microsoft-prod.deb && sudo apt-get install powershell
pwsh -NoProfile -File tools/test_scripts/Test-InstallScripts.ps1            # all checks
pwsh -NoProfile -File tools/test_scripts/Test-InstallScripts.ps1 -SkipE2E   # static + unit only
pwsh -NoProfile -File tools/test_scripts/Test-InstallScripts.ps1 -KeepTemp  # keep e2e transcripts
python3 tools/test_scripts/check_scripts.py                                 # without pwsh
python3 tools/test_scripts/normalize_scripts.py                             # fix BOM/CRLF after edits
```

The end-to-end part needs `/usr/bin/ffmpeg`, `python3` and the dev `.venv` (with torch).
On Windows it would need adapting: the fake executables are shell scripts.
