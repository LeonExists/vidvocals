# Voice Filter

Keeps only the **female** vocal sounds in a video (speech, moans, grunts, sighs, laughs,
screams and other vocalizations) and removes the male ones. Music and background stay
untouched. Runs locally on a Windows PC with an NVIDIA GPU, and you can use it from your
phone on the same WiFi. Everything is free and runs offline after installation.

How it works, briefly: ffmpeg extracts the audio; a BS-RoFormer model splits vocals from
music and noise; every vocal event on the vocal track is found and classified male/female
(WavLM features + pitch, plus a check of consistency per speaker across the whole video);
male parts of the vocal track are muted with 15 ms crossfades; the result is mixed back with
the untouched background and muxed into the original video **without re-encoding the video
stream**. Details: [PLAN.md](PLAN.md), [docs/MODELS.md](docs/MODELS.md).

## Requirements

- Windows 11 (10 should also work), 64-bit
- NVIDIA GPU with a current driver (tested target: RTX 3080 10 GB). Works on CPU too, but very slowly.
- About 10 GB free disk space (bundled Python, PyTorch with CUDA, models); the installer warns below 15 GB
- Internet during installation only

Nothing needs to be installed beforehand. The installer puts Python, PyTorch, ffmpeg and all
models **inside this folder**. It does not change your PATH, your system Python or any
existing ffmpeg.

## Install

1. Put this folder where you want the app to live (e.g. `C:\Apps\Voice Filter`). Avoid
   OneDrive-synced folders.
2. Double-click **`install.bat`**.
   - One Windows admin prompt (UAC) appears to create the firewall rule for phone access.
     If you decline it, everything works except phone access.
   - If your WiFi is set to *Public*, the installer explains that the phone will be blocked
     and asks whether to switch it to *Private* (default: No).
3. Wait. The first install downloads ~3–4 GB (PyTorch with CUDA, models) and takes 5–20 minutes.

Options (run from PowerShell in this folder, e.g. `.\install.bat -Presets fast,balanced`):

| Option | Meaning |
|---|---|
| `-Presets fast,balanced,best` | which presets' models to download (default: all) |
| `-Port 8765` | port for the web app |
| `-SkipModels` | don't pre-download models (they download on first use) |
| `-NoFirewall`, `-NoShortcuts` | skip those steps |
| `-Cpu` | install the CPU build of PyTorch |
| `-TorchIndex cu128` | force a specific PyTorch CUDA build (troubleshooting) |
| `-Yes` | no questions, take the defaults |

Re-running `install.bat` is safe: finished steps are reused and broken ones repaired.
Logs: `data\logs\install-*.log`.

## Start

Double-click **Voice Filter** on the Desktop or in the Start menu (or `start.bat` in this
folder). A console window shows:

```
Open on this PC:      http://localhost:8765
Open on your phone (same WiFi):
                      http://192.168.x.y:8765
```

and the browser opens. Close the console window (or press Ctrl+C) to stop the app.

## Use it from your phone

1. Phone and PC on the same WiFi.
2. Open the `http://192.168.x.y:8765` address from the console (it is also shown at the
   top of the app, with a copy button) in the phone's browser.
3. If it does not load: the PC's network must be *Private* (Settings → Network & internet →
   WiFi → your network → Private), and the firewall rule must exist (re-run `install.bat`).

There is no login: anyone on your WiFi who knows the address can use the app while it runs.
Don't start it on untrusted networks.

## Using the app

1. Drop a video into the upload box (or tap it to pick a file).
2. Choose a preset:
   - **Fast**: quickest, lighter vocal separation.
   - **Balanced**: better separation, the recommended default.
   - **Best**: highest-quality separation plus separating overlapping male+female speech so
     that only the female voice is kept there. Slowest.
3. Optionally turn off *Keep music & background* (output then contains only the kept voices),
   or turn on *Export automatically* to skip the review.
4. When analysis is done, open **Review**:
   - The timeline shows every detected vocal segment: filled blue = kept, orange outline with
     a cross = removed; ♀/♂ icons show the model's label; striped with ⚠ = the model is unsure.
   - Tap a segment to see its details and play it, then **Keep** / **Remove** / **Reset**.
   - Use the ⚠ arrows to jump through all unsure segments; this is where mistakes are most likely.
   - The *Speakers* tab lets you keep or remove everything from one detected speaker at once.
   - Switch the player between *Original*, *Vocals* and *Preview* (the filtered result).
5. **Export video**, then **Download**. The output is `<name>.voicefilter.<ext>`: same video
   stream (bit-identical), new audio track, same length and sync.

Headless use: `.venv\Scripts\python -m voicefilter.cli process in.mp4 -o out.mp4 --preset balanced`.

## Uninstall

Double-click **`uninstall.bat`**. It stops the app and removes the venv, the bundled Python,
uv, the portable ffmpeg, all models and caches, jobs/uploads/temp files, the generated start
scripts and config, the firewall rule and the shortcuts. Anything outside this folder that
might be shared (an ffmpeg you had before, Hugging Face/torch caches in your user profile
from other tools, a network profile the installer switched to Private) is only touched after
you confirm. At the end it offers to delete the app folder itself.

Options: `-Yes` (no questions; shared things are left alone), `-KeepModels`,
`-RemoveAppFolder`. Safe to run more than once.

## Accuracy (honest numbers)

ACCURACY_SECTION

## Limitations

- **Not 100 %.** Use the review view, especially the ⚠ unsure segments, before exporting
  anything important.
- **Overlapping voices**: when a man and a woman are audible at the same moment, Fast and
  Balanced can only keep or remove the whole moment. Best separates the two voices and keeps
  only the female one, with some artifacts.
- **Hard sounds**: breathing, sighs, panting, whispers, clicks and very short sounds
  (< 0.4 s) carry little gender information; the app leans on the rest of the same speaker's
  sounds, which can be wrong when two people sound similar.
- **Atypical voices**: very low female or very high male voices, pitch-shifted or heavily
  processed audio, and children's voices are less reliable.
- **Training data licenses**: the gender classifier was trained partly on datasets licensed
  for non-commercial use (EARS, Deeply Nonverbal). Personal use is fine; don't use it
  commercially. See [eval/DATA.md](eval/DATA.md) and [docs/MODELS.md](docs/MODELS.md).
- **Not tested on the target PC**: this was built and tested on a Linux machine without a
  GPU. See *What was verified* below.

## What was verified

VERIFIED_SECTION

## Project layout

| Path | What |
|---|---|
| `voicefilter/` | Python backend (pipeline, server, model download) and `web/` (UI) |
| `eval/` | evaluation data builder, metrics, gender-classifier training, reports |
| `tests/` | pytest suite (`.venv\Scripts\python -m pytest -q`) |
| `docs/` | model research, install notes, UI screenshots |
| `install.*`, `uninstall.*` | Windows scripts; `tools/test_scripts/` checks them |
