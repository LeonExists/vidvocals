<#
    Mocked end-to-end scenarios for install.ps1 / uninstall.ps1 (dot-sourced by
    Test-InstallScripts.ps1, which provides Assert-True / Assert-Equal).

    Fake uv.exe / python.exe / nvidia-smi.exe / ffmpeg.exe are small shell scripts; the
    "venv" python forwards to the real dev interpreter (needs torch installed), so the
    real Python code paths of the installer (torch version + CUDA check, model download
    and self check modules) are exercised.
#>
param([Parameter(Mandatory = $true)][string]$TmpRoot)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$realPy = Join-Path $repo '.venv/bin/python'
if (-not (Test-Path -LiteralPath $realPy)) { $realPy = (Get-Command python3).Source }

$root = Join-Path $TmpRoot 'e2e'
$fx = Join-Path $root 'fixtures'
$home2 = Join-Path $root 'home'
$sysTemp = Join-Path $root 'systemp'
$app = Join-Path $root "Jörg Müller/Voice Filter (test)"
$stateFile = Join-Path $root 'state.json'
$uvLog = Join-Path $root 'uv-calls.log'
foreach ($d in @($fx, (Join-Path $fx 'bin'), (Join-Path $fx 'pathffmpeg'), $home2, $sysTemp, $app)) { $null = New-Item -ItemType Directory -Path $d -Force }

function Write-Exe([string]$Path, [string]$Body) {
    [System.IO.File]::WriteAllText($Path, "#!/bin/sh`n" + $Body.Replace("`r`n", "`n") + "`n")
    & chmod 755 $Path
}

# ---- fixtures --------------------------------------------------------------
Write-Exe (Join-Path $fx 'bin/nvidia-smi.exe') @'
if [ "$1" = "--query-gpu=name,memory.total" ]; then echo "NVIDIA GeForce RTX 3080, 10240 MiB"; exit 0; fi
echo "+-----------------------------------------------------------------------------+"
echo "| NVIDIA-SMI 581.57       Driver Version: 581.57       CUDA Version: ${VF_MOCK_CUDA:-13.2}     |"
exit 0
'@
$uvSrc = Join-Path $fx 'uvsrc/uv-x86_64-pc-windows-msvc'
$null = New-Item -ItemType Directory -Path $uvSrc -Force
Write-Exe (Join-Path $uvSrc 'uv.exe') @'
echo "$*" >> "$VF_MOCK_UVLOG"
case "$1" in
  --version) echo "uv 0.11.99 (mock)"; exit 0;;
  venv)
    for last; do :; done
    mkdir -p "$last/Scripts"
    printf '#!/bin/sh\nif [ -n "$VF_EXPECT_BASE" ]; then VF_EXPECT_BASE=/; export VF_EXPECT_BASE; fi\nexec "$VF_MOCK_REALPY" "$@"\n' > "$last/Scripts/python.exe"
    chmod 755 "$last/Scripts/python.exe"
    echo "Creating virtual environment at: $last"; exit 0;;
  pip) echo "mock pip $*"; exit ${VF_MOCK_PIP_EXIT:-0};;
  cache) exit 0;;
esac
exit 0
'@
Write-Exe (Join-Path $uvSrc 'uvx.exe') 'exit 0'
$ffSrc = Join-Path $fx 'ffsrc/ffmpeg-7.1-essentials_build'
$null = New-Item -ItemType Directory -Path (Join-Path $ffSrc 'bin') -Force
Write-Exe (Join-Path $ffSrc 'bin/ffmpeg.exe') 'exec /usr/bin/ffmpeg "$@"'
Write-Exe (Join-Path $ffSrc 'bin/ffprobe.exe') 'exec /usr/bin/ffprobe "$@"'
[System.IO.File]::WriteAllText((Join-Path $ffSrc 'LICENSE'), 'GPL')
Write-Exe (Join-Path $fx 'pathffmpeg/ffmpeg.exe') 'exec /usr/bin/ffmpeg "$@"'
Write-Exe (Join-Path $fx 'pathffmpeg/ffprobe.exe') 'exec /usr/bin/ffprobe "$@"'
# zips with unix permissions (python's zipfile keeps them in external_attr)
$zipper = @'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    base = os.path.dirname(src)
    for d, _, files in os.walk(src):
        for f in files:
            p = os.path.join(d, f)
            info = zipfile.ZipInfo(os.path.relpath(p, base))
            info.external_attr = (os.stat(p).st_mode & 0xFFFF) << 16
            with open(p, "rb") as fh:
                z.writestr(info, fh.read())
'@
$zipperPath = Join-Path $root 'zipper.py'
[System.IO.File]::WriteAllText($zipperPath, $zipper)
& python3 -I $zipperPath $uvSrc (Join-Path $fx 'uv.zip')
& python3 -I $zipperPath $ffSrc (Join-Path $fx 'ffmpeg.zip')

# ---- test app folder -------------------------------------------------------
foreach ($f in @('install.ps1', 'uninstall.ps1', 'install.bat', 'uninstall.bat')) { Copy-Item -LiteralPath (Join-Path $repo $f) -Destination (Join-Path $app $f) }
$pkg = Join-Path $app 'voicefilter'
$null = New-Item -ItemType Directory -Path $pkg -Force
[System.IO.File]::WriteAllText((Join-Path $pkg '__init__.py'), '')
[System.IO.File]::WriteAllText((Join-Path $pkg 'download.py'), @'
import argparse, json, os, pathlib, sys, tempfile
ap = argparse.ArgumentParser(); ap.add_argument("--presets"); a = ap.parse_args()
hub = pathlib.Path(os.environ["HF_HOME"]) / "hub" / "models--org--voice-model" / "snapshots"
hub.mkdir(parents=True, exist_ok=True); (hub / "weights.bin").write_bytes(b"x" * 10)
ck = pathlib.Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
ck.mkdir(parents=True, exist_ok=True); (ck / "model.ckpt").write_bytes(b"y")
snap = {k: os.environ.get(k) for k in ("VF_APP_DIR", "HF_HOME", "TORCH_HOME", "TEMP", "TMP", "XDG_CACHE_HOME", "PYTHONUTF8",
                                        "HF_HUB_DISABLE_TELEMETRY", "UV_CACHE_DIR", "VF_PORT", "VF_FFMPEG")}
snap["presets"] = a.presets
(pathlib.Path(os.environ["VF_APP_DIR"]) / "data" / "env-snapshot.json").write_text(json.dumps(snap), encoding="utf-8")
print("downloaded", a.presets)
sys.exit(int(os.environ.get("VF_MOCK_DOWNLOAD_EXIT", "0")))
'@)
[System.IO.File]::WriteAllText((Join-Path $pkg 'selfcheck.py'), @'
import json, shutil, sys, torch
print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "gpu_name": None, "ffmpeg_found": True}))
sys.exit(0)
'@)
[System.IO.File]::WriteAllText((Join-Path $app 'requirements.txt'), "fastapi`n")
$originalListing = @(Get-ChildItem -LiteralPath $app -Recurse -Force | ForEach-Object { $_.FullName.Substring($app.Length) } | Sort-Object)

[System.IO.File]::WriteAllText($stateFile, '{"rules": [], "category": "Public", "events": []}')

# ---- environment for the child processes ------------------------------------
$basePath = $env:PATH
$env:VF_MOCK_FIXTURES = $fx
$env:VF_MOCK_HOME = $home2
$env:VF_MOCK_UVLOG = $uvLog
$env:VF_MOCK_REALPY = $realPy
$env:USERPROFILE = $home2
$env:TEMP = $sysTemp; $env:TMP = $sysTemp; $env:TMPDIR = $sysTemp + '/'
$env:VF_MOCK_GYAN_FAIL = '1'

function Invoke-Mocked([string]$ScriptName, [string]$ArgsJson, [string]$Answers, [string]$Label) {
    $out = & pwsh -NoProfile -File (Join-Path $PSScriptRoot 'Invoke-MockedRun.ps1') -AppDir $app -Script $ScriptName -StateFile $stateFile -ArgsJson $ArgsJson -Answers $Answers 2>&1
    $text = ($out | ForEach-Object { "$_" }) -join "`n"
    [System.IO.File]::WriteAllText((Join-Path $root "$Label.out.txt"), $text)
    $m = [regex]::Match($text, 'EXITCODE=(-?\d+)')
    if (-not $m.Success) { Write-Host $text; return -999 }
    return [int]$m.Groups[1].Value
}
function Read-Json([string]$p) { return ([System.IO.File]::ReadAllText($p) | ConvertFrom-Json) }

# ---- S1: fresh install, portable ffmpeg (primary URL fails), Public network, -Yes
$env:PATH = (Join-Path $fx 'bin') + ':' + $basePath
$env:VF_MOCK_CUDA = '13.2'
$code = Invoke-Mocked 'install' '{"Yes": true}' '' 's1-install'
Assert-Equal 0 $code 'S1 install exit code'
foreach ($rel in @('.venv/Scripts/python.exe', 'runtime/uv/uv.exe', 'runtime/ffmpeg/bin/ffmpeg.exe', 'runtime/ffmpeg/bin/ffprobe.exe',
                   'runtime/ffmpeg/LICENSE', 'runtime/torch-constraints.txt', 'start.bat', 'start.ps1', 'config.json', 'install-manifest.json',
                   'models/hf/hub/models--org--voice-model', 'data/logs', 'data/tmp')) {
    Assert-True (Test-Path -LiteralPath (Join-Path $app $rel)) "S1 created $rel"
}
Assert-True (-not (Test-Path -LiteralPath (Join-Path $app 'runtime/uv/_extract'))) 'S1 removed the uv extraction folder'
Assert-True (-not (Test-Path -LiteralPath (Join-Path $app 'runtime/ffmpeg/_extract'))) 'S1 removed the ffmpeg extraction folder'
$cfg = Read-Json (Join-Path $app 'config.json')
Assert-Equal 8765 $cfg.port 'S1 config port'
Assert-Equal 'cpu' $cfg.device 'S1 config device (no real GPU here)'
Assert-Equal 'portable' $cfg.ffmpeg_source 'S1 config ffmpeg source'
Assert-Equal (Join-Path $app 'runtime/ffmpeg/bin/ffmpeg.exe') $cfg.ffmpeg 'S1 config ffmpeg path'
$man = Read-Json (Join-Path $app 'install-manifest.json')
Assert-Equal 'portable' $man.ffmpeg.source 'S1 manifest ffmpeg'
Assert-Equal 'uv-managed' $man.python.source 'S1 manifest python'
Assert-Equal 'Voice Filter (TCP 8765)' $man.firewall_rule 'S1 manifest firewall rule'
Assert-True ($null -eq $man.network_profile_changed) 'S1 network profile not changed in -Yes mode'
Assert-Equal 2 @($man.shortcuts).Count 'S1 manifest shortcuts'
Assert-True (@($man.created) -contains (Join-Path $app '.venv')) 'S1 manifest created contains .venv'
Assert-Equal @('models--org--voice-model') @($man.hf_models) 'S1 manifest hf models'
Assert-True ($man.torch.torch -like '*+cu132') 'S1 manifest torch version'
$st = Read-Json $stateFile
Assert-Equal @('Voice Filter (TCP 8765)') @($st.rules | ForEach-Object { $_.DisplayName }) 'S1 firewall rule created'
Assert-Equal 'Private' $st.rules[0].Profile 'S1 firewall rule profile'
Assert-Equal 'Public' $st.category 'S1 network still Public'
Assert-True (@($st.events) -contains 'download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip') 'S1 tried gyan.dev first'
Assert-True (@($st.events | Where-Object { $_ -like 'download: *BtbN*' }).Count -eq 1) 'S1 fell back to BtbN'
$uvCalls = [System.IO.File]::ReadAllText($uvLog)
Assert-True ($uvCalls -match 'venv --python 3\.12 --managed-python --no-project') 'S1 uv venv call'
Assert-True ($uvCalls -match 'pip install --python .*python\.exe -r .*requirements\.txt -c .*torch-constraints\.txt') 'S1 requirements installed with constraints'
Assert-True ($uvCalls -notmatch '--index') 'S1 torch already matching cu132 -> not reinstalled'
$cons = [System.IO.File]::ReadAllText((Join-Path $app 'runtime/torch-constraints.txt'))
Assert-True ($cons -match 'torch==\S+\+cu132' -and $cons -match 'torchaudio==\S+\+cu132') 'S1 constraints pin torch + torchaudio'
$snap = Read-Json (Join-Path $app 'data/env-snapshot.json')
Assert-Equal (Join-Path $app 'models/hf') $snap.HF_HOME 'S1 model download HF_HOME inside app'
Assert-Equal (Join-Path $app 'data/tmp') $snap.TEMP 'S1 model download TEMP inside app'
Assert-Equal (Join-Path $app 'runtime/uv-cache') $snap.UV_CACHE_DIR 'S1 uv cache inside app'
Assert-Equal 'fast,balanced,best' $snap.presets 'S1 default presets'
Assert-Equal '1' $snap.PYTHONUTF8 'S1 PYTHONUTF8'
$logs = @(Get-ChildItem -LiteralPath (Join-Path $app 'data/logs') -Filter 'install-*.log')
Assert-Equal 1 $logs.Count 'S1 log file'
$logText = [System.IO.File]::ReadAllText($logs[0].FullName)
Assert-True ($logText.Contains('Self check passed')) 'S1 log has self check'
Assert-True ($logText.Contains('http://192.168.1.23:8765')) 'S1 summary prints the phone URL'
Assert-True ($logText.Contains("network 'HomeWifi' (WLAN) is set to 'Public'") -or $logText.Contains("The network 'HomeWifi' (WLAN) is set to 'Public'")) 'S1 warns about the Public network'
$bat = [System.IO.File]::ReadAllBytes((Join-Path $app 'start.bat'))
Assert-True (@($bat | Where-Object { $_ -gt 127 }).Count -eq 0) 'S1 start.bat ASCII'
$sp = [System.IO.File]::ReadAllBytes((Join-Path $app 'start.ps1'))
Assert-True ($sp[0] -eq 0xEF -and $sp[1] -eq 0xBB) 'S1 start.ps1 has BOM'
Assert-True (Test-Path -LiteralPath (Join-Path $home2 'Desktop/Voice Filter.lnk')) 'S1 desktop shortcut'
$s1Installed = $man.first_installed_at

# ---- S2: re-run with another port, pre-existing ffmpeg on PATH, -SkipModels --------
$env:PATH = (Join-Path $fx 'bin') + ':' + (Join-Path $fx 'pathffmpeg') + ':' + $basePath
$code = Invoke-Mocked 'install' '{"Yes": true, "Port": 9000, "SkipModels": true}' '' 's2-install'
Assert-Equal 0 $code 'S2 re-install exit code'
$st = Read-Json $stateFile
Assert-Equal @('Voice Filter (TCP 9000)') @($st.rules | ForEach-Object { $_.DisplayName }) 'S2 firewall rule replaced for the new port'
$man = Read-Json (Join-Path $app 'install-manifest.json')
Assert-Equal $s1Installed $man.first_installed_at 'S2 first install time kept'
Assert-Equal 'pre-existing' $man.ffmpeg.source 'S2 pre-existing ffmpeg detected'
Assert-Equal @('Voice Filter (TCP 8765)', 'Voice Filter (TCP 9000)') @($man.firewall_rules_created) 'S2 manifest remembers both rules'
Assert-Equal @('models--org--voice-model') @($man.hf_models) 'S2 keeps hf model list from S1'
$cfg = Read-Json (Join-Path $app 'config.json')
Assert-Equal 9000 $cfg.port 'S2 config port'
Assert-Equal (Join-Path $fx 'pathffmpeg/ffmpeg.exe') $cfg.ffmpeg 'S2 config uses PATH ffmpeg'
$batText = [System.IO.File]::ReadAllText((Join-Path $app 'start.bat'))
Assert-True ($batText.Contains('set "VF_PORT=9000"')) 'S2 start.bat port updated'
Assert-True ($batText.Contains('rem VF_FFMPEG: path contains non-ASCII')) 'S2 start.bat leaves the non-ASCII ffmpeg path to config.json'
Assert-True (@(Get-ChildItem -LiteralPath (Join-Path $app 'data/logs') -Filter 'install-*.log').Count -ge 1) 'S2 log written'
$uvCalls = [System.IO.File]::ReadAllText($uvLog)
Assert-Equal 1 ([regex]::Matches($uvCalls, '(?m)^venv ').Count) 'S2 reused the existing .venv'

# ---- S3: interactive re-run, user agrees to switch the network to Private --------
$code = Invoke-Mocked 'install' '{"Port": 9000, "SkipModels": true}' '|y' 's3-install'
Assert-Equal 0 $code 'S3 interactive install exit code'
$st = Read-Json $stateFile
Assert-Equal 'Private' $st.category 'S3 network switched to Private after consent'
Assert-Equal 1 @($st.events | Where-Object { $_ -like 'firewall new:*' -and $_ -like '*9000*' }).Count 'S3 did not recreate a current rule'
$man = Read-Json (Join-Path $app 'install-manifest.json')
Assert-Equal 'WLAN' $man.network_profile_changed.interface 'S3 manifest records the network change'
Assert-Equal 'Public' $man.network_profile_changed.from 'S3 manifest records the old category'

# ---- S4: stray caches + interactive uninstall (switch back yes, strays no, folder no)
$strayHub = Join-Path $home2 '.cache/huggingface/hub/models--org--voice-model'
$null = New-Item -ItemType Directory -Path $strayHub -Force
$strayTmp = Join-Path $sysTemp 'voicefilter-old.tmp'
[System.IO.File]::WriteAllText($strayTmp, 'x')
$code = Invoke-Mocked 'uninstall' '{}' '|y|n|n' 's4-uninstall'
Assert-Equal 0 $code 'S4 uninstall exit code'
$now = @(Get-ChildItem -LiteralPath $app -Recurse -Force | ForEach-Object { $_.FullName.Substring($app.Length) } | Sort-Object)
$extra = @($now | Where-Object { $originalListing -notcontains $_ })
Assert-Equal @() $extra 'S4 app folder is back to its original content'
$st = Read-Json $stateFile
Assert-Equal 0 @($st.rules | Where-Object { $null -ne $_ }).Count 'S4 firewall rules removed'
Assert-Equal 'Public' $st.category 'S4 network switched back after consent'
Assert-True (-not (Test-Path -LiteralPath (Join-Path $home2 'Desktop/Voice Filter.lnk'))) 'S4 desktop shortcut removed'
Assert-True (-not (Test-Path -LiteralPath (Join-Path $home2 'Start Menu/Programs/Voice Filter.lnk'))) 'S4 start menu shortcut removed'
Assert-True (Test-Path -LiteralPath $strayHub) 'S4 stray HF cache kept without consent'
Assert-True (Test-Path -LiteralPath $strayTmp) 'S4 stray temp file kept without consent'
$out4 = [System.IO.File]::ReadAllText((Join-Path $root 's4-uninstall.out.txt'))
Assert-True ($out4.Contains($strayHub)) 'S4 lists the stray HF cache'
Assert-True ($out4.Contains($strayTmp)) 'S4 lists the stray temp file'
Assert-True ($out4.Contains('already installed before Voice Filter; left alone')) 'S4 reports pre-existing ffmpeg as left alone'

# ---- S5: uninstall again (idempotent) -------------------------------------------
$code = Invoke-Mocked 'uninstall' '{"Yes": true}' '' 's5-uninstall'
Assert-Equal 0 $code 'S5 second uninstall exit code'
$out5 = [System.IO.File]::ReadAllText((Join-Path $root 's5-uninstall.out.txt'))
Assert-True ($out5.Contains('Nothing needed to be removed')) 'S5 reports nothing to remove'

# ---- S6: model download fails (soft step) -> exit code 2, rest installed ---------
$env:PATH = (Join-Path $fx 'bin') + ':' + $basePath
$env:VF_MOCK_DOWNLOAD_EXIT = '1'
$code = Invoke-Mocked 'install' '{"Yes": true, "Presets": "fast"}' '' 's6-install'
$env:VF_MOCK_DOWNLOAD_EXIT = '0'
Assert-Equal 2 $code 'S6 install with failing model download exits 2'
Assert-True (Test-Path -LiteralPath (Join-Path $app 'start.bat')) 'S6 start.bat still written'
$man = Read-Json (Join-Path $app 'install-manifest.json')
Assert-Equal $false $man.models_downloaded 'S6 manifest models_downloaded false'
$out6 = [System.IO.File]::ReadAllText((Join-Path $root 's6-install.out.txt'))
Assert-True ($out6.Contains('Models:     incomplete')) 'S6 summary says models are incomplete'

# ---- S7: interactive uninstall, strays removed with consent ------------------------
$code = Invoke-Mocked 'uninstall' '{}' '|y|n' 's7-uninstall'
Assert-Equal 0 $code 'S7 uninstall exit code'
Assert-True (-not (Test-Path -LiteralPath $strayHub)) 'S7 stray HF cache removed with consent'
Assert-True (-not (Test-Path -LiteralPath $strayTmp)) 'S7 stray temp file removed with consent'
$now = @(Get-ChildItem -LiteralPath $app -Recurse -Force | ForEach-Object { $_.FullName.Substring($app.Length) } | Sort-Object)
Assert-Equal @() @($now | Where-Object { $originalListing -notcontains $_ }) 'S7 app folder clean again'

# ---- S8: install fails at PyTorch (driver says CUDA 12.8, index never delivers) ----
$env:VF_MOCK_CUDA = '12.8'
[System.IO.File]::WriteAllText($uvLog, '')
$code = Invoke-Mocked 'install' '{"Yes": true}' '' 's8-install'
Assert-Equal 1 $code 'S8 failing install exit code 1'
$uvCalls = [System.IO.File]::ReadAllText($uvLog)
foreach ($tag in @('cu128', 'cu126', 'cu124')) {
    Assert-True ($uvCalls.Contains("--index https://download.pytorch.org/whl/$tag")) "S8 tried the $tag index"
}
Assert-True ($uvCalls -notmatch 'whl/cu13|whl/cpu') 'S8 did not try indexes the driver cannot run / CPU'
Assert-True ($uvCalls.Contains('--reinstall-package torch')) 'S8 reinstalls torch when switching builds'
Assert-True (Test-Path -LiteralPath (Join-Path $app 'install-manifest.json')) 'S8 manifest written despite the failure'
$out8 = [System.IO.File]::ReadAllText((Join-Path $root 's8-install.out.txt'))
Assert-True ($out8.Contains('PyTorch could not be installed')) 'S8 explains the failing step'
Assert-True ($out8.Contains('run install.bat again')) 'S8 hints to re-run'

# ---- S9: non-interactive uninstall after the failed install -------------------------
$code = Invoke-Mocked 'uninstall' '{"Yes": true}' '' 's9-uninstall'
Assert-Equal 0 $code 'S9 uninstall exit code'
$now = @(Get-ChildItem -LiteralPath $app -Recurse -Force | ForEach-Object { $_.FullName.Substring($app.Length) } | Sort-Object)
Assert-Equal @() @($now | Where-Object { $originalListing -notcontains $_ }) 'S9 app folder clean again'

$env:PATH = $basePath
