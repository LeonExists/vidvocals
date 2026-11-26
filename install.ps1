#Requires -Version 5.1
<#
.SYNOPSIS
    Installs Voice Filter into the folder this script lives in.

.DESCRIPTION
    Everything (uv, Python, virtual environment, PyTorch, ffmpeg if needed, models,
    caches, logs) is placed inside the app folder. The only things created outside
    it are the Windows Firewall rule and the Desktop / Start Menu shortcuts.
    The script never changes the user PATH, the PowerShell profile or the registry
    (apart from what Windows itself does for shortcuts and firewall rules).
    It is safe to run again: finished steps are detected and reused, broken ones repaired.

.PARAMETER Presets
    Comma separated model presets to pre-download (default: fast,balanced,best).
.PARAMETER Port
    TCP port of the web server (default 8765).
.PARAMETER SkipModels
    Do not download models now (they are downloaded on first use instead).
.PARAMETER NoFirewall
    Do not create the Windows Firewall rule (phones may not be able to connect).
.PARAMETER NoShortcuts
    Do not create Desktop / Start Menu shortcuts.
.PARAMETER Cpu
    Install the CPU build of PyTorch even if an NVIDIA GPU is present.
.PARAMETER TorchIndex
    Force a PyTorch wheel index tag, e.g. cu128, cu126 or cpu (troubleshooting only).
.PARAMETER Yes
    Non-interactive: accept the default answer for every question. In this mode the
    network profile is never changed.

.EXAMPLE
    install.bat
.EXAMPLE
    install.bat -Yes -Presets fast -Port 9000
#>
[CmdletBinding()]
param(
    [string]$Presets = 'fast,balanced,best',
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [switch]$SkipModels,
    [switch]$NoFirewall,
    [switch]$NoShortcuts,
    [switch]$Cpu,
    [string]$TorchIndex = '',
    [switch]$Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$script:AppName = 'Voice Filter'
$script:FirewallRuleName = "Voice Filter (TCP $Port)"
$script:FirewallRulePatterns = @('Voice Filter (TCP *)', 'VoiceFilter (TCP *)')
$script:UvUrl = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
$script:FfmpegUrls = @(
    'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
    'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
)
$script:PythonVersion = '3.12'
# Packages installed from the PyTorch index. Pin here if the backend needs a specific
# version, e.g. @('torch==2.9.*', 'torchaudio==2.9.*').
$script:TorchPackages = @('torch', 'torchaudio')
$script:TorchIndexBase = 'https://download.pytorch.org/whl/'
# PyTorch CUDA wheel tags and the CUDA version the driver must support for them,
# newest first. A driver supports every CUDA version up to the one nvidia-smi reports.
$script:CudaTags = @(
    @{ Tag = 'cu132'; Cuda = [version]'13.2' },
    @{ Tag = 'cu130'; Cuda = [version]'13.0' },
    @{ Tag = 'cu129'; Cuda = [version]'12.9' },
    @{ Tag = 'cu128'; Cuda = [version]'12.8' },
    @{ Tag = 'cu126'; Cuda = [version]'12.6' },
    @{ Tag = 'cu124'; Cuda = [version]'12.4' }
)
# How many CUDA indexes to try before giving up (newest supported first).
$script:MaxCudaAttempts = 4
$script:MinFreeGB = 15

# ---------------------------------------------------------------------------
# State shared by all steps
# ---------------------------------------------------------------------------
$script:State = @{
    LogFile          = $null
    Warnings         = New-Object System.Collections.Generic.List[string]
    Problems         = New-Object System.Collections.Generic.List[string]
    Created          = New-Object System.Collections.Generic.List[string]
    Steps            = New-Object System.Collections.Generic.List[object]
    Nvidia           = $null
    TorchTag         = $null
    TorchVersion     = $null
    TorchAudioVersion = $null
    TorchMeta        = $null
    CudaAvailable    = $false
    GpuName          = $null
    Device           = 'cpu'
    FfmpegSource     = $null
    Ffmpeg           = $null
    Ffprobe          = $null
    FirewallRule     = $null
    FirewallOk       = $false
    Shortcuts        = New-Object System.Collections.Generic.List[string]
    NetworkChange    = $null
    PublicProfiles   = @()
    ModelsOk         = $null
    HfModels         = @()
    SelfCheck        = $null
    SelfCheckOk      = $null
    LanUrls          = @()
    OldConsoleEncoding = $null
}

# ---------------------------------------------------------------------------
# Generic helpers (no Windows-only calls here, so they can be tested anywhere)
# ---------------------------------------------------------------------------
function Get-AppPaths {
    param([Parameter(Mandatory = $true)][string]$AppDir)
    $a = [System.IO.Path]::GetFullPath($AppDir).TrimEnd('\', '/')
    # On Windows the Replace is a no-op; it keeps the tests on Linux consistent.
    $j = { param($rel) [System.IO.Path]::Combine($a, $rel.Replace('\', [System.IO.Path]::DirectorySeparatorChar)) }
    $p = @{
        AppDir       = $a
        Runtime      = (& $j 'runtime')
        UvDir        = (& $j 'runtime\uv')
        UvExe        = (& $j 'runtime\uv\uv.exe')
        PythonDir    = (& $j 'runtime\python')
        PythonBin    = (& $j 'runtime\python\bin')
        UvCache      = (& $j 'runtime\uv-cache')
        UvTools      = (& $j 'runtime\uv-tools')
        UvToolsBin   = (& $j 'runtime\uv-tools\bin')
        FfmpegDir    = (& $j 'runtime\ffmpeg')
        FfmpegBin    = (& $j 'runtime\ffmpeg\bin')
        Constraints  = (& $j 'runtime\torch-constraints.txt')
        Venv         = (& $j '.venv')
        VenvPython   = (& $j '.venv\Scripts\python.exe')
        Models       = (& $j 'models')
        HfHome       = (& $j 'models\hf')
        TorchHome    = (& $j 'models\torch')
        XdgCache     = (& $j 'models\cache')
        Data         = (& $j 'data')
        Logs         = (& $j 'data\logs')
        Tmp          = (& $j 'data\tmp')
        Cache        = (& $j 'data\cache')
        Requirements = (& $j 'requirements.txt')
        Package      = (& $j 'voicefilter')
        Config       = (& $j 'config.json')
        Manifest     = (& $j 'install-manifest.json')
        StartBat     = (& $j 'start.bat')
        StartPs1     = (& $j 'start.ps1')
    }
    return $p
}

function Write-Log {
    param(
        [Parameter(Position = 0)][AllowEmptyString()][string]$Message = '',
        [ValidateSet('Info', 'Ok', 'Warn', 'Error', 'Step', 'Detail', 'FileOnly')]
        [string]$Level = 'Info'
    )
    $stamp = (Get-Date).ToString('HH:mm:ss')
    $prefix = ''
    $color = $null
    switch ($Level) {
        'Ok'     { $prefix = '[ OK ] '; $color = 'Green' }
        'Warn'   { $prefix = '[WARN] '; $color = 'Yellow' }
        'Error'  { $prefix = '[FAIL] '; $color = 'Red' }
        'Step'   { $prefix = '';        $color = 'Cyan' }
        'Detail' { $prefix = '       '; $color = 'DarkGray' }
        default  { $prefix = '' }
    }
    if ($Level -ne 'FileOnly') {
        if ($null -ne $color) { Write-Host ($prefix + $Message) -ForegroundColor $color }
        else { Write-Host ($prefix + $Message) }
    }
    if ($Level -eq 'Warn') { $script:State.Warnings.Add($Message) }
    if ($script:State.LogFile) {
        try {
            $enc = New-Object System.Text.UTF8Encoding($false)
            [System.IO.File]::AppendAllText($script:State.LogFile, "$stamp $prefix$Message`r`n", $enc)
        }
        catch {
            # Logging must never break the installation.
            $null = $_
        }
    }
}

function Write-Section {
    param([string]$Title)
    Write-Log ''
    Write-Log ('=== ' + $Title + ' ===') -Level Step
}

function Write-Utf8File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text,
        [switch]$Bom
    )
    $enc = New-Object System.Text.UTF8Encoding($Bom.IsPresent)
    [System.IO.File]::WriteAllText($Path, $Text, $enc)
}

function Write-AsciiFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text
    )
    if (-not (Test-IsAscii $Text)) { throw "Internal error: non-ASCII text for $Path" }
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.Encoding]::ASCII)
}

function Test-IsAscii {
    param([AllowEmptyString()][string]$Text)
    foreach ($ch in $Text.ToCharArray()) {
        if ([int]$ch -gt 126) { return $false }
        if ([int]$ch -lt 32 -and $ch -ne "`r" -and $ch -ne "`n" -and $ch -ne "`t") { return $false }
    }
    return $true
}

function ConvertTo-PsLiteral {
    # Single-quoted PowerShell string literal. PowerShell also treats the typographic
    # quotes U+2018..U+201B as single quotes, so they are doubled as well.
    param([AllowEmptyString()][string]$Value)
    $quoteChars = @([char]0x27, [char]0x2018, [char]0x2019, [char]0x201A, [char]0x201B)
    $sb = New-Object System.Text.StringBuilder
    $null = $sb.Append("'")
    foreach ($ch in $Value.ToCharArray()) {
        if ($quoteChars -contains $ch) { $null = $sb.Append($ch) }
        $null = $sb.Append($ch)
    }
    $null = $sb.Append("'")
    return $sb.ToString()
}

function ConvertTo-CmdArgument {
    # Quote one argument for CreateProcess / MSVCRT command line parsing.
    param([AllowEmptyString()][string]$Value)
    if ($Value -ne '' -and $Value -notmatch '[\s"]') { return $Value }
    $sb = New-Object System.Text.StringBuilder
    $null = $sb.Append('"')
    $backslashes = 0
    foreach ($ch in $Value.ToCharArray()) {
        if ($ch -eq '\') { $backslashes++; continue }
        if ($ch -eq '"') {
            $null = $sb.Append([char]'\', (2 * $backslashes + 1))
            $null = $sb.Append('"')
        }
        else {
            if ($backslashes -gt 0) { $null = $sb.Append([char]'\', $backslashes) }
            $null = $sb.Append($ch)
        }
        $backslashes = 0
    }
    if ($backslashes -gt 0) { $null = $sb.Append([char]'\', (2 * $backslashes)) }
    $null = $sb.Append('"')
    return $sb.ToString()
}

function Test-PathInside {
    # True when $Path is $Root itself or below it (case-insensitive, Windows semantics).
    param([string]$Path, [string]$Root)
    if ([string]::IsNullOrEmpty($Path) -or [string]::IsNullOrEmpty($Root)) { return $false }
    try {
        $p = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
        $r = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    }
    catch { return $false }
    if ($p.Equals($r, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    $sep = [System.IO.Path]::DirectorySeparatorChar
    return ($p.StartsWith($r + $sep, [System.StringComparison]::OrdinalIgnoreCase) -or
            $p.StartsWith($r + '\', [System.StringComparison]::OrdinalIgnoreCase))
}

function Get-RelativeAppPath {
    # 'C:\App\runtime\ffmpeg\bin\ffmpeg.exe' -> 'runtime\ffmpeg\bin\ffmpeg.exe' (or $null if outside)
    param([string]$Path, [string]$AppDir)
    if (-not (Test-PathInside -Path $Path -Root $AppDir)) { return $null }
    $p = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $r = [System.IO.Path]::GetFullPath($AppDir).TrimEnd('\', '/')
    if ($p.Length -le $r.Length) { return '' }
    return $p.Substring($r.Length + 1)
}

function New-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        $null = New-Item -ItemType Directory -Path $Path -Force
    }
}

function Remove-PathIfExists {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Add-Created {
    param([string]$Path)
    if (-not $script:State.Created.Contains($Path)) { $script:State.Created.Add($Path) }
}

function Read-Choice {
    # Ask a yes/no question. In -Yes mode (or without a console) the default is returned.
    param([string]$Prompt, [bool]$Default)
    if ($Yes) {
        $d = 'no'
        if ($Default) { $d = 'yes' }
        Write-Log "$Prompt -> $d (non-interactive default)" -Level Detail
        return $Default
    }
    $suffix = '[y/N]'
    if ($Default) { $suffix = '[Y/n]' }
    while ($true) {
        try { $answer = Read-Host "$Prompt $suffix" }
        catch { return $Default }
        if ($null -eq $answer) { return $Default }
        $a = $answer.Trim().ToLowerInvariant()
        Write-Log "$Prompt $suffix -> '$a'" -Level FileOnly
        if ($a -eq '') { return $Default }
        if (@('y', 'yes', 'j', 'ja') -contains $a) { return $true }
        if (@('n', 'no', 'nein') -contains $a) { return $false }
        Write-Host 'Please answer y or n.'
    }
}

function Invoke-Native {
    <#
      Runs a native program and returns @{ExitCode; Output}.
      Default: output is captured line by line, shown and logged.
      -Live: the program writes straight to this console (progress bars stay
      visible); only the command and its exit code are logged.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [switch]$Live,
        [switch]$Quiet,
        [string]$WorkingDirectory = ''
    )
    $display = (@($FilePath) + $Arguments | ForEach-Object { ConvertTo-CmdArgument $_ }) -join ' '
    Write-Log "> $display" -Level FileOnly
    $lines = New-Object System.Collections.Generic.List[string]
    $code = $null
    if ($Live) {
        $argLine = ($Arguments | ForEach-Object { ConvertTo-CmdArgument $_ }) -join ' '
        $spArgs = @{ FilePath = $FilePath; NoNewWindow = $true; Wait = $true; PassThru = $true }
        if ($argLine -ne '') { $spArgs['ArgumentList'] = $argLine }
        if ($WorkingDirectory -ne '') { $spArgs['WorkingDirectory'] = $WorkingDirectory }
        $proc = Start-Process @spArgs
        $code = $proc.ExitCode
        Write-Log "  (output shown on screen only) exit code $code" -Level FileOnly
    }
    else {
        $oldEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $pushed = $false
        try {
            if ($WorkingDirectory -ne '') { Push-Location -LiteralPath $WorkingDirectory; $pushed = $true }
            & $FilePath @Arguments 2>&1 | ForEach-Object {
                $line = "$_"
                $lines.Add($line)
                if ($Quiet) { Write-Log "  $line" -Level FileOnly } else { Write-Log "  $line" -Level Detail }
            }
            $code = $LASTEXITCODE
        }
        finally {
            if ($pushed) { Pop-Location }
            $ErrorActionPreference = $oldEap
        }
        Write-Log "  exit code $code" -Level FileOnly
    }
    if ($null -eq $code) { $code = -1 }
    return [pscustomobject]@{ ExitCode = [int]$code; Output = $lines.ToArray() }
}

function Get-JsonFromOutput {
    # Finds the last line of program output that parses as a JSON object.
    param([string[]]$Lines)
    $text = ($Lines -join "`n")
    $start = $text.IndexOf('{')
    $end = $text.LastIndexOf('}')
    if ($start -ge 0 -and $end -gt $start) {
        try { return ($text.Substring($start, $end - $start + 1) | ConvertFrom-Json) } catch { $null = $_ }
    }
    for ($i = $Lines.Count - 1; $i -ge 0; $i--) {
        $l = $Lines[$i].Trim()
        if ($l.StartsWith('{')) {
            try { return ($l | ConvertFrom-Json) } catch { $null = $_ }
        }
    }
    return $null
}

function Get-ObjectValue {
    # StrictMode-safe property lookup on PSCustomObject / hashtable; first name that exists wins.
    param($Object, [string[]]$Names, $Default = $null)
    if ($null -eq $Object) { return $Default }
    foreach ($n in $Names) {
        if ($Object -is [System.Collections.IDictionary]) {
            if ($Object.Contains($n)) { return $Object[$n] }
        }
        else {
            $prop = $Object.PSObject.Properties[$n]
            if ($null -ne $prop) { return $prop.Value }
        }
    }
    return $Default
}

function ConvertTo-Hashtable {
    # ConvertFrom-Json -AsHashtable does not exist in PowerShell 5.1.
    param($InputObject)
    if ($null -eq $InputObject) { return $null }
    if ($InputObject -is [System.Collections.IDictionary]) {
        $h = [ordered]@{}
        foreach ($k in $InputObject.Keys) { $h[$k] = ConvertTo-Hashtable $InputObject[$k] }
        return $h
    }
    if ($InputObject -is [System.Management.Automation.PSCustomObject]) {
        $h = [ordered]@{}
        foreach ($prop in $InputObject.PSObject.Properties) { $h[$prop.Name] = ConvertTo-Hashtable $prop.Value }
        return $h
    }
    if ($InputObject -is [System.Collections.IEnumerable] -and -not ($InputObject -is [string])) {
        $list = New-Object System.Collections.ArrayList
        foreach ($item in $InputObject) { $null = $list.Add((ConvertTo-Hashtable $item)) }
        return , $list.ToArray()
    }
    return $InputObject
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try {
        $text = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
        if ($text.Trim() -eq '') { return $null }
        return (ConvertTo-Hashtable ($text | ConvertFrom-Json))
    }
    catch {
        Write-Log "Could not read $Path ($($_.Exception.Message)); it will be rewritten." -Level Warn
        return $null
    }
}

function ConvertFrom-NvidiaSmi {
    # Parses the header of plain `nvidia-smi` output.
    param([string[]]$Lines)
    $text = ($Lines -join "`n")
    $info = [ordered]@{ Present = $false; Driver = $null; Cuda = $null; Gpus = @() }
    $m = [regex]::Match($text, 'Driver Version:\s*([0-9][0-9.]*)')
    if ($m.Success) { $info.Driver = $m.Groups[1].Value; $info.Present = $true }
    $m = [regex]::Match($text, 'CUDA(?: Driver)? Version:\s*([0-9]+)\.([0-9]+)')
    if ($m.Success) { $info.Cuda = [version]("{0}.{1}" -f $m.Groups[1].Value, $m.Groups[2].Value) }
    return $info
}

function Get-CudaFromDriverVersion {
    # Fallback when nvidia-smi does not print a CUDA version: minimum Windows driver
    # versions from NVIDIA's CUDA toolkit release notes.
    param([string]$Driver)
    if (-not $Driver) { return $null }
    try { $d = [version]$Driver } catch { return $null }
    if ($d -ge [version]'580.0') { return [version]'13.0' }
    if ($d -ge [version]'576.02') { return [version]'12.9' }
    if ($d -ge [version]'570.65') { return [version]'12.8' }
    if ($d -ge [version]'560.76') { return [version]'12.6' }
    if ($d -ge [version]'551.61') { return [version]'12.4' }
    return $null
}

function Get-TorchCandidateTags {
    # Ordered list of PyTorch index tags to try, e.g. @('cu128','cu126','cu124') or @('cpu').
    param($CudaVersion, [bool]$ForceCpu, [string]$Override)
    if ($Override) { return @($Override.Trim().ToLowerInvariant()) }
    if ($ForceCpu -or $null -eq $CudaVersion) { return @('cpu') }
    $tags = @($script:CudaTags | Where-Object { $_.Cuda -le $CudaVersion } | ForEach-Object { $_.Tag })
    if ($tags.Count -eq 0) { return @('cpu') }
    return @($tags | Select-Object -First $script:MaxCudaAttempts)
}

function Get-PresetList {
    param([string]$Value)
    $list = @($Value -split '[,; ]+' | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ -ne '' })
    foreach ($p in $list) {
        if ($p -notmatch '^[a-z0-9][a-z0-9_-]*$') { throw "Invalid preset name '$p'." }
    }
    if ($list.Count -eq 0) { throw 'No preset given (use for example -Presets fast,balanced,best).' }
    return $list
}

function Get-AppEnvironment {
    <#
      Environment for the app, as ordered list of @{Name; Rel; Value}.
      Rel = path relative to the app folder (used by start.bat as %VF_APP_DIR%\<Rel>),
      Value = literal value. Exactly one of them is set.
    #>
    param([int]$PortNumber, [string]$Device, [string]$FfmpegPath, [string]$FfprobePath, [string]$AppDir)
    $envList = New-Object System.Collections.Generic.List[object]
    $add = {
        param($n, $rel, $val)
        $envList.Add([pscustomobject]@{ Name = $n; Rel = $rel; Value = $val })
    }
    & $add 'VF_DATA_DIR' 'data' $null
    & $add 'VF_MODELS_DIR' 'models' $null
    & $add 'HF_HOME' 'models\hf' $null
    & $add 'TORCH_HOME' 'models\torch' $null
    & $add 'XDG_CACHE_HOME' 'models\cache' $null
    & $add 'NUMBA_CACHE_DIR' 'data\cache\numba' $null
    & $add 'MPLCONFIGDIR' 'data\cache\mpl' $null
    & $add 'CUDA_CACHE_PATH' 'data\cache\cuda' $null
    & $add 'TORCH_EXTENSIONS_DIR' 'data\cache\torch_extensions' $null
    & $add 'TEMP' 'data\tmp' $null
    & $add 'TMP' 'data\tmp' $null
    & $add 'VF_PORT' $null ([string]$PortNumber)
    & $add 'VF_DEVICE' $null $Device
    foreach ($pair in @(@('VF_FFMPEG', $FfmpegPath), @('VF_FFPROBE', $FfprobePath))) {
        if ($pair[1]) {
            $rel = Get-RelativeAppPath -Path $pair[1] -AppDir $AppDir
            if ($null -ne $rel -and $rel -ne '') { & $add $pair[0] $rel $null }
            else { & $add $pair[0] $null $pair[1] }
        }
    }
    & $add 'PYTHONUTF8' $null '1'
    & $add 'PYTHONIOENCODING' $null 'utf-8'
    & $add 'PYTHONNOUSERSITE' $null '1'
    & $add 'HF_HUB_DISABLE_TELEMETRY' $null '1'
    & $add 'HF_HUB_DISABLE_SYMLINKS_WARNING' $null '1'
    & $add 'TOKENIZERS_PARALLELISM' $null 'false'
    # Hook for gated Hugging Face models (not needed today): the backend would read
    # HF_TOKEN. If it is ever required, ask for it here and store it in data\hf_token,
    # never in start.bat.
    # & $add 'HF_TOKEN' $null '<token>'
    return , $envList.ToArray()
}

function Set-ProcessEnvironment {
    # Applies the app environment to this PowerShell process (inherited by child processes).
    param([object[]]$EnvList, [string]$AppDir)
    [Environment]::SetEnvironmentVariable('VF_APP_DIR', $AppDir, 'Process')
    foreach ($e in $EnvList) {
        $v = $e.Value
        if ($null -ne $e.Rel) { $v = [System.IO.Path]::Combine($AppDir, $e.Rel.Replace('\', [System.IO.Path]::DirectorySeparatorChar)) }
        [Environment]::SetEnvironmentVariable($e.Name, $v, 'Process')
    }
    foreach ($n in @('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'VIRTUAL_ENV', 'CONDA_PREFIX')) {
        [Environment]::SetEnvironmentVariable($n, $null, 'Process')
    }
}

function New-StartBatText {
    param([object[]]$EnvList, [int]$PortNumber, [string]$FfmpegBinRel)
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add('@echo off')
    $L.Add('rem Voice Filter launcher. Generated by install.ps1 - changes are overwritten.')
    $L.Add('rem Re-run install.bat to change the port or other settings.')
    $L.Add('rem Usage: start.bat [--no-browser]')
    $L.Add('setlocal')
    $L.Add('title Voice Filter')
    $L.Add('set "VF_APP_DIR=%~dp0"')
    $L.Add('if "%VF_APP_DIR:~-1%"=="\" set "VF_APP_DIR=%VF_APP_DIR:~0,-1%"')
    foreach ($e in $EnvList) {
        if ($null -ne $e.Rel) {
            $L.Add(('set "{0}=%VF_APP_DIR%\{1}"' -f $e.Name, $e.Rel))
        }
        elseif (Test-IsAscii $e.Value) {
            $L.Add(('set "{0}={1}"' -f $e.Name, ($e.Value -replace '%', '%%')))
        }
        else {
            $L.Add(('rem {0}: path contains non-ASCII characters; it is read from config.json instead.' -f $e.Name))
        }
    }
    foreach ($n in @('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'VIRTUAL_ENV', 'CONDA_PREFIX')) {
        $L.Add(('set "{0}="' -f $n))
    }
    if ($FfmpegBinRel) {
        $L.Add(('set "PATH=%VF_APP_DIR%\{0};%PATH%"' -f $FfmpegBinRel))
    }
    $L.Add('set "VF_PYTHON=%VF_APP_DIR%\.venv\Scripts\python.exe"')
    $L.Add('if not exist "%VF_PYTHON%" goto :notinstalled')
    foreach ($d in @('data\tmp', 'data\cache', 'data\logs')) {
        $L.Add(('if not exist "%VF_APP_DIR%\{0}" mkdir "%VF_APP_DIR%\{0}"' -f $d))
    }
    $L.Add('cd /d "%VF_APP_DIR%"')
    $L.Add('if /i "%~1"=="--no-browser" goto :run')
    $poll = ("`$u='http://localhost:{0}/'; `$t='http://127.0.0.1:{0}/api/info'; " +
             "for(`$i=0;`$i -lt 180;`$i++){{try{{`$null=Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri `$t; Start-Process `$u; break}}catch{{Start-Sleep -Milliseconds 700}}}}") -f $PortNumber
    $L.Add(('start "" /min powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "{0}"' -f $poll))
    $L.Add(':run')
    $L.Add('echo Starting Voice Filter on port %VF_PORT% ... (close this window or press Ctrl+C to stop)')
    $L.Add('"%VF_PYTHON%" -m voicefilter')
    $L.Add('set "VF_EXIT=%ERRORLEVEL%"')
    $L.Add('if not "%VF_EXIT%"=="0" goto :failed')
    $L.Add('exit /b 0')
    $L.Add(':failed')
    $L.Add('echo.')
    $L.Add('echo Voice Filter stopped with error code %VF_EXIT%. See the messages above.')
    $L.Add('echo Logs: "%VF_APP_DIR%\data\logs"')
    $L.Add('pause')
    $L.Add('exit /b %VF_EXIT%')
    $L.Add(':notinstalled')
    $L.Add('echo Voice Filter is not installed completely: .venv\Scripts\python.exe is missing.')
    $L.Add('echo Run install.bat in this folder first.')
    $L.Add('pause')
    $L.Add('exit /b 1')
    return (($L.ToArray() -join "`r`n") + "`r`n")
}

function New-StartPs1Text {
    param([object[]]$EnvList, [int]$PortNumber, [string]$FfmpegBinRel)
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add('# Voice Filter launcher. Generated by install.ps1 - changes are overwritten.')
    $L.Add('# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1 [-NoBrowser]')
    $L.Add('param([switch]$NoBrowser)')
    $L.Add('Set-StrictMode -Version Latest')
    $L.Add("`$ErrorActionPreference = 'Stop'")
    $L.Add('$AppDir = $PSScriptRoot')
    $L.Add('function Set-VfEnv([string]$Name, [string]$Value) { [Environment]::SetEnvironmentVariable($Name, $Value, ''Process'') }')
    $L.Add('Set-VfEnv ''VF_APP_DIR'' $AppDir')
    foreach ($e in $EnvList) {
        if ($null -ne $e.Rel) {
            $L.Add(('Set-VfEnv {0} (Join-Path $AppDir {1})' -f (ConvertTo-PsLiteral $e.Name), (ConvertTo-PsLiteral $e.Rel)))
        }
        else {
            $L.Add(('Set-VfEnv {0} {1}' -f (ConvertTo-PsLiteral $e.Name), (ConvertTo-PsLiteral $e.Value)))
        }
    }
    foreach ($n in @('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'VIRTUAL_ENV', 'CONDA_PREFIX')) {
        $L.Add(('Set-VfEnv {0} $null' -f (ConvertTo-PsLiteral $n)))
    }
    if ($FfmpegBinRel) {
        $L.Add(('$env:PATH = (Join-Path $AppDir {0}) + '';'' + $env:PATH' -f (ConvertTo-PsLiteral $FfmpegBinRel)))
    }
    $L.Add('$py = Join-Path $AppDir ''.venv\Scripts\python.exe''')
    $L.Add('if (-not (Test-Path -LiteralPath $py)) { Write-Host ''Voice Filter is not installed completely (.venv is missing). Run install.bat first.'' -ForegroundColor Red; exit 1 }')
    $L.Add('foreach ($d in @(''data\tmp'', ''data\cache'', ''data\logs'')) { $null = New-Item -ItemType Directory -Force -Path (Join-Path $AppDir $d) }')
    $L.Add('Set-Location -LiteralPath $AppDir')
    $L.Add('if (-not $NoBrowser) {')
    $poll = ("`$u='http://localhost:{0}/'; `$t='http://127.0.0.1:{0}/api/info'; " +
             "for(`$i=0;`$i -lt 180;`$i++){{try{{`$null=Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri `$t; Start-Process `$u; break}}catch{{Start-Sleep -Milliseconds 700}}}}") -f $PortNumber
    $encodedPoll = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($poll))
    $L.Add('    # Opens the browser as soon as the server answers (hidden helper window).')
    $L.Add(('    Start-Process -FilePath ''powershell.exe'' -WindowStyle Hidden -ArgumentList ''-NoProfile -ExecutionPolicy Bypass -EncodedCommand {0}''' -f $encodedPoll))
    $L.Add('}')
    $L.Add(('Write-Host ''Starting Voice Filter on port {0} ... (press Ctrl+C to stop)''' -f $PortNumber))
    $L.Add('& $py -m voicefilter')
    $L.Add('exit $LASTEXITCODE')
    return (($L.ToArray() -join "`r`n") + "`r`n")
}

function Merge-Manifest {
    # Merges the facts of this run into an existing manifest (hashtable or $null).
    param($Old, [System.Collections.IDictionary]$New)
    $m = [ordered]@{}
    if ($null -ne $Old) { foreach ($k in $Old.Keys) { $m[$k] = $Old[$k] } }
    foreach ($k in $New.Keys) { $m[$k] = $New[$k] }
    # first install time survives re-runs
    $first = Get-ObjectValue $Old @('first_installed_at') $null
    if ($null -eq $first) { $first = Get-ObjectValue $Old @('installed_at') $null }
    if ($null -eq $first) { $first = $New['installed_at'] }
    $m['first_installed_at'] = $first
    # union of created paths
    $created = New-Object System.Collections.Generic.List[string]
    foreach ($src in @((Get-ObjectValue $Old @('created') @()), $New['created'])) {
        foreach ($c in @($src)) {
            if ($null -ne $c -and -not $created.Contains([string]$c)) { $created.Add([string]$c) }
        }
    }
    $m['created'] = $created.ToArray()
    # union of shortcuts
    $sc = New-Object System.Collections.Generic.List[string]
    foreach ($src in @((Get-ObjectValue $Old @('shortcuts') @()), $New['shortcuts'])) {
        foreach ($c in @($src)) {
            if ($null -ne $c -and -not $sc.Contains([string]$c)) { $sc.Add([string]$c) }
        }
    }
    $m['shortcuts'] = $sc.ToArray()
    # keep the record of a network profile change made by an earlier run
    if ($null -eq $New['network_profile_changed']) {
        $m['network_profile_changed'] = Get-ObjectValue $Old @('network_profile_changed') $null
    }
    # remember every firewall rule name we ever created (port may have changed)
    $rules = New-Object System.Collections.Generic.List[string]
    foreach ($src in @((Get-ObjectValue $Old @('firewall_rules_created') @()), $New['firewall_rules_created'])) {
        foreach ($c in @($src)) {
            if ($null -ne $c -and -not $rules.Contains([string]$c)) { $rules.Add([string]$c) }
        }
    }
    $m['firewall_rules_created'] = $rules.ToArray()
    return $m
}

function New-ElevatedScriptText {
    <#
      Script run in ONE elevated PowerShell: firewall rule (remove old + create) and,
      optionally, switching network profiles to Private. Writes a JSON result file.
    #>
    param(
        [string]$RuleName,
        [string[]]$RemovePatterns,
        [int]$PortNumber,
        [bool]$CreateRule,
        [object[]]$ProfileChanges,
        [string]$ResultFile
    )
    $patterns = ($RemovePatterns | ForEach-Object { ConvertTo-PsLiteral $_ }) -join ', '
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add("`$ErrorActionPreference = 'Stop'")
    $L.Add('$res = [ordered]@{ rule_ok = $null; rule_error = $null; profiles = @() }')
    $L.Add('try {')
    if ($CreateRule) {
        # Remove-and-recreate, so a changed port or edited rule is replaced.
        $L.Add("    foreach (`$pat in @($patterns)) {")
        $L.Add('        Get-NetFirewallRule -DisplayName $pat -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction Stop')
        $L.Add('    }')
        $L.Add(('    $null = New-NetFirewallRule -DisplayName {0} -Description {1} -Direction Inbound -Protocol TCP -LocalPort {2} -Action Allow -Profile Private -Enabled True' -f
                (ConvertTo-PsLiteral $RuleName), (ConvertTo-PsLiteral 'Allows phones on the home network to reach the Voice Filter web app. Created by install.ps1, removed by uninstall.ps1.'), $PortNumber))
    }
    $L.Add('    $res.rule_ok = $true')
    $L.Add('} catch { $res.rule_ok = $false; $res.rule_error = $_.Exception.Message }')
    foreach ($pc in @($ProfileChanges)) {
        if ($null -eq $pc) { continue }
        $L.Add('try {')
        $L.Add(('    Set-NetConnectionProfile -InterfaceIndex {0} -NetworkCategory {1} -ErrorAction Stop' -f [int]$pc.InterfaceIndex, $pc.To))
        $L.Add(('    $res.profiles += @{{ index = {0}; ok = $true; error = $null }}' -f [int]$pc.InterfaceIndex))
        $L.Add(('}} catch {{ $res.profiles += @{{ index = {0}; ok = $false; error = $_.Exception.Message }} }}' -f [int]$pc.InterfaceIndex))
    }
    $L.Add(('[System.IO.File]::WriteAllText({0}, ($res | ConvertTo-Json -Depth 4), (New-Object System.Text.UTF8Encoding($false)))' -f (ConvertTo-PsLiteral $ResultFile)))
    return ($L.ToArray() -join "`r`n")
}

# ---------------------------------------------------------------------------
# Windows-specific helpers
# ---------------------------------------------------------------------------
function Get-WindowsVersion {
    return [Environment]::OSVersion.Version
}

function Get-ShortcutPaths {
    # Desktop and Start Menu > Programs of the current user (OneDrive-redirected Desktop included).
    return @(
        (Join-Path ([Environment]::GetFolderPath('Desktop')) "$($script:AppName).lnk"),
        (Join-Path ([Environment]::GetFolderPath('Programs')) "$($script:AppName).lnk")
    )
}

function Test-IsAdmin {
    try {
        $id = [Security.Principal.WindowsIdentity]::GetCurrent()
        $pr = New-Object Security.Principal.WindowsPrincipal($id)
        return $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    catch { return $false }
}

function Invoke-Elevated {
    # Runs a script text in one elevated PowerShell (UAC prompt). Returns the parsed result or $null.
    param([string]$ScriptText, [string]$ResultFile)
    Remove-PathIfExists $ResultFile
    if (Test-IsAdmin) {
        Write-Log 'Already running as administrator; applying directly.' -Level Detail
        $sb = [scriptblock]::Create($ScriptText)
        & $sb
    }
    else {
        $psExe = (Get-Process -Id $PID).Path
        if (-not $psExe) { $psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe' }
        $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($ScriptText))
        Write-Log 'Windows will ask for administrator permission (UAC) for this step only.'
        try {
            $proc = Start-Process -FilePath $psExe -Verb RunAs -Wait -PassThru -WindowStyle Hidden `
                -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encoded)
            Write-Log "Elevated helper exit code: $($proc.ExitCode)" -Level FileOnly
        }
        catch {
            Write-Log "Administrator permission was not granted ($($_.Exception.Message))." -Level Warn
            return $null
        }
    }
    if (Test-Path -LiteralPath $ResultFile) {
        try { return ([System.IO.File]::ReadAllText($ResultFile) | ConvertFrom-Json) }
        finally { Remove-PathIfExists $ResultFile }
    }
    throw 'The administrator helper did not report a result (see the log for its exit code).'
}

function Get-AppProcesses {
    # Processes whose executable lives inside the app folder (python from .venv/runtime, uv, ffmpeg).
    param([string]$AppDir)
    $result = @()
    try {
        $procs = Get-CimInstance -ClassName Win32_Process -ErrorAction Stop
        foreach ($p in $procs) {
            $exe = $p.ExecutablePath
            if ($exe -and (Test-PathInside -Path $exe -Root $AppDir) -and $p.ProcessId -ne $PID) {
                $result += [pscustomobject]@{ Id = [int]$p.ProcessId; Path = $exe; Name = $p.Name }
            }
        }
    }
    catch {
        foreach ($p in (Get-Process -ErrorAction SilentlyContinue)) {
            $exe = $null
            try { $exe = $p.Path } catch { $exe = $null }
            if ($exe -and (Test-PathInside -Path $exe -Root $AppDir) -and $p.Id -ne $PID) {
                $result += [pscustomobject]@{ Id = [int]$p.Id; Path = $exe; Name = $p.ProcessName }
            }
        }
    }
    return , $result
}

function Get-PortOwner {
    param([int]$PortNumber)
    try {
        $c = Get-NetTCPConnection -LocalPort $PortNumber -State Listen -ErrorAction Stop | Select-Object -First 1
        if ($null -eq $c) { return $null }
        $path = $null
        try { $path = (Get-Process -Id $c.OwningProcess -ErrorAction Stop).Path } catch { $path = $null }
        return [pscustomobject]@{ Id = [int]$c.OwningProcess; Path = $path }
    }
    catch { return $null }
}

function Invoke-Download {
    param([string]$Url, [string]$OutFile, [int]$Attempts = 3)
    $part = $OutFile + '.part'
    $lastError = $null
    for ($i = 1; $i -le $Attempts; $i++) {
        try {
            Remove-PathIfExists $part
            Write-Log "Downloading $Url (attempt $i of $Attempts) ..."
            $oldProgress = $ProgressPreference
            $ProgressPreference = 'SilentlyContinue'
            try { Invoke-WebRequest -Uri $Url -OutFile $part -UseBasicParsing }
            finally { $ProgressPreference = $oldProgress }
            $len = (Get-Item -LiteralPath $part).Length
            if ($len -lt 1024) { throw "download is only $len bytes" }
            Remove-PathIfExists $OutFile
            Move-Item -LiteralPath $part -Destination $OutFile -Force
            Write-Log ("Downloaded {0:N1} MB" -f ($len / 1MB)) -Level Detail
            return
        }
        catch {
            $lastError = $_.Exception.Message
            Write-Log "Download attempt $i failed: $lastError" -Level Detail
            Start-Sleep -Seconds (2 * $i)
        }
    }
    # Last resort: curl.exe ships with Windows 10 1803+ and handles some proxies/TLS setups better.
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    if (Test-Path -LiteralPath $curl) {
        Write-Log 'Trying again with curl.exe ...' -Level Detail
        Remove-PathIfExists $part
        $r = Invoke-Native -FilePath $curl -Arguments @('-L', '--fail', '--retry', '3', '-sS', '-o', $part, $Url) -Quiet
        if ($r.ExitCode -eq 0 -and (Test-Path -LiteralPath $part) -and (Get-Item -LiteralPath $part).Length -ge 1024) {
            Remove-PathIfExists $OutFile
            Move-Item -LiteralPath $part -Destination $OutFile -Force
            return
        }
        Remove-PathIfExists $part
    }
    throw "Could not download $Url ($lastError). Check the internet connection / proxy and re-run install.bat."
}

function Expand-ZipFile {
    param([string]$ZipFile, [string]$Destination)
    Remove-PathIfExists $Destination
    New-Directory $Destination
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction Stop
        [System.IO.Compression.ZipFile]::ExtractToDirectory($ZipFile, $Destination)
    }
    catch {
        Write-Log "Fast unzip failed ($($_.Exception.Message)); using Expand-Archive." -Level Detail
        Expand-Archive -LiteralPath $ZipFile -DestinationPath $Destination -Force
    }
}

function Get-LanAddresses {
    $ips = @()
    try {
        $profiled = @(Get-NetConnectionProfile -ErrorAction Stop | ForEach-Object { $_.InterfaceIndex })
        $addrs = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop | Where-Object {
            $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and
            $_.InterfaceAlias -notlike 'vEthernet*' -and $_.InterfaceAlias -notlike '*Loopback*'
        }
        $preferred = @($addrs | Where-Object { $profiled -contains $_.InterfaceIndex })
        if ($preferred.Count -eq 0) { $preferred = @($addrs) }
        $ips = @($preferred | ForEach-Object { $_.IPAddress })
    }
    catch {
        try {
            $ips = @([System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
                Where-Object { $_.AddressFamily -eq 'InterNetwork' -and $_.ToString() -notlike '127.*' -and $_.ToString() -notlike '169.254.*' } |
                ForEach-Object { $_.ToString() })
        }
        catch { $ips = @() }
    }
    return , @($ips | Select-Object -Unique)
}

function New-Shortcut {
    param([string]$LinkPath, [string]$Target, [string]$WorkingDirectory, [string]$Description, [string]$Icon)
    $dir = Split-Path -Parent $LinkPath
    New-Directory $dir
    $shell = New-Object -ComObject WScript.Shell
    try {
        $lnk = $shell.CreateShortcut($LinkPath)
        $lnk.TargetPath = $Target
        $lnk.WorkingDirectory = $WorkingDirectory
        $lnk.Description = $Description
        $lnk.WindowStyle = 1
        if ($Icon) { $lnk.IconLocation = $Icon }
        $lnk.Save()
    }
    finally {
        $null = [System.Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
    }
}

function Get-TorchInfo {
    <#
      Installed torch/torchaudio: Torch/Audio = distribution versions (what pip/uv see, used
      for the constraints file); TorchRt/AudioRt = __version__ at runtime, which always
      carries the build tag (+cu128, +cpu), also for wheels whose metadata has no tag.
    #>
    param([string]$Python)
    if (-not (Test-Path -LiteralPath $Python)) { return $null }
    $code = 'import json; from importlib import metadata as m; ' +
            'd = {(x.metadata[''Name''] or '''').lower(): x.version for x in m.distributions()}; ' +
            'print(json.dumps({''torch'': d.get(''torch''), ''torchaudio'': d.get(''torchaudio'')}))'
    $r = Invoke-Native -FilePath $Python -Arguments @('-c', $code) -Quiet
    if ($r.ExitCode -ne 0) { return $null }
    $j = Get-JsonFromOutput $r.Output
    if ($null -eq $j) { return $null }
    $info = [pscustomobject]@{ Torch = (Get-ObjectValue $j @('torch')); Audio = (Get-ObjectValue $j @('torchaudio')); TorchRt = $null; AudioRt = $null }
    if ($info.Torch -and $info.Audio) {
        $code = 'import json, torch, torchaudio; print(json.dumps({''t'': torch.__version__, ''a'': torchaudio.__version__}))'
        $r = Invoke-Native -FilePath $Python -Arguments @('-c', $code) -Quiet
        $j = Get-JsonFromOutput $r.Output
        if ($r.ExitCode -eq 0 -and $null -ne $j) {
            $info.TorchRt = [string](Get-ObjectValue $j @('t'))
            $info.AudioRt = [string](Get-ObjectValue $j @('a'))
        }
    }
    return $info
}

function Test-TorchTag {
    param($Info, [string]$Tag)
    if ($null -eq $Info -or -not $Info.TorchRt -or -not $Info.AudioRt) { return $false }
    return ($Info.TorchRt.EndsWith('+' + $Tag) -and $Info.AudioRt.EndsWith('+' + $Tag))
}

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
function Invoke-Step {
    param([string]$StepTitle, [scriptblock]$StepAction, [switch]$Soft)
    Write-Section $StepTitle
    $started = Get-Date
    try {
        & $StepAction
        $script:State.Steps.Add([pscustomobject]@{ Name = $StepTitle; Ok = $true; Seconds = [int]((Get-Date) - $started).TotalSeconds })
        return $true
    }
    catch {
        $msg = $_.Exception.Message
        Write-Log "$StepTitle failed: $msg" -Level Error
        try {
            if ($null -ne $_.InvocationInfo) { Write-Log ("  " + ($_.InvocationInfo.PositionMessage -replace "`r?`n", ' | ')) -Level FileOnly }
            Write-Log ("  " + ("$($_.ScriptStackTrace)" -replace "`r?`n", ' | ')) -Level FileOnly
        }
        catch { $null = $_ }
        $script:State.Steps.Add([pscustomobject]@{ Name = $StepTitle; Ok = $false; Seconds = [int]((Get-Date) - $started).TotalSeconds })
        $script:State.Problems.Add("${StepTitle}: $msg")
        if ($Soft) {
            Write-Log 'Continuing with the remaining steps. Re-run install.bat later to retry this step.' -Level Warn
            return $false
        }
        throw "STEP_FAILED: $StepTitle"
    }
}

function Step-Preflight {
    param($P)
    if ($env:OS -ne 'Windows_NT') { throw 'This installer only runs on Windows.' }
    $os = Get-WindowsVersion
    Write-Log ("Windows version {0} (build {1}), PowerShell {2}" -f $os, $os.Build, $PSVersionTable.PSVersion)
    if ($os.Major -lt 10) { throw 'Windows 10 or 11 is required.' }
    if ($os.Build -lt 22000) { Write-Log 'Windows 10 detected. Voice Filter is built for Windows 11 but should work.' -Level Warn }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'A 64-bit Windows is required.' }
    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }
    if ($arch -eq 'ARM64') { Write-Log 'ARM64 Windows: x64 emulation will be used and there is no CUDA support.' -Level Warn }

    if (-not (Test-Path -LiteralPath $P.Package -PathType Container)) {
        throw "The folder '$($P.Package)' is missing. Extract the complete Voice Filter folder and run install.bat from inside it."
    }
    if (-not (Test-Path -LiteralPath $P.Requirements -PathType Leaf)) {
        throw "requirements.txt is missing in '$($P.AppDir)'. The app folder seems incomplete."
    }
    # The folder must be writable.
    $probe = Join-Path $P.AppDir ('.write-test-' + [guid]::NewGuid().ToString('N'))
    try { [System.IO.File]::WriteAllText($probe, 'x'); Remove-Item -LiteralPath $probe -Force }
    catch { throw "Cannot write into '$($P.AppDir)'. Move the folder somewhere inside your user folder (e.g. Documents) and try again." }
    if ($P.AppDir -match '^[A-Za-z]:\\(Program Files|Windows)') {
        Write-Log 'The app folder is inside Program Files/Windows; this may cause permission problems.' -Level Warn
    }
    $longPaths = $false
    try {
        $fsKey = Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -ErrorAction Stop
        $lp = $fsKey.PSObject.Properties['LongPathsEnabled']
        if ($null -ne $lp -and [int]$lp.Value -eq 1) { $longPaths = $true }
    }
    catch { $longPaths = $false }
    if ($P.AppDir.Length -gt 100 -and -not $longPaths) {
        Write-Log "The folder path is long ($($P.AppDir.Length) characters) and Windows long path support is off. PyTorch has deeply nested files; if the installation fails, move the folder to a short path such as C:\VoiceFilter." -Level Warn
    }
    foreach ($od in @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)) {
        if ($od -and (Test-PathInside -Path $P.AppDir -Root $od)) {
            Write-Log "The app folder is inside OneDrive ($od). OneDrive would upload the 10+ GB of Python packages and models. Better: move the folder to e.g. C:\VoiceFilter and run install.bat there." -Level Warn
            if (-not (Read-Choice -Prompt 'Install here anyway?' -Default $true)) { throw 'Cancelled by the user (folder is inside OneDrive).' }
            break
        }
    }

    # Free disk space
    try {
        $root = [System.IO.Path]::GetPathRoot($P.AppDir)
        $drive = New-Object System.IO.DriveInfo($root)
        $freeGB = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
        Write-Log "Free space on ${root}: $freeGB GB"
        if ($freeGB -lt $script:MinFreeGB) {
            Write-Log "Less than $($script:MinFreeGB) GB free. The full install (PyTorch with CUDA + all models) needs about 10-15 GB." -Level Warn
        }
    }
    catch { Write-Log "Could not determine free disk space ($($_.Exception.Message))." -Level Warn }

    # NVIDIA driver
    $smi = $null
    $cmd = Get-Command -Name 'nvidia-smi.exe' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $cmd) { $smi = $cmd.Path }
    if (-not $smi) {
        foreach ($cand in @((Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
                            (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'))) {
            if (Test-Path -LiteralPath $cand) { $smi = $cand; break }
        }
    }
    $nv = [ordered]@{ Present = $false; Driver = $null; Cuda = $null; Gpus = @() }
    if ($smi) {
        $r = Invoke-Native -FilePath $smi -Quiet
        if ($r.ExitCode -eq 0) { $nv = ConvertFrom-NvidiaSmi $r.Output }
        $q = Invoke-Native -FilePath $smi -Arguments @('--query-gpu=name,memory.total', '--format=csv,noheader') -Quiet
        if ($q.ExitCode -eq 0) { $nv.Gpus = @($q.Output | Where-Object { $_.Trim() -ne '' } | ForEach-Object { $_.Trim() }) }
        if ($nv.Present -and $null -eq $nv.Cuda) { $nv.Cuda = Get-CudaFromDriverVersion $nv.Driver }
    }
    $script:State.Nvidia = $nv
    if ($nv.Present) {
        Write-Log ("NVIDIA driver {0}, supports CUDA up to {1}; GPU: {2}" -f $nv.Driver, $nv.Cuda, (@($nv.Gpus) -join '; ')) -Level Ok
        if ($null -eq $nv.Cuda -or $nv.Cuda -lt [version]'12.4') {
            Write-Log 'The NVIDIA driver is too old for current PyTorch CUDA builds (needs CUDA 12.4+). Update the driver from nvidia.com and re-run install.bat; installing the CPU version for now.' -Level Warn
        }
    }
    else {
        Write-Log 'No NVIDIA GPU/driver found (nvidia-smi not available). The CPU version of PyTorch will be installed; processing will be MUCH slower.' -Level Warn
    }

    # A running Voice Filter would lock files in .venv.
    $running = Get-AppProcesses -AppDir $P.AppDir
    if (@($running).Count -gt 0) {
        Write-Log ('Voice Filter seems to be running from this folder (' + ((@($running) | ForEach-Object { "$($_.Name) pid $($_.Id)" }) -join ', ') + ').')
        if (Read-Choice -Prompt 'Stop it so the installation can update files?' -Default $true) {
            foreach ($rp in @($running)) {
                try { Stop-Process -Id $rp.Id -Force -ErrorAction Stop; Write-Log "Stopped $($rp.Name) (pid $($rp.Id))" -Level Detail }
                catch { Write-Log "Could not stop pid $($rp.Id): $($_.Exception.Message)" -Level Warn }
            }
            Start-Sleep -Seconds 1
        }
        else {
            Write-Log 'Continuing while it runs; updating files may fail.' -Level Warn
        }
    }
    $owner = Get-PortOwner -PortNumber $Port
    if ($null -ne $owner -and -not (Test-PathInside -Path $owner.Path -Root $P.AppDir)) {
        Write-Log "Port $Port is already used by another program ($($owner.Path), pid $($owner.Id)). Voice Filter will not be able to start; re-run with -Port <other number>." -Level Warn
    }
}

function Set-UvEnvironment {
    param($P)
    $vars = [ordered]@{
        UV_PYTHON_INSTALL_DIR      = $P.PythonDir
        UV_PYTHON_BIN_DIR          = $P.PythonBin
        UV_PYTHON_INSTALL_REGISTRY = '0'
        UV_MANAGED_PYTHON          = '1'
        UV_PYTHON_DOWNLOADS        = 'automatic'
        UV_CACHE_DIR               = $P.UvCache
        UV_TOOL_DIR                = $P.UvTools
        UV_TOOL_BIN_DIR            = $P.UvToolsBin
        UV_NO_MODIFY_PATH          = '1'
        UV_LINK_MODE               = 'copy'
        UV_NO_CONFIG               = '1'
        UV_HTTP_TIMEOUT            = '300'
        UV_INSTALL_DIR             = $P.UvDir
    }
    foreach ($k in $vars.Keys) { [Environment]::SetEnvironmentVariable($k, $vars[$k], 'Process') }
    # Variables a user might have set globally that would send uv elsewhere.
    foreach ($n in @('UV_NO_MANAGED_PYTHON', 'UV_PYTHON', 'UV_SYSTEM_PYTHON', 'UV_PROJECT_ENVIRONMENT',
                     'UV_INDEX_URL', 'UV_EXTRA_INDEX_URL', 'UV_INDEX', 'UV_DEFAULT_INDEX', 'UV_FIND_LINKS',
                     'UV_CONSTRAINT', 'UV_OVERRIDE', 'UV_TORCH_BACKEND', 'UV_PYTHON_PREFERENCE',
                     'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL')) {
        [Environment]::SetEnvironmentVariable($n, $null, 'Process')
    }
}

function Step-Uv {
    param($P)
    New-Directory $P.UvDir
    if (Test-Path -LiteralPath $P.UvExe) {
        $r = Invoke-Native -FilePath $P.UvExe -Arguments @('--version') -Quiet
        if ($r.ExitCode -eq 0) {
            Write-Log ("uv already present: " + ($r.Output -join ' ')) -Level Ok
            return
        }
        Write-Log 'runtime\uv\uv.exe does not run; downloading it again.' -Level Warn
    }
    $zip = Join-Path $P.UvDir 'uv-download.zip'
    $extract = Join-Path $P.UvDir '_extract'
    Invoke-Download -Url $script:UvUrl -OutFile $zip
    Expand-ZipFile -ZipFile $zip -Destination $extract
    $found = Get-ChildItem -LiteralPath $extract -Recurse -File -Filter 'uv.exe' | Select-Object -First 1
    if ($null -eq $found) { throw 'uv.exe was not found in the downloaded archive.' }
    foreach ($f in (Get-ChildItem -LiteralPath $found.DirectoryName -File)) {
        Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $P.UvDir $f.Name) -Force
    }
    Remove-PathIfExists $extract
    Remove-PathIfExists $zip
    $r = Invoke-Native -FilePath $P.UvExe -Arguments @('--version') -Quiet
    if ($r.ExitCode -ne 0) { throw 'The downloaded uv.exe does not run (blocked by antivirus?).' }
    Write-Log ("uv installed: " + ($r.Output -join ' ')) -Level Ok
}

function Test-Venv {
    param($P)
    if (-not (Test-Path -LiteralPath $P.VenvPython)) { return $false }
    [Environment]::SetEnvironmentVariable('VF_EXPECT_BASE', $P.PythonDir, 'Process')
    $code = 'import os, sys; ' +
            'b = os.path.normcase(os.path.realpath(sys.base_prefix)); ' +
            'e = os.path.normcase(os.path.realpath(os.environ[''VF_EXPECT_BASE''])); ' +
            'print(''%d.%d'' % sys.version_info[:2]); ' +
            'print(''inside'' if b.startswith(e) else ''outside'')'
    $r = Invoke-Native -FilePath $P.VenvPython -Arguments @('-c', $code) -Quiet
    [Environment]::SetEnvironmentVariable('VF_EXPECT_BASE', $null, 'Process')
    if ($r.ExitCode -ne 0 -or $r.Output.Count -lt 2) { return $false }
    $ver = $r.Output[$r.Output.Count - 2].Trim()
    $where = $r.Output[$r.Output.Count - 1].Trim()
    if ($ver -ne $script:PythonVersion) { Write-Log ".venv uses Python $ver, expected $($script:PythonVersion)." -Level Detail; return $false }
    if ($where -ne 'inside') { Write-Log '.venv is based on a Python outside this folder.' -Level Detail; return $false }
    return $true
}

function Step-Python {
    param($P)
    if (Test-Venv $P) {
        Write-Log "Existing .venv (Python $($script:PythonVersion), uv-managed) is fine; reusing it." -Level Ok
        return
    }
    if (Test-Path -LiteralPath $P.Venv) {
        Write-Log 'The existing .venv is broken or from another Python; recreating it.' -Level Warn
        try { Remove-PathIfExists $P.Venv }
        catch { throw "Could not delete the old .venv ($($_.Exception.Message)). Close Voice Filter (start.bat window) and re-run install.bat." }
    }
    $r = Invoke-Native -FilePath $P.UvExe -Arguments @('venv', '--python', $script:PythonVersion, '--managed-python', '--no-project', $P.Venv) -WorkingDirectory $P.AppDir
    if ($r.ExitCode -ne 0) { throw "uv venv failed with exit code $($r.ExitCode)." }
    if (-not (Test-Venv $P)) { throw 'The new .venv does not work.' }
    Add-Created $P.Venv
    Add-Created $P.PythonDir
    Write-Log "Python $($script:PythonVersion) virtual environment created in .venv" -Level Ok
}

function Step-Torch {
    param($P)
    $nv = $script:State.Nvidia
    $cuda = $null
    if ($null -ne $nv -and $nv.Present) { $cuda = $nv.Cuda }
    if ($null -ne $cuda -and $cuda -lt [version]'12.4') { $cuda = $null }
    $tags = @(Get-TorchCandidateTags -CudaVersion $cuda -ForceCpu $Cpu.IsPresent -Override $TorchIndex)
    if ($tags[0] -eq 'cpu') {
        $reason = 'no usable NVIDIA driver'
        if ($Cpu) { $reason = '-Cpu was given' }
        if ($TorchIndex) { $reason = '-TorchIndex cpu was given' }
        Write-Log "Installing the CPU build of PyTorch ($reason). Processing will be much slower than on the GPU." -Level Warn
    }
    else {
        Write-Log ('PyTorch CUDA builds to try, in order: ' + ($tags -join ', '))
    }
    $current = Get-TorchInfo -Python $P.VenvPython
    $chosen = $null
    foreach ($tag in $tags) {
        if (Test-TorchTag -Info $current -Tag $tag) {
            Write-Log "PyTorch $($current.TorchRt) / torchaudio $($current.AudioRt) already installed." -Level Ok
            $chosen = $tag
            break
        }
        $url = $script:TorchIndexBase + $tag
        Write-Log "Installing PyTorch from $url (about 2.5 GB for CUDA builds, this can take a while) ..."
        $uvArgs = @('pip', 'install', '--python', $P.VenvPython, '--index', $url)
        if ($null -ne $current -and $current.Torch) {
            $uvArgs += @('--reinstall-package', 'torch', '--reinstall-package', 'torchaudio')
        }
        $uvArgs += $script:TorchPackages
        $r = Invoke-Native -FilePath $P.UvExe -Arguments $uvArgs -Live -WorkingDirectory $P.AppDir
        if ($r.ExitCode -eq 0) {
            $current = Get-TorchInfo -Python $P.VenvPython
            if (Test-TorchTag -Info $current -Tag $tag) { $chosen = $tag; break }
            $seen = 'not importable'
            if ($null -ne $current -and $current.TorchRt) { $seen = "$($current.TorchRt)" }
            Write-Log "Installed PyTorch does not carry the expected '+$tag' tag ($seen)." -Level Warn
        }
        else {
            Write-Log "Installing PyTorch from the $tag index failed (exit code $($r.ExitCode))." -Level Warn
        }
    }
    if (-not $chosen) {
        throw ('PyTorch could not be installed from: ' + ($tags -join ', ') + '. Check the internet connection, or try -TorchIndex cu126 / -Cpu.')
    }
    $script:State.TorchTag = $chosen
    $script:State.TorchVersion = $current.TorchRt
    $script:State.TorchAudioVersion = $current.AudioRt
    $script:State.TorchMeta = $current.Torch
    $constraints = "# Written by install.ps1: keeps the CUDA build of PyTorch when installing requirements.txt`r`n" +
                   "torch==$($current.Torch)`r`ntorchaudio==$($current.Audio)`r`n"
    New-Directory $P.Runtime
    Write-Utf8File -Path $P.Constraints -Text $constraints
    Write-Log "PyTorch $($current.TorchRt), torchaudio $($current.AudioRt) ($chosen)" -Level Ok
}

function Step-Requirements {
    param($P)
    $r = Invoke-Native -FilePath $P.UvExe -Arguments @('pip', 'install', '--python', $P.VenvPython,
        '-r', $P.Requirements, '-c', $P.Constraints) -WorkingDirectory $P.AppDir
    if ($r.ExitCode -ne 0) { throw "Installing requirements.txt failed (exit code $($r.ExitCode)); see the messages above." }
    $after = Get-TorchInfo -Python $P.VenvPython
    $afterVersion = $null
    if ($null -ne $after) { $afterVersion = $after.Torch }
    if ($afterVersion -ne $script:State.TorchMeta) {
        throw "requirements.txt replaced PyTorch ($($script:State.TorchMeta) -> $afterVersion). Re-run install.bat."
    }
    Write-Log 'Python packages installed.' -Level Ok

    # Check that torch really sees the GPU.
    $code = 'import json, torch; a = torch.cuda.is_available(); ' +
            'print(json.dumps({''torch'': torch.__version__, ''cuda'': torch.version.cuda, ''available'': a, ''name'': torch.cuda.get_device_name(0) if a else None}))'
    $t = Invoke-Native -FilePath $P.VenvPython -Arguments @('-c', $code) -Quiet
    $j = Get-JsonFromOutput $t.Output
    if ($t.ExitCode -ne 0 -or $null -eq $j) {
        throw ('"import torch" failed: ' + (($t.Output | Select-Object -Last 5) -join ' | '))
    }
    $script:State.CudaAvailable = [bool](Get-ObjectValue $j @('available') $false)
    $script:State.GpuName = Get-ObjectValue $j @('name')
    if ($script:State.CudaAvailable) {
        $script:State.Device = 'cuda'
        Write-Log "CUDA works: $($script:State.GpuName) (torch CUDA $(Get-ObjectValue $j @('cuda')))" -Level Ok
    }
    else {
        $script:State.Device = 'cpu'
        if ($script:State.TorchTag -ne 'cpu') {
            Write-Log ''
            Write-Log '*****************************************************************' -Level Warn
            Write-Log '* A CUDA build of PyTorch was installed, but torch cannot use the GPU.' -Level Warn
            Write-Log '* Voice Filter will run on the CPU (very slow).' -Level Warn
            Write-Log '* Fix: update the NVIDIA driver (nvidia.com/drivers), reboot, re-run install.bat.' -Level Warn
            Write-Log '*****************************************************************' -Level Warn
        }
    }
}

function Test-FfmpegPair {
    param([string]$Ffmpeg, [string]$Ffprobe)
    if (-not $Ffmpeg -or -not $Ffprobe) { return $false }
    if (-not (Test-Path -LiteralPath $Ffmpeg) -or -not (Test-Path -LiteralPath $Ffprobe)) { return $false }
    $a = Invoke-Native -FilePath $Ffmpeg -Arguments @('-hide_banner', '-version') -Quiet
    $b = Invoke-Native -FilePath $Ffprobe -Arguments @('-hide_banner', '-version') -Quiet
    return ($a.ExitCode -eq 0 -and $b.ExitCode -eq 0)
}

function Step-Ffmpeg {
    param($P)
    $f = Get-Command -Name 'ffmpeg.exe' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $g = Get-Command -Name 'ffprobe.exe' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $f -and $null -ne $g -and -not (Test-PathInside -Path $f.Path -Root $P.AppDir)) {
        if (Test-FfmpegPair -Ffmpeg $f.Path -Ffprobe $g.Path) {
            $script:State.FfmpegSource = 'pre-existing'
            $script:State.Ffmpeg = $f.Path
            $script:State.Ffprobe = $g.Path
            Write-Log "Using the ffmpeg already installed on this PC: $($f.Path)" -Level Ok
            Write-Log 'It is not modified and will not be removed by uninstall.' -Level Detail
            return
        }
        Write-Log "ffmpeg on PATH ($($f.Path)) does not run; using a portable copy instead." -Level Warn
    }
    $exe = Join-Path $P.FfmpegBin 'ffmpeg.exe'
    $probe = Join-Path $P.FfmpegBin 'ffprobe.exe'
    if (Test-FfmpegPair -Ffmpeg $exe -Ffprobe $probe) {
        Write-Log 'Portable ffmpeg in runtime\ffmpeg is already present.' -Level Ok
    }
    else {
        New-Directory $P.FfmpegDir
        $zip = Join-Path $P.FfmpegDir 'ffmpeg-download.zip'
        $extract = Join-Path $P.FfmpegDir '_extract'
        $ok = $false
        foreach ($url in $script:FfmpegUrls) {
            try {
                Invoke-Download -Url $url -OutFile $zip
                Expand-ZipFile -ZipFile $zip -Destination $extract
                $found = Get-ChildItem -LiteralPath $extract -Recurse -File -Filter 'ffmpeg.exe' |
                    Where-Object { $_.Directory.Name -eq 'bin' } | Select-Object -First 1
                if ($null -eq $found) { throw 'bin\ffmpeg.exe not found in the archive' }
                $buildRoot = $found.Directory.Parent.FullName
                Remove-PathIfExists $P.FfmpegBin
                New-Directory $P.FfmpegBin
                foreach ($item in (Get-ChildItem -LiteralPath $found.DirectoryName -File)) {
                    Copy-Item -LiteralPath $item.FullName -Destination (Join-Path $P.FfmpegBin $item.Name) -Force
                }
                foreach ($lic in (Get-ChildItem -LiteralPath $buildRoot -File -ErrorAction SilentlyContinue)) {
                    Copy-Item -LiteralPath $lic.FullName -Destination (Join-Path $P.FfmpegDir $lic.Name) -Force
                }
                if (-not (Test-FfmpegPair -Ffmpeg $exe -Ffprobe $probe)) { throw 'the extracted ffmpeg.exe does not run' }
                $ok = $true
                Write-Log "Portable ffmpeg installed from $url" -Level Ok
                break
            }
            catch {
                Write-Log "ffmpeg from $url failed: $($_.Exception.Message)" -Level Warn
            }
            finally {
                Remove-PathIfExists $extract
                Remove-PathIfExists $zip
            }
        }
        if (-not $ok) { throw 'Could not get a portable ffmpeg. Install ffmpeg yourself (e.g. "winget install Gyan.FFmpeg"), open a new window and re-run install.bat.' }
        Add-Created $P.FfmpegDir
    }
    $script:State.FfmpegSource = 'portable'
    $script:State.Ffmpeg = $exe
    $script:State.Ffprobe = $probe
}

function Step-Config {
    param($P)
    foreach ($d in @($P.Data, $P.Tmp, $P.Logs, $P.Cache, $P.Models, $P.HfHome, $P.TorchHome)) { New-Directory $d }
    Add-Created $P.Models
    Add-Created $P.Data

    # config.json: merge our keys, keep anything else the user tuned.
    $cfg = Read-JsonFile $P.Config
    if ($null -eq $cfg) { $cfg = [ordered]@{} }
    $cfg['port'] = $Port
    $cfg['device'] = $script:State.Device
    $cfg['ffmpeg'] = $script:State.Ffmpeg
    $cfg['ffprobe'] = $script:State.Ffprobe
    $cfg['ffmpeg_source'] = $script:State.FfmpegSource
    $cfg['torch_index'] = $script:State.TorchTag
    Write-Utf8File -Path $P.Config -Text (($cfg | ConvertTo-Json -Depth 8) + "`r`n")
    Add-Created $P.Config
    Write-Log "Wrote config.json (port $Port, device $($script:State.Device))." -Level Ok

    $envList = Get-AppEnvironment -PortNumber $Port -Device $script:State.Device -FfmpegPath $script:State.Ffmpeg `
        -FfprobePath $script:State.Ffprobe -AppDir $P.AppDir
    $binRel = $null
    if ($script:State.FfmpegSource -eq 'portable') { $binRel = Get-RelativeAppPath -Path $P.FfmpegBin -AppDir $P.AppDir }
    Write-AsciiFile -Path $P.StartBat -Text (New-StartBatText -EnvList $envList -PortNumber $Port -FfmpegBinRel $binRel)
    Write-Utf8File -Path $P.StartPs1 -Text (New-StartPs1Text -EnvList $envList -PortNumber $Port -FfmpegBinRel $binRel) -Bom
    Add-Created $P.StartBat
    Add-Created $P.StartPs1
    Write-Log 'Wrote start.bat and start.ps1.' -Level Ok
    # From here on, child processes (model download, self check) get the app environment.
    Set-ProcessEnvironment -EnvList $envList -AppDir $P.AppDir
    if ($binRel) { $env:PATH = $P.FfmpegBin + ';' + $env:PATH }
}

function Step-Models {
    param($P)
    $list = @(Get-PresetList $Presets)
    Write-Log ('Downloading models for presets: ' + ($list -join ', ') + ' (several GB, may take a while) ...')
    $r = Invoke-Native -FilePath $P.VenvPython -Arguments @('-m', 'voicefilter.download', '--presets', ($list -join ',')) -Live -WorkingDirectory $P.AppDir
    # Remember which Hugging Face repos ended up in models\hf, so uninstall can look for
    # stray copies of exactly these in the user's default cache.
    $hub = Join-Path $P.HfHome 'hub'
    if (Test-Path -LiteralPath $hub) {
        $script:State.HfModels = @(Get-ChildItem -LiteralPath $hub -Directory -Filter 'models--*' | ForEach-Object { $_.Name })
    }
    if ($r.ExitCode -ne 0) {
        $script:State.ModelsOk = $false
        throw "The model download failed (exit code $($r.ExitCode)). Re-run install.bat to resume; finished files are kept."
    }
    $script:State.ModelsOk = $true
    Write-Log 'Models downloaded.' -Level Ok
}

function Get-PublicProfiles {
    $result = @()
    try {
        foreach ($cp in (Get-NetConnectionProfile -ErrorAction Stop)) {
            if ("$($cp.NetworkCategory)" -eq 'Public') {
                $result += [pscustomobject]@{
                    InterfaceIndex = [int]$cp.InterfaceIndex
                    InterfaceAlias = [string]$cp.InterfaceAlias
                    Name           = [string]$cp.Name
                }
            }
        }
    }
    catch { Write-Log "Could not read the network profiles ($($_.Exception.Message))." -Level Detail }
    return , $result
}

function Test-FirewallRuleCurrent {
    # True when exactly our rule exists with the right settings (no admin needed to read).
    try {
        $all = @()
        foreach ($pat in $script:FirewallRulePatterns) {
            $all += @(Get-NetFirewallRule -DisplayName $pat -ErrorAction SilentlyContinue)
        }
        if ($all.Count -ne 1) { return $false }
        $r = $all[0]
        if ($r.DisplayName -ne $script:FirewallRuleName) { return $false }
        if ("$($r.Enabled)" -ne 'True' -or "$($r.Direction)" -ne 'Inbound' -or "$($r.Action)" -ne 'Allow') { return $false }
        if ("$($r.Profile)" -ne 'Private') { return $false }
        $pf = $r | Get-NetFirewallPortFilter -ErrorAction Stop
        return ("$($pf.Protocol)" -eq 'TCP' -and "$($pf.LocalPort)" -eq "$Port")
    }
    catch { return $false }
}

function Step-Firewall {
    param($P)
    $script:State.FirewallRule = $script:FirewallRuleName
    $public = Get-PublicProfiles
    $script:State.PublicProfiles = $public
    $changes = @()
    foreach ($pp in $public) {
        Write-Log ''
        Write-Log ("The network '{0}' ({1}) is set to 'Public'." -f $pp.Name, $pp.InterfaceAlias) -Level Warn
        Write-Log 'Windows blocks phones on Public networks, so the firewall rule (Private networks only) will not help there.'
        Write-Log 'Switching your home WiFi to Private is safe at home; do NOT do it for cafe/hotel/office networks.'
        if (Read-Choice -Prompt ("Switch '{0}' to Private?" -f $pp.Name) -Default $false) {
            $changes += [pscustomobject]@{ InterfaceIndex = $pp.InterfaceIndex; InterfaceAlias = $pp.InterfaceAlias; Name = $pp.Name; From = 'Public'; To = 'Private' }
        }
        else {
            Write-Log ("Left '{0}' as Public. Phones can only connect after you set it to Private (Settings > Network & internet > Wi-Fi > properties)." -f $pp.Name) -Level Warn
        }
    }
    $ruleCurrent = Test-FirewallRuleCurrent
    if ($ruleCurrent -and $changes.Count -eq 0) {
        Write-Log "Firewall rule '$($script:FirewallRuleName)' is already in place." -Level Ok
        $script:State.FirewallOk = $true
        return
    }
    New-Directory $P.Tmp
    $resultFile = Join-Path $P.Tmp ('elevated-result-' + [guid]::NewGuid().ToString('N') + '.json')
    $text = New-ElevatedScriptText -RuleName $script:FirewallRuleName -RemovePatterns $script:FirewallRulePatterns `
        -PortNumber $Port -CreateRule (-not $ruleCurrent) -ProfileChanges $changes -ResultFile $resultFile
    if (-not $ruleCurrent) { Write-Log "Creating firewall rule '$($script:FirewallRuleName)' (inbound TCP $Port, Private networks only) ..." }
    $res = Invoke-Elevated -ScriptText $text -ResultFile $resultFile
    if ($null -eq $res) {
        throw 'The firewall rule could not be created (no administrator permission). The app works on this PC; for phones re-run install.bat and accept the Windows prompt.'
    }
    foreach ($pc in $changes) {
        $item = @($res.profiles | Where-Object { [int]$_.index -eq $pc.InterfaceIndex }) | Select-Object -First 1
        if ($null -ne $item -and $item.ok) {
            Write-Log ("Network '{0}' is now Private." -f $pc.Name) -Level Ok
            $script:State.NetworkChange = [ordered]@{ interface = $pc.InterfaceAlias; interface_index = $pc.InterfaceIndex; name = $pc.Name; from = $pc.From; to = $pc.To }
        }
        else {
            $err = 'unknown error'
            if ($null -ne $item) { $err = $item.error }
            Write-Log ("Could not switch '{0}' to Private: {1}" -f $pc.Name, $err) -Level Warn
        }
    }
    if (-not $res.rule_ok) { throw "Creating the firewall rule failed: $($res.rule_error)" }
    if (-not (Test-FirewallRuleCurrent)) { throw 'The firewall rule was created but could not be verified.' }
    $script:State.FirewallOk = $true
    Write-Log "Firewall rule '$($script:FirewallRuleName)' created." -Level Ok
}

function Step-Shortcuts {
    param($P)
    $icon = $null
    foreach ($cand in @((Join-Path $P.Package 'assets\voicefilter.ico'), (Join-Path $P.Package 'web\favicon.ico'))) {
        if (Test-Path -LiteralPath $cand) { $icon = $cand + ',0'; break }
    }
    foreach ($lnk in @(Get-ShortcutPaths)) {
        New-Shortcut -LinkPath $lnk -Target $P.StartBat -WorkingDirectory $P.AppDir -Description 'Start Voice Filter (opens the browser)' -Icon $icon
        if (-not $script:State.Shortcuts.Contains($lnk)) { $script:State.Shortcuts.Add($lnk) }
        Write-Log "Shortcut: $lnk" -Level Ok
    }
}

function Step-SelfCheck {
    param($P)
    $r = Invoke-Native -FilePath $P.VenvPython -Arguments @('-m', 'voicefilter.selfcheck') -Quiet -WorkingDirectory $P.AppDir
    $j = Get-JsonFromOutput $r.Output
    $script:State.SelfCheck = $j
    $script:State.SelfCheckOk = ($r.ExitCode -eq 0)
    if ($null -ne $j) {
        Write-Log ('Self check report: ' + (($j | ConvertTo-Json -Depth 6 -Compress))) -Level FileOnly
        foreach ($prop in $j.PSObject.Properties) {
            $v = $prop.Value
            if ($v -is [System.Management.Automation.PSCustomObject] -or ($v -is [System.Array])) { $v = ($v | ConvertTo-Json -Depth 4 -Compress) }
            Write-Log ("{0,-22} {1}" -f $prop.Name, $v) -Level Detail
        }
    }
    else {
        foreach ($l in ($r.Output | Select-Object -Last 15)) { Write-Log $l -Level Detail }
    }
    if ($r.ExitCode -ne 0) { throw "The self check reported a problem (exit code $($r.ExitCode)); see above." }
    Write-Log 'Self check passed.' -Level Ok
}

function Save-Manifest {
    param($P)
    $python = $null
    if (Test-Path -LiteralPath $P.VenvPython) { $python = [ordered]@{ source = 'uv-managed'; path = $P.VenvPython; base_dir = $P.PythonDir; version = $script:PythonVersion } }
    $ff = $null
    if ($script:State.FfmpegSource) { $ff = [ordered]@{ source = $script:State.FfmpegSource; path = $script:State.Ffmpeg; ffprobe = $script:State.Ffprobe } }
    $rules = @()
    if ($script:State.FirewallOk) { $rules = @($script:FirewallRuleName) }
    $new = [ordered]@{
        installed_at            = (Get-Date).ToString('o')
        app_dir                 = $P.AppDir
        port                    = $Port
        presets                 = $Presets
        created                 = $script:State.Created.ToArray()
        firewall_rule           = $(if ($script:State.FirewallOk) { $script:FirewallRuleName } else { $null })
        firewall_rules_created  = $rules
        shortcuts               = $script:State.Shortcuts.ToArray()
        ffmpeg                  = $ff
        python                  = $python
        uv                      = [ordered]@{ path = $P.UvExe; cache_dir = $P.UvCache; python_install_dir = $P.PythonDir }
        torch                   = [ordered]@{ index = $script:State.TorchTag; torch = $script:State.TorchVersion; torchaudio = $script:State.TorchAudioVersion; cuda_available = $script:State.CudaAvailable; gpu = $script:State.GpuName }
        device                  = $script:State.Device
        models_downloaded       = $script:State.ModelsOk
        hf_models               = @($script:State.HfModels)
        network_profile_changed = $script:State.NetworkChange
        last_run_problems       = $script:State.Problems.ToArray()
    }
    $old = Read-JsonFile $P.Manifest
    # Values that a step did not touch in this run keep their old value.
    foreach ($k in @('ffmpeg', 'python', 'firewall_rule', 'models_downloaded', 'device')) {
        if ($null -eq $new[$k] -and $null -ne $old) { $new[$k] = Get-ObjectValue $old @($k) $null }
    }
    if ($null -eq $script:State.TorchTag -and $null -ne $old) { $new['torch'] = Get-ObjectValue $old @('torch') $null }
    if (@($script:State.HfModels).Count -eq 0 -and $null -ne $old) { $new['hf_models'] = @(Get-ObjectValue $old @('hf_models') @()) }
    $merged = Merge-Manifest -Old $old -New $new
    if (-not (@($merged['created']) -contains $P.Manifest)) { $merged['created'] = @($merged['created']) + @($P.Manifest) }
    Write-Utf8File -Path $P.Manifest -Text (($merged | ConvertTo-Json -Depth 8) + "`r`n")
}

function Write-Summary {
    param($P, [bool]$Fatal)
    Write-Section 'Summary'
    $nv = $script:State.Nvidia
    if ($null -ne $nv -and $nv.Present) { Write-Log ("GPU:        {0} (driver {1})" -f (@($nv.Gpus) -join '; '), $nv.Driver) }
    else { Write-Log 'GPU:        none detected' }
    if ($script:State.TorchVersion) {
        $c = 'NOT available'
        if ($script:State.CudaAvailable) { $c = 'available' }
        Write-Log ("PyTorch:    {0} ({1}), CUDA {2}" -f $script:State.TorchVersion, $script:State.TorchTag, $c)
    }
    if ($script:State.FfmpegSource) { Write-Log ("ffmpeg:     {0} ({1})" -f $script:State.Ffmpeg, $script:State.FfmpegSource) }
    if ($null -ne $script:State.ModelsOk) {
        $mtext = 'incomplete - re-run install.bat'
        if ($script:State.ModelsOk) { $mtext = 'downloaded' }
        Write-Log "Models:     $mtext"
    }
    elseif ($SkipModels) { Write-Log 'Models:     skipped (-SkipModels); they download on first use' }
    if (-not $NoFirewall) {
        $ftext = 'NOT created - phones cannot connect until it exists'
        if ($script:State.FirewallOk) { $ftext = $script:FirewallRuleName }
        Write-Log "Firewall:   $ftext"
    }
    if (-not $Fatal) {
        Write-Log ''
        Write-Log "Start:      double-click 'Voice Filter' on the Desktop, or start.bat in this folder" -Level Ok
        Write-Log "This PC:    http://localhost:$Port"
        $urls = @()
        try {
            $ips = Get-LanAddresses
            foreach ($ip in $ips) { $urls += "http://${ip}:$Port" }
        }
        catch { $urls = @() }
        if ($urls.Count -gt 0) {
            Write-Log ('Phone:      ' + ($urls -join '   or   ') + '   (same WiFi)')
        }
        if (@($script:State.PublicProfiles).Count -gt 0 -and $null -eq $script:State.NetworkChange) {
            Write-Log 'Phone access is blocked while your network is set to Public (see above).' -Level Warn
        }
    }
    if ($script:State.Problems.Count -gt 0) {
        Write-Log ''
        Write-Log 'Problems:' -Level Error
        foreach ($p in $script:State.Problems) { Write-Log "  - $p" -Level Error }
        Write-Log 'Fix the cause and run install.bat again; finished steps are skipped.'
    }
    if ($script:State.LogFile) { Write-Log "Log file:   $($script:State.LogFile)" }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function Invoke-Main {
    $P = Get-AppPaths -AppDir $PSScriptRoot
    # Log file first, so everything after this is recorded.
    try {
        New-Directory $P.Logs
        $script:State.LogFile = Join-Path $P.Logs ('install-' + (Get-Date).ToString('yyyyMMdd-HHmmss') + '.log')
    }
    catch { Write-Host "Warning: cannot create the log folder ($($_.Exception.Message))." -ForegroundColor Yellow }
    try {
        $script:State.OldConsoleEncoding = [Console]::OutputEncoding
        [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    }
    catch { $script:State.OldConsoleEncoding = $null }
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    }
    catch { $null = $_ }

    $presetList = @()
    try { $presetList = @(Get-PresetList $Presets) } catch { Write-Log $_.Exception.Message -Level Error; return 1 }

    Write-Log "Voice Filter installer" -Level Step
    Write-Log "App folder: $($P.AppDir)"
    Write-Log ''
    Write-Log 'This will (everything inside the app folder unless noted):'
    Write-Log '  1. check Windows, disk space and the NVIDIA driver'
    Write-Log '  2. download uv (Python manager) to runtime\uv'
    Write-Log "  3. create a private Python $($script:PythonVersion) in runtime\python and .venv"
    $torchText = 'PyTorch with CUDA (matching your driver)'
    if ($Cpu) { $torchText = 'PyTorch (CPU build, -Cpu)' }
    Write-Log "  4. install $torchText and requirements.txt into .venv"
    Write-Log '  5. use ffmpeg from PATH, or download a portable ffmpeg to runtime\ffmpeg'
    if ($SkipModels) { Write-Log '  6. (skipped) model download' }
    else { Write-Log ('  6. download models for presets ' + ($presetList -join ', ') + ' to models\') }
    if ($NoFirewall) { Write-Log '  7. (skipped) firewall rule' }
    else { Write-Log "  7. OUTSIDE the folder: firewall rule '$($script:FirewallRuleName)', Private networks only (asks for admin once)" }
    Write-Log '  8. write config.json, start.bat and start.ps1'
    if ($NoShortcuts) { Write-Log '  9. (skipped) shortcuts' }
    else { Write-Log "  9. OUTSIDE the folder: 'Voice Filter' shortcuts on the Desktop and in the Start Menu" }
    Write-Log ' 10. run a self check and write install-manifest.json'
    Write-Log 'Your PATH, profile and system Python/ffmpeg are not changed. uninstall.bat removes everything again.'
    Write-Log ''
    if (-not (Read-Choice -Prompt 'Continue?' -Default $true)) {
        Write-Log 'Cancelled; nothing was changed.'
        return 1
    }

    $fatal = $false
    try {
        $null = Invoke-Step 'Checking the system' { Step-Preflight $P }
        Set-UvEnvironment $P
        $null = Invoke-Step 'uv (Python manager)' { Step-Uv $P }
        $null = Invoke-Step "Python $($script:PythonVersion) virtual environment" { Step-Python $P }
        $null = Invoke-Step 'PyTorch' { Step-Torch $P }
        $null = Invoke-Step 'Python packages (requirements.txt)' { Step-Requirements $P }
        $null = Invoke-Step 'ffmpeg' { Step-Ffmpeg $P }
        $null = Invoke-Step 'Configuration and start scripts' { Step-Config $P }
        if (-not $SkipModels) { $null = Invoke-Step 'Models' { Step-Models $P } -Soft }
        if (-not $NoFirewall) { $null = Invoke-Step 'Firewall and network' { Step-Firewall $P } -Soft }
        if (-not $NoShortcuts) { $null = Invoke-Step 'Shortcuts' { Step-Shortcuts $P } -Soft }
        $null = Invoke-Step 'Self check' { Step-SelfCheck $P } -Soft
        # Free some space: drop cache entries that are no longer referenced.
        try { $null = Invoke-Native -FilePath $P.UvExe -Arguments @('cache', 'prune') -Quiet } catch { $null = $_ }
    }
    catch {
        $fatal = $true
        if ("$($_.Exception.Message)" -notlike 'STEP_FAILED:*') {
            Write-Log "Unexpected error: $($_.Exception.Message)" -Level Error
            Write-Log ($_.ScriptStackTrace -replace "`r?`n", ' | ') -Level FileOnly
            $script:State.Problems.Add("Unexpected error: $($_.Exception.Message)")
        }
        Write-Log 'The installation stopped. Fix the problem above and run install.bat again; finished steps are reused.' -Level Error
    }
    finally {
        try { Save-Manifest $P } catch { Write-Log "Could not write install-manifest.json: $($_.Exception.Message)" -Level Warn }
    }
    Write-Summary -P $P -Fatal $fatal
    if ($null -ne $script:State.OldConsoleEncoding) {
        try { [Console]::OutputEncoding = $script:State.OldConsoleEncoding } catch { $null = $_ }
    }
    if ($fatal) { return 1 }
    if ($script:State.Problems.Count -gt 0) { return 2 }
    Write-Log ''
    Write-Log 'Installation finished.' -Level Ok
    return 0
}

# Dot-sourcing (". .\install.ps1") only loads the functions, e.g. for tests.
if ($MyInvocation.InvocationName -ne '.') {
    $exitCode = Invoke-Main
    exit $exitCode
}
