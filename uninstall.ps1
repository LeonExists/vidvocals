#Requires -Version 5.1
<#
.SYNOPSIS
    Removes everything install.ps1 created for Voice Filter.

.DESCRIPTION
    Stops a running Voice Filter server (only processes started from this folder),
    then removes .venv, runtime (uv, Python, portable ffmpeg, caches), models, data,
    generated start scripts and config, __pycache__ folders, the firewall rule, the
    shortcuts and install-manifest.json. Anything outside the app folder that might be
    shared (stray caches, the network profile) is only touched after asking.
    Safe to run again: things that are already gone are simply reported.

.PARAMETER Yes
    Non-interactive: remove everything that belongs to Voice Filter without asking.
    Shared things (caches outside the folder, network profile) are then left alone.
.PARAMETER KeepModels
    Keep the downloaded models (models\ folder), e.g. for a later re-install.
.PARAMETER RemoveAppFolder
    Also delete the app folder itself (source code) after this script has finished.

.EXAMPLE
    uninstall.bat
.EXAMPLE
    uninstall.bat -Yes -KeepModels
#>
[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$KeepModels,
    [switch]$RemoveAppFolder
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:AppName = 'Voice Filter'
$script:FirewallRulePatterns = @('Voice Filter (TCP *)', 'VoiceFilter (TCP *)')
$script:Removed = New-Object System.Collections.Generic.List[string]
$script:Left = New-Object System.Collections.Generic.List[string]
$script:Failed = New-Object System.Collections.Generic.List[string]
$script:OutsideFailed = $false

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Log {
    param(
        [Parameter(Position = 0)][AllowEmptyString()][string]$Message = '',
        [ValidateSet('Info', 'Ok', 'Warn', 'Error', 'Step', 'Detail')]
        [string]$Level = 'Info'
    )
    switch ($Level) {
        'Ok'     { Write-Host ('[ OK ] ' + $Message) -ForegroundColor Green }
        'Warn'   { Write-Host ('[WARN] ' + $Message) -ForegroundColor Yellow }
        'Error'  { Write-Host ('[FAIL] ' + $Message) -ForegroundColor Red }
        'Step'   { Write-Host $Message -ForegroundColor Cyan }
        'Detail' { Write-Host ('       ' + $Message) -ForegroundColor DarkGray }
        default  { Write-Host $Message }
    }
}

function Write-Section {
    param([string]$Title)
    Write-Log ''
    Write-Log ('=== ' + $Title + ' ===') -Level Step
}

function Read-Choice {
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
        if ($a -eq '') { return $Default }
        if (@('y', 'yes', 'j', 'ja') -contains $a) { return $true }
        if (@('n', 'no', 'nein') -contains $a) { return $false }
        Write-Host 'Please answer y or n.'
    }
}

function ConvertTo-PsLiteral {
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

function Test-PathInside {
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

function Get-ObjectValue {
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

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try {
        $text = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
        if ($text.Trim() -eq '') { return $null }
        return ($text | ConvertFrom-Json)
    }
    catch {
        Write-Log "Could not read $Path ($($_.Exception.Message)); continuing without it." -Level Warn
        return $null
    }
}

function Format-Size {
    param([double]$Bytes)
    if ($Bytes -ge 1GB) { return ('{0:N1} GB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N0} MB' -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ('{0:N0} KB' -f ($Bytes / 1KB)) }
    return ('{0:N0} bytes' -f $Bytes)
}

function Get-ItemSize {
    param([string]$Path)
    try {
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        if (-not $item.PSIsContainer) { return [double]$item.Length }
        $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
                Measure-Object -Property Length -Sum).Sum
        if ($null -eq $sum) { return 0.0 }
        return [double]$sum
    }
    catch { return 0.0 }
}

function Test-IsAdmin {
    try {
        $id = [Security.Principal.WindowsIdentity]::GetCurrent()
        $pr = New-Object Security.Principal.WindowsPrincipal($id)
        return $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    catch { return $false }
}

function Get-PowerShellExe {
    $exe = $null
    try { $exe = (Get-Process -Id $PID).Path } catch { $exe = $null }
    if (-not $exe) { $exe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe' }
    return $exe
}

function Invoke-Elevated {
    # Runs a script in one elevated PowerShell; returns its JSON result, or $null if UAC was declined.
    param([string]$ScriptText, [string]$ResultFile)
    if (Test-Path -LiteralPath $ResultFile) { Remove-Item -LiteralPath $ResultFile -Force }
    if (Test-IsAdmin) {
        & ([scriptblock]::Create($ScriptText))
    }
    else {
        $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($ScriptText))
        Write-Log 'Windows will ask for administrator permission (UAC) for this step only.'
        try {
            $null = Start-Process -FilePath (Get-PowerShellExe) -Verb RunAs -Wait -PassThru -WindowStyle Hidden `
                -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encoded)
        }
        catch {
            Write-Log "Administrator permission was not granted ($($_.Exception.Message))." -Level Warn
            return $null
        }
    }
    if (Test-Path -LiteralPath $ResultFile) {
        try { return ([System.IO.File]::ReadAllText($ResultFile) | ConvertFrom-Json) }
        finally { Remove-Item -LiteralPath $ResultFile -Force -ErrorAction SilentlyContinue }
    }
    return $null
}

function New-ElevatedUninstallText {
    param([string[]]$Patterns, [object[]]$ProfileChanges, [string]$ResultFile)
    $pats = ($Patterns | ForEach-Object { ConvertTo-PsLiteral $_ }) -join ', '
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add("`$ErrorActionPreference = 'Stop'")
    $L.Add('$res = [ordered]@{ rules_removed = @(); rule_error = $null; profiles = @() }')
    $L.Add('try {')
    $L.Add("    foreach (`$pat in @($pats)) {")
    $L.Add('        foreach ($r in @(Get-NetFirewallRule -DisplayName $pat -ErrorAction SilentlyContinue)) {')
    $L.Add('            $n = $r.DisplayName; Remove-NetFirewallRule -InputObject $r -ErrorAction Stop; $res.rules_removed += $n')
    $L.Add('        }')
    $L.Add('    }')
    $L.Add('} catch { $res.rule_error = $_.Exception.Message }')
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

function New-FolderRemovalText {
    # Script for a hidden PowerShell that waits until this uninstaller (and the cmd.exe
    # running uninstall.bat) has exited, then deletes the app folder.
    param([string]$Folder, [int[]]$WaitIds)
    $ids = ($WaitIds | ForEach-Object { [string][int]$_ }) -join ', '
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add("`$ErrorActionPreference = 'SilentlyContinue'")
    $L.Add(('$d = {0}' -f (ConvertTo-PsLiteral $Folder)))
    $L.Add(('foreach ($i in @({0})) {{ try {{ Wait-Process -Id $i -Timeout 900 -ErrorAction Stop }} catch {{ $null = $_ }} }}' -f $ids))
    $L.Add('Start-Sleep -Seconds 2')
    $L.Add('Set-Location -LiteralPath $env:SystemRoot')
    $L.Add('for ($n = 0; $n -lt 15; $n++) {')
    $L.Add('    if (-not (Test-Path -LiteralPath $d)) { break }')
    $L.Add('    try { Remove-Item -LiteralPath $d -Recurse -Force -ErrorAction Stop } catch { $null = $_ }')
    $L.Add('    if (Test-Path -LiteralPath $d) {')
    $L.Add('        $p = Start-Process -FilePath (Join-Path $env:SystemRoot ''System32\cmd.exe'') -ArgumentList (''/d /c rd /s /q "'' + $d + ''"'') -WindowStyle Hidden -Wait -PassThru')
    $L.Add('        Start-Sleep -Seconds 2')
    $L.Add('    }')
    $L.Add('}')
    return ($L.ToArray() -join "`r`n")
}

function Get-AppProcesses {
    # Processes started from inside the app folder, plus cmd.exe windows running our start.bat.
    param([string]$AppDir)
    $result = @()
    $startBat = Join-Path $AppDir 'start.bat'
    try {
        foreach ($p in (Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)) {
            if ([int]$p.ProcessId -eq $PID) { continue }
            $exe = $p.ExecutablePath
            $cmdLine = $p.CommandLine
            $ours = $false
            if ($exe -and (Test-PathInside -Path $exe -Root $AppDir)) { $ours = $true }
            elseif ($p.Name -eq 'cmd.exe' -and $cmdLine -and $cmdLine.IndexOf($startBat, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { $ours = $true }
            if ($ours) {
                $shown = $exe
                if (-not $shown) { $shown = $p.Name }
                $result += [pscustomobject]@{ Id = [int]$p.ProcessId; Name = [string]$p.Name; Path = $shown }
            }
        }
    }
    catch {
        foreach ($p in (Get-Process -ErrorAction SilentlyContinue)) {
            if ($p.Id -eq $PID) { continue }
            $exe = $null
            try { $exe = $p.Path } catch { $exe = $null }
            if ($exe -and (Test-PathInside -Path $exe -Root $AppDir)) {
                $result += [pscustomobject]@{ Id = [int]$p.Id; Name = [string]$p.ProcessName; Path = $exe }
            }
        }
    }
    return , $result
}

function Remove-AppItem {
    # Removes a file or folder INSIDE the app folder, with retries and a long-path fallback.
    param([string]$Path, [string]$Label, [string]$AppDir)
    if (-not (Test-PathInside -Path $Path -Root $AppDir) -or
        ([System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')).Equals($AppDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Internal error: refusing to remove '$Path' (not inside the app folder)."
    }
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Log "$Label - already gone" -Level Detail
        return
    }
    $lastError = $null
    for ($i = 1; $i -le 3; $i++) {
        try {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
        }
        catch { $lastError = $_.Exception.Message }
        if (-not (Test-Path -LiteralPath $Path)) { break }
        # Fallback for paths longer than 260 characters and read-only leftovers.
        if (Test-Path -LiteralPath $Path -PathType Container) {
            $cmdExe = Join-Path $env:SystemRoot 'System32\cmd.exe'
            try {
                $null = Start-Process -FilePath $cmdExe -ArgumentList ('/d /c rd /s /q "\\?\' + $Path + '"') -WindowStyle Hidden -Wait -PassThru
            }
            catch { $lastError = $_.Exception.Message }
        }
        if (-not (Test-Path -LiteralPath $Path)) { break }
        Start-Sleep -Seconds (2 * $i)
    }
    if (Test-Path -LiteralPath $Path) {
        $script:Failed.Add("$Label ($Path): $lastError")
        Write-Log "$Label could not be removed: $lastError" -Level Error
    }
    else {
        $script:Removed.Add($Label)
        Write-Log "$Label removed" -Level Ok
    }
}

function Get-ShortcutTarget {
    param([string]$LinkPath)
    $shell = New-Object -ComObject WScript.Shell
    try { return [string]$shell.CreateShortcut($LinkPath).TargetPath }
    finally { $null = [System.Runtime.InteropServices.Marshal]::ReleaseComObject($shell) }
}

function Test-SafeFolderToDelete {
    # Refuses obviously wrong targets for -RemoveAppFolder.
    param([string]$Folder)
    $full = [System.IO.Path]::GetFullPath($Folder).TrimEnd('\', '/')
    if ($full.Length -le 3) { return 'it is the root of a drive' }
    $special = @($env:USERPROFILE, $env:SystemRoot, $env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:APPDATA, $env:LOCALAPPDATA)
    foreach ($sf in @('Desktop', 'MyDocuments', 'UserProfile', 'Windows', 'ProgramFiles')) {
        try { $special += [Environment]::GetFolderPath($sf) } catch { $null = $_ }
    }
    if ($env:USERPROFILE) { $special += (Join-Path $env:USERPROFILE 'Downloads') }
    foreach ($s in $special) {
        if ($s -and $full.Equals(([string]$s).TrimEnd('\', '/'), [System.StringComparison]::OrdinalIgnoreCase)) {
            return "it is a system or user folder ($s)"
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $full 'uninstall.ps1'))) { return 'uninstall.ps1 is not in it' }
    if (-not ((Test-Path -LiteralPath (Join-Path $full 'voicefilter')) -or (Test-Path -LiteralPath (Join-Path $full 'install.ps1')))) {
        return 'it does not look like the Voice Filter folder'
    }
    return $null
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function Invoke-Main {
    if ($env:OS -ne 'Windows_NT') { Write-Log 'This uninstaller only runs on Windows.' -Level Error; return 1 }
    $AppDir = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
    $manifestPath = Join-Path $AppDir 'install-manifest.json'
    $manifest = Read-JsonFile $manifestPath
    $config = Read-JsonFile (Join-Path $AppDir 'config.json')
    $port = Get-ObjectValue $manifest @('port') $null
    if ($null -eq $port) { $port = Get-ObjectValue $config @('port') 8765 }
    $port = [int]$port

    Write-Log 'Voice Filter uninstaller' -Level Step
    Write-Log "App folder: $AppDir"
    if ($null -eq $manifest) { Write-Log 'No install-manifest.json found; removing whatever is present.' -Level Detail }

    # --- what is inside the folder -----------------------------------------
    $internal = New-Object System.Collections.Generic.List[object]
    $addInternal = {
        param($rel, $label)
        $internal.Add([pscustomobject]@{ Path = (Join-Path $AppDir $rel); Label = $label })
    }
    & $addInternal '.venv' '.venv (Python packages)'
    & $addInternal 'runtime' 'runtime (uv, Python, portable ffmpeg, caches)'
    if (-not $KeepModels) { & $addInternal 'models' 'models (downloaded AI models)' }
    & $addInternal 'data' 'data (jobs, uploads, processed videos, logs, temp)'
    & $addInternal 'start.bat' 'start.bat'
    & $addInternal 'start.ps1' 'start.ps1'
    & $addInternal 'config.json' 'config.json'
    & $addInternal 'voicefilter.env' 'voicefilter.env'

    Write-Log ''
    Write-Log 'This will remove (inside the app folder):'
    $anyInternal = $false
    foreach ($it in $internal) {
        if (Test-Path -LiteralPath $it.Path) {
            $anyInternal = $true
            Write-Log ('  - {0}  [{1}]' -f $it.Label, (Format-Size (Get-ItemSize $it.Path)))
        }
    }
    Write-Log '  - __pycache__ folders and install-manifest.json'
    if (-not $anyInternal) { Write-Log '  (nothing of the installation is left inside the folder)' -Level Detail }
    if ($KeepModels) { Write-Log '  models\ is KEPT (-KeepModels)' -Level Detail }
    $dataJobs = Join-Path $AppDir 'data\jobs'
    if (Test-Path -LiteralPath $dataJobs) {
        Write-Log '  Note: data\jobs contains your processed videos. Download what you want to keep first.' -Level Warn
    }
    Write-Log 'and outside the folder: the Voice Filter shortcuts and the firewall rule.'
    Write-Log 'Shared things (caches outside the folder, network settings) are only changed if you agree.'
    Write-Log ''
    if (-not (Read-Choice -Prompt 'Uninstall Voice Filter?' -Default $true)) {
        Write-Log 'Cancelled; nothing was changed.'
        return 1
    }

    # --- 1. stop running server -------------------------------------------
    Write-Section 'Stopping Voice Filter'
    $procs = Get-AppProcesses -AppDir $AppDir
    if (@($procs).Count -eq 0) { Write-Log 'Voice Filter is not running.' -Level Detail }
    # Python first, then the start.bat windows.
    $ordered = @(@($procs) | Where-Object { $_.Name -ne 'cmd.exe' }) + @(@($procs) | Where-Object { $_.Name -eq 'cmd.exe' })
    foreach ($p in $ordered) {
        try {
            Stop-Process -Id $p.Id -Force -ErrorAction Stop
            Write-Log "Stopped $($p.Name) (pid $($p.Id), $($p.Path))" -Level Ok
        }
        catch {
            if (Get-Process -Id $p.Id -ErrorAction SilentlyContinue) {
                Write-Log "Could not stop $($p.Name) (pid $($p.Id)): $($_.Exception.Message)" -Level Warn
            }
        }
    }
    if (@($procs).Count -gt 0) { Start-Sleep -Seconds 2 }
    try {
        foreach ($c in @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
            $ownerPath = $null
            try { $ownerPath = (Get-Process -Id $c.OwningProcess -ErrorAction Stop).Path } catch { $ownerPath = $null }
            if ($ownerPath -and (Test-PathInside -Path $ownerPath -Root $AppDir)) {
                Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
                Write-Log "Stopped the process listening on port $port (pid $($c.OwningProcess))." -Level Ok
            }
            else {
                $who = $ownerPath
                if (-not $who) { $who = "pid $($c.OwningProcess)" }
                Write-Log "Port $port is used by another program ($who); it is not part of Voice Filter and was left alone." -Level Detail
            }
        }
    }
    catch { Write-Log "Could not check port $port ($($_.Exception.Message))." -Level Detail }

    # --- 2. find stray caches outside the folder (before models\ is deleted) --
    $stray = New-Object System.Collections.Generic.List[object]
    $hfIds = New-Object System.Collections.Generic.List[string]
    foreach ($n in @(Get-ObjectValue $manifest @('hf_models') @())) { if ($n -and -not $hfIds.Contains([string]$n)) { $hfIds.Add([string]$n) } }
    $ourHub = Join-Path $AppDir 'models\hf\hub'
    if (Test-Path -LiteralPath $ourHub) {
        foreach ($d in (Get-ChildItem -LiteralPath $ourHub -Directory -Filter 'models--*' -ErrorAction SilentlyContinue)) {
            if (-not $hfIds.Contains($d.Name)) { $hfIds.Add($d.Name) }
        }
    }
    $ckptNames = @()
    $ourCkpt = Join-Path $AppDir 'models\torch\hub\checkpoints'
    if (Test-Path -LiteralPath $ourCkpt) { $ckptNames = @(Get-ChildItem -LiteralPath $ourCkpt -File -ErrorAction SilentlyContinue | ForEach-Object { $_.Name }) }
    $hubDirs = @()
    if ($env:USERPROFILE) { $hubDirs += (Join-Path $env:USERPROFILE '.cache\huggingface\hub') }
    foreach ($v in @('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_HOME')) {
        $val = [Environment]::GetEnvironmentVariable($v, 'User')
        if ($val -and -not (Test-PathInside -Path $val -Root $AppDir)) {
            if ($v -eq 'HF_HOME') { $val = Join-Path $val 'hub' }
            $hubDirs += $val
        }
    }
    foreach ($hd in ($hubDirs | Select-Object -Unique)) {
        foreach ($id in $hfIds) {
            $cand = Join-Path $hd $id
            if (Test-Path -LiteralPath $cand) { $stray.Add([pscustomobject]@{ Path = $cand; Why = 'Hugging Face model cache entry for a model Voice Filter uses' }) }
        }
    }
    if ($env:USERPROFILE) {
        $userCkpt = Join-Path $env:USERPROFILE '.cache\torch\hub\checkpoints'
        foreach ($f in $ckptNames) {
            $cand = Join-Path $userCkpt $f
            if (Test-Path -LiteralPath $cand) { $stray.Add([pscustomobject]@{ Path = $cand; Why = 'PyTorch hub checkpoint that Voice Filter uses' }) }
        }
    }
    $uvCache = Get-ObjectValue (Get-ObjectValue $manifest @('uv') $null) @('cache_dir') $null
    if ($uvCache -and -not (Test-PathInside -Path $uvCache -Root $AppDir) -and (Test-Path -LiteralPath $uvCache)) {
        $stray.Add([pscustomobject]@{ Path = $uvCache; Why = 'uv cache used by the installer (may be shared with other uv projects)' })
    }
    $tempDirs = @()
    foreach ($t in @($env:TEMP, $env:TMP, [System.IO.Path]::GetTempPath())) {
        if ($t -and -not (Test-PathInside -Path $t -Root $AppDir)) {
            try { $tempDirs += [System.IO.Path]::GetFullPath($t).TrimEnd('\', '/') } catch { $null = $_ }
        }
    }
    foreach ($td in ($tempDirs | Sort-Object -Unique)) {
        if (Test-Path -LiteralPath $td) {
            foreach ($f in (Get-ChildItem -LiteralPath $td -Filter 'voicefilter*' -Force -ErrorAction SilentlyContinue)) {
                $stray.Add([pscustomobject]@{ Path = $f.FullName; Why = 'temporary file from Voice Filter' })
            }
        }
    }

    # same file reachable through two variables (TEMP and TMP) -> list it once
    $seenStray = @{}
    $uniqueStray = New-Object System.Collections.Generic.List[object]
    foreach ($s in $stray) {
        $key = $s.Path.TrimEnd('\', '/').ToLowerInvariant()
        if (-not $seenStray.ContainsKey($key)) { $seenStray[$key] = $true; $uniqueStray.Add($s) }
    }
    $stray = $uniqueStray

    # --- 3. inside the folder ---------------------------------------------
    Write-Section 'Removing files inside the app folder'
    foreach ($it in $internal) { Remove-AppItem -Path $it.Path -Label $it.Label -AppDir $AppDir }
    foreach ($f in (Get-ChildItem -LiteralPath $AppDir -Filter '.write-test-*' -Force -File -ErrorAction SilentlyContinue)) {
        Remove-AppItem -Path $f.FullName -Label $f.Name -AppDir $AppDir
    }
    $pyc = @(Get-ChildItem -LiteralPath $AppDir -Recurse -Directory -Force -Filter '__pycache__' -ErrorAction SilentlyContinue |
        Where-Object { -not (Test-PathInside -Path $_.FullName -Root (Join-Path $AppDir '.git')) -and
                       -not (Test-PathInside -Path $_.FullName -Root (Join-Path $AppDir 'models')) })
    foreach ($d in $pyc) {
        try { Remove-Item -LiteralPath $d.FullName -Recurse -Force -ErrorAction Stop } catch { $script:Failed.Add("$($d.FullName): $($_.Exception.Message)") }
    }
    if ($pyc.Count -gt 0) { $script:Removed.Add("$($pyc.Count) __pycache__ folder(s)"); Write-Log "$($pyc.Count) __pycache__ folder(s) removed" -Level Ok }
    if ($KeepModels -and (Test-Path -LiteralPath (Join-Path $AppDir 'models'))) {
        $script:Left.Add("models\ in the app folder (kept because of -KeepModels)")
    }

    # --- 4. shortcuts -------------------------------------------------------
    Write-Section 'Shortcuts'
    $links = New-Object System.Collections.Generic.List[string]
    foreach ($l in @(Get-ObjectValue $manifest @('shortcuts') @())) { if ($l) { $links.Add([string]$l) } }
    foreach ($folder in @('Desktop', 'Programs', 'CommonDesktopDirectory', 'CommonPrograms')) {
        try {
            $base = [Environment]::GetFolderPath($folder)
            if ($base) {
                $lp = Join-Path $base "$($script:AppName).lnk"
                if (-not $links.Contains($lp)) { $links.Add($lp) }
            }
        }
        catch { $null = $_ }
    }
    $anyLink = $false
    foreach ($lnk in $links) {
        if ([System.IO.Path]::GetFileName($lnk) -ne "$($script:AppName).lnk") { continue }
        if (-not (Test-Path -LiteralPath $lnk -PathType Leaf)) { continue }
        $anyLink = $true
        $target = $null
        try { $target = Get-ShortcutTarget $lnk } catch { $target = $null }
        if ($target -and -not (Test-PathInside -Path $target -Root $AppDir)) {
            $script:Left.Add("$lnk (points to another Voice Filter copy: $target)")
            Write-Log "Left $lnk - it points to $target, not to this folder." -Level Warn
            continue
        }
        try {
            Remove-Item -LiteralPath $lnk -Force -ErrorAction Stop
            $script:Removed.Add("shortcut $lnk")
            Write-Log "Removed $lnk" -Level Ok
        }
        catch {
            $script:Failed.Add("shortcut $lnk : $($_.Exception.Message)")
            $script:OutsideFailed = $true
            Write-Log "Could not remove $lnk ($($_.Exception.Message))" -Level Error
        }
    }
    if (-not $anyLink) { Write-Log 'No shortcuts found.' -Level Detail }

    # --- 5. firewall rule + network profile (one elevated helper) ------------
    Write-Section 'Firewall rule and network'
    $rules = @()
    try {
        foreach ($pat in $script:FirewallRulePatterns) { $rules += @(Get-NetFirewallRule -DisplayName $pat -ErrorAction SilentlyContinue) }
    }
    catch { Write-Log "Could not read firewall rules ($($_.Exception.Message))." -Level Warn }
    if ($rules.Count -gt 0) { Write-Log ('Firewall rule(s) to remove: ' + (($rules | ForEach-Object { $_.DisplayName }) -join ', ')) }
    else { Write-Log 'No Voice Filter firewall rule found.' -Level Detail }

    $profileChanges = @()
    $nc = Get-ObjectValue $manifest @('network_profile_changed') $null
    if ($null -ne $nc) {
        $ifIndex = Get-ObjectValue $nc @('interface_index') $null
        $ifAlias = Get-ObjectValue $nc @('interface') ''
        $netName = Get-ObjectValue $nc @('name') $ifAlias
        $from = Get-ObjectValue $nc @('from') 'Public'
        $current = $null
        try {
            $current = @(Get-NetConnectionProfile -ErrorAction Stop | Where-Object {
                ($null -ne $ifIndex -and [int]$_.InterfaceIndex -eq [int]$ifIndex) -or ($ifAlias -and $_.InterfaceAlias -eq $ifAlias)
            }) | Select-Object -First 1
        }
        catch { $current = $null }
        if ($null -eq $current) {
            $script:Left.Add("network '$netName' was switched to Private by the installer; it is not connected now, so it was left as it is")
        }
        elseif ("$($current.NetworkCategory)" -ne 'Private') {
            Write-Log "Network '$netName' is already '$($current.NetworkCategory)'." -Level Detail
        }
        else {
            Write-Log "The installer switched the network '$netName' from $from to Private."
            Write-Log 'Private lets devices on this network see this PC; Public is stricter. At home Private is usually right.'
            if (Read-Choice -Prompt "Switch '$netName' back to ${from}?" -Default $false) {
                $profileChanges += [pscustomobject]@{ InterfaceIndex = [int]$current.InterfaceIndex; Name = $netName; To = $from }
            }
            else {
                $script:Left.Add("network '$netName' stays Private (your choice; change it in Settings > Network & internet)")
            }
        }
    }

    if ($rules.Count -gt 0 -or $profileChanges.Count -gt 0) {
        $tmpDir = [System.IO.Path]::GetTempPath()
        $resultFile = Join-Path $tmpDir ('vf-uninstall-' + [guid]::NewGuid().ToString('N') + '.json')
        $text = New-ElevatedUninstallText -Patterns $script:FirewallRulePatterns -ProfileChanges $profileChanges -ResultFile $resultFile
        $res = Invoke-Elevated -ScriptText $text -ResultFile $resultFile
        if ($null -eq $res) {
            $script:OutsideFailed = $true
            if ($rules.Count -gt 0) { $script:Failed.Add('firewall rule (no administrator permission) - run uninstall.bat again and accept the prompt') }
            if ($profileChanges.Count -gt 0) { $script:Failed.Add('network profile (no administrator permission)') }
        }
        else {
            foreach ($rn in @($res.rules_removed)) { if ($rn) { $script:Removed.Add("firewall rule '$rn'"); Write-Log "Removed firewall rule '$rn'" -Level Ok } }
            if ($res.rule_error) {
                $script:OutsideFailed = $true
                $script:Failed.Add("firewall rule: $($res.rule_error)")
                Write-Log "Removing the firewall rule failed: $($res.rule_error)" -Level Error
            }
            foreach ($pc in $profileChanges) {
                $item = @($res.profiles | Where-Object { [int]$_.index -eq $pc.InterfaceIndex }) | Select-Object -First 1
                if ($null -ne $item -and $item.ok) {
                    $script:Removed.Add("network '$($pc.Name)' switched back to $($pc.To)")
                    Write-Log "Network '$($pc.Name)' is $($pc.To) again." -Level Ok
                }
                else {
                    $err = 'unknown error'
                    if ($null -ne $item) { $err = $item.error }
                    $script:OutsideFailed = $true
                    $script:Failed.Add("network '$($pc.Name)': $err")
                }
            }
        }
    }

    # --- 6. stray caches outside the folder ---------------------------------
    Write-Section 'Caches outside the app folder'
    if ($stray.Count -eq 0) {
        Write-Log 'None found.' -Level Detail
    }
    else {
        Write-Log 'Found these items outside the app folder that may come from Voice Filter:'
        foreach ($s in $stray) { Write-Log ('  - {0}  [{1}]  ({2})' -f $s.Path, (Format-Size (Get-ItemSize $s.Path)), $s.Why) }
        Write-Log 'Other programs may use the same files.' -Level Detail
        if (Read-Choice -Prompt 'Remove them?' -Default $false) {
            foreach ($s in $stray) {
                if (-not (Test-Path -LiteralPath $s.Path)) { continue }
                try {
                    Remove-Item -LiteralPath $s.Path -Recurse -Force -ErrorAction Stop
                    $script:Removed.Add($s.Path)
                    Write-Log "Removed $($s.Path)" -Level Ok
                }
                catch { $script:Failed.Add("$($s.Path): $($_.Exception.Message)") }
            }
        }
        else {
            foreach ($s in $stray) { $script:Left.Add("$($s.Path) (outside the app folder; not confirmed)") }
        }
    }

    # --- 7. shared tools --------------------------------------------------------
    $ff = Get-ObjectValue $manifest @('ffmpeg') $null
    $ffSource = Get-ObjectValue $ff @('source') $null
    if ($ffSource -eq 'pre-existing') {
        $script:Left.Add("ffmpeg at $(Get-ObjectValue $ff @('path') '?') (it was already installed before Voice Filter; left alone)")
    }
    $script:Left.Add('No system-wide Python or ffmpeg was installed by Voice Filter, so none is removed.')

    # --- 8. manifest ------------------------------------------------------------
    if ($script:OutsideFailed) {
        $script:Left.Add('install-manifest.json (kept so that a second uninstall run can finish the steps that failed)')
    }
    else {
        Remove-AppItem -Path $manifestPath -Label 'install-manifest.json' -AppDir $AppDir
    }

    # --- 9. app folder ------------------------------------------------------------
    $deleteFolder = $false
    $reason = Test-SafeFolderToDelete $AppDir
    if ($RemoveAppFolder -or -not $Yes) {
        Write-Section 'App folder'
        if ($null -ne $reason) {
            Write-Log "The app folder will not be deleted automatically: $reason." -Level Warn
        }
        else {
            $known = @('voicefilter', 'docs', 'eval', 'tests', 'tools', 'models', '.git', '.github', '.gitignore', '.gitattributes',
                       'install.ps1', 'install.bat', 'uninstall.ps1', 'uninstall.bat', 'requirements.txt', 'README.md', 'PLAN.md',
                       'LICENSE', 'LICENSE.txt', 'pyproject.toml', '.pytest_cache', '.editorconfig', 'install-manifest.json')
            $unknown = @(Get-ChildItem -LiteralPath $AppDir -Force -ErrorAction SilentlyContinue | Where-Object { $known -notcontains $_.Name } | ForEach-Object { $_.Name })
            if ($unknown.Count -gt 0) {
                Write-Log ('The folder also contains items that are not part of Voice Filter: ' + ($unknown -join ', ')) -Level Warn
            }
            if ($RemoveAppFolder) {
                $deleteFolder = Read-Choice -Prompt "Delete the whole folder '$AppDir' (source code and everything in it)?" -Default $true
            }
            else {
                $deleteFolder = Read-Choice -Prompt "Also delete the app folder '$AppDir' itself (source code)?" -Default $false
            }
        }
    }

    # --- 10. report -------------------------------------------------------------
    Write-Section 'Result'
    if ($script:Removed.Count -gt 0) {
        Write-Log 'Removed:' -Level Ok
        foreach ($r in $script:Removed) { Write-Log "  - $r" }
    }
    else {
        Write-Log 'Nothing needed to be removed (already uninstalled).' -Level Ok
    }
    if ($script:Left.Count -gt 0) {
        Write-Log 'Left in place:'
        foreach ($l in $script:Left) { Write-Log "  - $l" -Level Detail }
    }
    if ($script:Failed.Count -gt 0) {
        Write-Log 'Could not remove:' -Level Error
        foreach ($f in $script:Failed) { Write-Log "  - $f" -Level Error }
        Write-Log 'Close programs that may use these files (Explorer windows, the Voice Filter window) and run uninstall.bat again.'
    }

    if ($deleteFolder) {
        $waitIds = @($PID)
        try {
            $me = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$PID" -ErrorAction Stop
            $parent = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$($me.ParentProcessId)" -ErrorAction Stop
            if ($null -ne $parent -and $parent.Name -eq 'cmd.exe') { $waitIds += [int]$parent.ProcessId }
        }
        catch { $null = $_ }
        $text = New-FolderRemovalText -Folder $AppDir -WaitIds $waitIds
        $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($text))
        Set-Location -LiteralPath $env:SystemRoot
        [Environment]::CurrentDirectory = $env:SystemRoot
        $null = Start-Process -FilePath (Get-PowerShellExe) -WindowStyle Hidden -WorkingDirectory $env:SystemRoot `
            -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -EncodedCommand ' + $encoded)
        Write-Log ''
        Write-Log "The folder '$AppDir' will be deleted as soon as this window closes." -Level Ok
    }
    elseif (-not $RemoveAppFolder) {
        Write-Log ''
        Write-Log "The app folder itself ('$AppDir') was kept. Delete it yourself if you no longer need it." -Level Detail
    }

    if ($script:Failed.Count -gt 0) { return 1 }
    Write-Log ''
    Write-Log 'Voice Filter has been uninstalled.' -Level Ok
    return 0
}

if ($MyInvocation.InvocationName -ne '.') {
    $code = 1
    try { $code = Invoke-Main }
    catch {
        Write-Log "Unexpected error: $($_.Exception.Message)" -Level Error
        Write-Log ($_.ScriptStackTrace -replace "`r?`n", ' | ') -Level Detail
        Write-Log 'Run uninstall.bat again; it continues where it stopped.'
        $code = 1
    }
    exit $code
}
