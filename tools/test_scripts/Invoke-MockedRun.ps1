<#
    Child process for Invoke-MockedE2E.ps1: loads install.ps1 or uninstall.ps1 from a test
    app folder (dot-sourced, so only the functions are defined), replaces everything that
    needs Windows (firewall, network profiles, UAC, shortcuts, downloads) with mocks that
    keep their state in a JSON file, runs Invoke-Main and prints EXITCODE=<n>.
#>
param(
    [Parameter(Mandatory = $true)][string]$AppDir,
    [Parameter(Mandatory = $true)][ValidateSet('install', 'uninstall')][string]$Script,
    [Parameter(Mandatory = $true)][string]$StateFile,
    [string]$ArgsJson = '{}',
    [string]$Answers = ''
)
$ErrorActionPreference = 'Stop'
$env:OS = 'Windows_NT'
if (-not $env:SystemRoot) { $env:SystemRoot = '/' }

$splat = @{}
$parsed = $ArgsJson | ConvertFrom-Json
foreach ($p in $parsed.PSObject.Properties) { $splat[$p.Name] = $p.Value }

$global:MockStateFile = $StateFile
$global:MockAnswers = New-Object System.Collections.Generic.Queue[string]
foreach ($a in ($Answers -split '\|')) { if ($Answers -ne '') { $global:MockAnswers.Enqueue($a) } }

function global:Get-MockState { return (Get-Content -LiteralPath $global:MockStateFile -Raw | ConvertFrom-Json) }
function global:Save-MockState($s) { [System.IO.File]::WriteAllText($global:MockStateFile, ($s | ConvertTo-Json -Depth 6)) }
function global:Add-MockEvent([string]$e) { $s = Get-MockState; $s.events = @($s.events) + @($e); Save-MockState $s }

function global:Read-Host {
    param([Parameter(Position = 0)]$Prompt)
    $ans = ''
    if ($global:MockAnswers.Count -gt 0) { $ans = $global:MockAnswers.Dequeue() }
    Add-MockEvent "prompt: $Prompt -> '$ans'"
    Write-Host "$Prompt : $ans"
    return $ans
}
function global:Get-NetFirewallRule {
    [CmdletBinding()] param([string]$DisplayName)
    $s = Get-MockState
    foreach ($r in @($s.rules)) { if ($null -ne $r -and $r.DisplayName -like $DisplayName) { $r } }
}
function global:Get-NetFirewallPortFilter {
    [CmdletBinding()] param([Parameter(ValueFromPipeline = $true)]$InputObject)
    process { [pscustomobject]@{ Protocol = $InputObject.Protocol; LocalPort = $InputObject.LocalPort } }
}
function global:Remove-NetFirewallRule {
    [CmdletBinding()] param([Parameter(ValueFromPipeline = $true)]$InputObject)
    process {
        $s = Get-MockState
        $s.rules = @(@($s.rules) | Where-Object { $null -ne $_ -and $_.DisplayName -ne $InputObject.DisplayName })
        Save-MockState $s
        Add-MockEvent "firewall remove: $($InputObject.DisplayName)"
    }
}
function global:New-NetFirewallRule {
    [CmdletBinding()] param($DisplayName, $Description, $Direction, $Protocol, $LocalPort, $Action, $Profile, $Enabled)
    $s = Get-MockState
    $s.rules = @(@($s.rules) | Where-Object { $null -ne $_ }) + @([pscustomobject]@{
        DisplayName = $DisplayName; Direction = "$Direction"; Protocol = "$Protocol"; LocalPort = "$LocalPort"
        Action = "$Action"; Profile = "$Profile"; Enabled = "$Enabled" })
    Save-MockState $s
    Add-MockEvent "firewall new: $DisplayName"
}
function global:Get-NetConnectionProfile {
    [CmdletBinding()] param()
    $s = Get-MockState
    [pscustomobject]@{ InterfaceIndex = 7; InterfaceAlias = 'WLAN'; Name = 'HomeWifi'; NetworkCategory = $s.category; IPv4Connectivity = 'Internet' }
}
function global:Set-NetConnectionProfile {
    [CmdletBinding()] param($InterfaceIndex, $NetworkCategory)
    $s = Get-MockState
    $s.category = "$NetworkCategory"
    Save-MockState $s
    Add-MockEvent "profile $InterfaceIndex -> $NetworkCategory"
}

. (Join-Path $AppDir "$Script.ps1") @splat

# --- overrides of script functions (defined after dot-sourcing, so they win) ---
function Invoke-Elevated {
    param([string]$ScriptText, [string]$ResultFile)
    Add-MockEvent 'elevated'
    & ([scriptblock]::Create($ScriptText))
    if (Test-Path -LiteralPath $ResultFile) {
        try { return ([System.IO.File]::ReadAllText($ResultFile) | ConvertFrom-Json) }
        finally { Remove-Item -LiteralPath $ResultFile -Force }
    }
    throw 'mock: no result file'
}
if ($Script -eq 'install') {
    function Get-WindowsVersion { return [version]'10.0.22631.0' }
    function Get-LanAddresses { return , @('192.168.1.23') }
    function Get-ShortcutPaths {
        return @((Join-Path $env:VF_MOCK_HOME 'Desktop/Voice Filter.lnk'), (Join-Path $env:VF_MOCK_HOME 'Start Menu/Programs/Voice Filter.lnk'))
    }
    function New-Shortcut {
        param([string]$LinkPath, [string]$Target, [string]$WorkingDirectory, [string]$Description, [string]$Icon)
        New-Directory (Split-Path -Parent $LinkPath)
        [System.IO.File]::WriteAllText($LinkPath, $Target)
    }
    function Invoke-Download {
        param([string]$Url, [string]$OutFile, [int]$Attempts = 3)
        Add-MockEvent "download: $Url"
        if ($Url -like '*gyan.dev*' -and $env:VF_MOCK_GYAN_FAIL -eq '1') { throw "Could not download $Url (mock failure)" }
        $src = $null
        if ($Url -like '*uv-x86_64*') { $src = Join-Path $env:VF_MOCK_FIXTURES 'uv.zip' }
        elseif ($Url -like '*ffmpeg*') { $src = Join-Path $env:VF_MOCK_FIXTURES 'ffmpeg.zip' }
        if (-not $src) { throw "mock: unexpected download $Url" }
        Copy-Item -LiteralPath $src -Destination $OutFile -Force
    }
}
else {
    function Get-ShortcutTarget { param([string]$LinkPath) return [System.IO.File]::ReadAllText($LinkPath) }
}

$code = Invoke-Main
Write-Output "EXITCODE=$code"
