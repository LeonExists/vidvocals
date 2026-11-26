<#
.SYNOPSIS
    Static checks + unit tests + mocked end-to-end runs for install.ps1 / uninstall.ps1.

.DESCRIPTION
    Runs on PowerShell 7 on Linux (or Windows). It cannot prove Windows behaviour, but it
    catches syntax errors, PowerShell-7-only syntax, StrictMode mistakes, quoting bugs and
    install/uninstall asymmetries.

      pwsh -NoProfile -File tools/test_scripts/Test-InstallScripts.ps1 [-SkipE2E]

    Exit code 0 = all checks passed.
#>
param([switch]$SkipE2E)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$InstallPs1 = Join-Path $Repo 'install.ps1'
$UninstallPs1 = Join-Path $Repo 'uninstall.ps1'
$script:Failures = New-Object System.Collections.Generic.List[string]
$script:Passed = 0

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if ($Condition) { $script:Passed++ }
    else { $script:Failures.Add($Message); Write-Host "FAIL: $Message" -ForegroundColor Red }
}
function Assert-Equal {
    param($Expected, $Actual, [string]$Message)
    $e = ($Expected | ConvertTo-Json -Compress -Depth 5)
    $a = ($Actual | ConvertTo-Json -Compress -Depth 5)
    Assert-True ($e -eq $a) "$Message (expected $e, got $a)"
}

# ---------------------------------------------------------------------------
# A. Static checks
# ---------------------------------------------------------------------------
Write-Host '--- static checks' -ForegroundColor Cyan
$ps7OnlyCommandParams = @(
    @{ Cmd = 'ConvertFrom-Json'; Param = 'AsHashtable' }, @{ Cmd = 'ConvertFrom-Json'; Param = 'Depth' },
    @{ Cmd = 'Join-Path'; Param = 'AdditionalChildPath' }, @{ Cmd = 'Invoke-WebRequest'; Param = 'SkipCertificateCheck' },
    @{ Cmd = 'Invoke-WebRequest'; Param = 'Resume' }, @{ Cmd = 'Get-Content'; Param = 'AsByteStream' },
    @{ Cmd = 'ForEach-Object'; Param = 'Parallel' }, @{ Cmd = 'Test-Path'; Param = 'IsValid' },
    @{ Cmd = 'Start-Process'; Param = 'Environment' }, @{ Cmd = 'Get-ChildItem'; Param = 'FollowSymlink' }
)
$ps7OnlyCommands = @('Join-String', 'Get-Error', 'Test-Json', 'ConvertTo-CliXml', 'Get-Uptime', 'Remove-Alias', 'Clear-RecycleBin')
$ps7OnlyVariables = @('IsWindows', 'IsLinux', 'IsMacOS', 'IsCoreCLR', 'PSStyle')
$literalPathCmds = @('Test-Path', 'Get-Item', 'Remove-Item', 'Copy-Item', 'Move-Item', 'Get-ChildItem', 'Get-Content',
                     'Set-Content', 'Add-Content', 'Expand-Archive', 'Push-Location', 'Set-Location', 'Resolve-Path', 'Out-File')
$aliasNames = @('%', '?', 'ls', 'dir', 'cat', 'cd', 'cp', 'mv', 'rm', 'rmdir', 'del', 'echo', 'gci', 'gi', 'gc', 'sc', 'select',
                'where', 'foreach', 'sort', 'measure', 'iwr', 'irm', 'curl', 'wget', 'ni', 'ri', 'mi', 'ci', 'sls', 'gps', 'ps',
                'kill', 'spps', 'saps', 'start', 'sleep', 'cls', 'clear', 'type', 'pwd', 'gl', 'sl', 'pushd', 'popd', 'md', 'mkdir',
                'iex', 'icm', 'gcm', 'gm', 'ft', 'fl', 'oh', 'h', 'r', 'gal', 'gv', 'sv', 'nv', 'ac', 'tee', 'write', 'diff', 'compare')

$all = @{}
$toParse = @($InstallPs1, $UninstallPs1) + @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.ps1' | ForEach-Object { $_.FullName })
foreach ($file in $toParse) {
    $name = Split-Path -Leaf $file
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$tokens, [ref]$errors)
    Assert-True (@($errors).Count -eq 0) "$name parses without errors: $(@($errors) | ForEach-Object { "$($_.Extent.StartLineNumber): $($_.Message)" })"
    $all[$name] = @{ Ast = $ast; Tokens = $tokens }
}

foreach ($name in @('install.ps1', 'uninstall.ps1')) {
    $file = Join-Path $Repo $name
    $ast = $all[$name].Ast
    $tokens = $all[$name].Tokens
    $bytes = [System.IO.File]::ReadAllBytes($file)
    Assert-True ($bytes.Length -gt 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) "$name starts with a UTF-8 BOM"
    $text = [System.Text.Encoding]::UTF8.GetString($bytes)
    $lfOnly = [regex]::Matches($text, '(?<!\r)\n').Count
    Assert-True ($lfOnly -eq 0) "$name uses CRLF line endings only ($lfOnly bare LF)"

    # PowerShell 7-only syntax
    $bad = @($ast.FindAll({ param($n)
        $t = $n.GetType().Name
        ($t -eq 'TernaryExpressionAst') -or ($t -eq 'PipelineChainAst') -or
        ($n -is [System.Management.Automation.Language.BinaryExpressionAst] -and "$($n.Operator)" -eq 'QuestionQuestion') -or
        ($n -is [System.Management.Automation.Language.AssignmentStatementAst] -and "$($n.Operator)" -eq 'QuestionQuestionEquals') -or
        ($n -is [System.Management.Automation.Language.MemberExpressionAst] -and $n.PSObject.Properties['NullConditional'] -and $n.NullConditional) -or
        ($n -is [System.Management.Automation.Language.IndexExpressionAst] -and $n.PSObject.Properties['NullConditional'] -and $n.NullConditional)
    }, $true))
    Assert-True ($bad.Count -eq 0) "$name has no PS7-only operators: $($bad | ForEach-Object { "$($_.Extent.StartLineNumber): $($_.Extent.Text)" })"
    $badTok = @($tokens | Where-Object { @('QuestionQuestion', 'QuestionQuestionEquals', 'QuestionDot', 'QuestionLBracket', 'AndAnd', 'OrOr') -contains "$($_.Kind)" })
    Assert-True ($badTok.Count -eq 0) "$name has no PS7-only tokens: $($badTok | ForEach-Object { "$($_.Extent.StartLineNumber): $($_.Text)" })"

    $cmds = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true))
    foreach ($c in $cmds) {
        $cn = $c.GetCommandName()
        if (-not $cn) { continue }
        $line = $c.Extent.StartLineNumber
        Assert-True (-not ($aliasNames -contains $cn.ToLowerInvariant())) "$name line ${line}: alias '$cn' used"
        Assert-True (-not ($ps7OnlyCommands -contains $cn)) "$name line ${line}: PS7-only command '$cn'"
        $params = @($c.CommandElements | Where-Object { $_ -is [System.Management.Automation.Language.CommandParameterAst] } | ForEach-Object { $_.ParameterName })
        foreach ($pp in $ps7OnlyCommandParams) {
            if ($cn -eq $pp.Cmd -and ($params -contains $pp.Param)) { Assert-True $false "$name line ${line}: $cn -$($pp.Param) is PS6+ only" }
        }
        if ($literalPathCmds -contains $cn) {
            # A path given positionally or with -Path must not be a variable (wildcards!) - use -LiteralPath.
            $els = @($c.CommandElements)
            for ($i = 1; $i -lt $els.Count; $i++) {
                $e = $els[$i]
                if ($e -is [System.Management.Automation.Language.CommandParameterAst] -and $e.ParameterName -eq 'Path') {
                    Assert-True $false "$name line ${line}: $cn -Path (use -LiteralPath)"
                }
            }
            $first = $null
            if ($els.Count -gt 1) { $first = $els[1] }
            if ($null -ne $first -and -not ($first -is [System.Management.Automation.Language.CommandParameterAst]) -and
                -not ($cn -eq 'Set-Location' -or $cn -eq 'Push-Location')) {
                Assert-True $false "$name line ${line}: $cn with a positional path (use -LiteralPath): $($c.Extent.Text)"
            }
        }
        if ($cn -eq 'Out-File' -or $cn -eq 'Set-Content' -or $cn -eq 'Add-Content') {
            Assert-True $false "$name line ${line}: $cn (encoding differs between PS 5.1 and 7; use [IO.File]::WriteAllText)"
        }
    }
    $vars = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.VariableExpressionAst] }, $true))
    foreach ($v in $vars) {
        $vn = $v.VariablePath.UserPath
        Assert-True (-not ($ps7OnlyVariables -contains $vn)) "$name line $($v.Extent.StartLineNumber): PS6+ variable `$$vn"
    }
    # Every script parameter is used.
    $paramNames = @($ast.ParamBlock.Parameters | ForEach-Object { $_.Name.VariablePath.UserPath })
    foreach ($pn in $paramNames) {
        $uses = @($vars | Where-Object { $_.VariablePath.UserPath -eq $pn -and $_.Parent -isnot [System.Management.Automation.Language.ParameterAst] })
        Assert-True ($uses.Count -gt 0) "$name parameter -$pn is used"
    }
    # Every function that is defined is called somewhere (dead code check), except Invoke-Main.
    $defs = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true) | ForEach-Object { $_.Name })
    $called = @($cmds | ForEach-Object { $_.GetCommandName() } | Where-Object { $_ })
    foreach ($d in $defs) {
        Assert-True ($called -contains $d) "$name function $d is used"
    }
    # Native exit codes are checked: every 'Invoke-Native' result goes into a variable or $null.
    Assert-True ($text -notmatch '(?m)^\s*Invoke-Native ') "${name}: no Invoke-Native call ignores its result"
}

# Spec items that must be present (simple text probes).
$inst = [System.IO.File]::ReadAllText($InstallPs1)
foreach ($probe in @('UV_PYTHON_INSTALL_DIR', 'UV_CACHE_DIR', 'UV_TOOL_DIR', 'UV_NO_MODIFY_PATH', 'UV_LINK_MODE', 'UV_PYTHON_INSTALL_REGISTRY',
                     'Tls12', "ProgressPreference = 'SilentlyContinue'", '-UseBasicParsing', '-Verb RunAs', '-Profile Private',
                     'Get-NetConnectionProfile', 'WScript.Shell', 'voicefilter.download', 'voicefilter.selfcheck',
                     'network_profile_changed', 'install-manifest.json', 'torch-constraints.txt', 'torch.cuda.is_available',
                     'gyan.dev', 'BtbN', 'HF_HUB_DISABLE_TELEMETRY', 'PYTHONUTF8', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME')) {
    Assert-True ($inst.Contains($probe)) "install.ps1 mentions $probe"
}

# .bat files
foreach ($bat in @('install.bat', 'uninstall.bat')) {
    $b = [System.IO.File]::ReadAllBytes((Join-Path $Repo $bat))
    Assert-True (@($b | Where-Object { $_ -gt 127 }).Count -eq 0) "$bat is ASCII"
    $t = [System.Text.Encoding]::ASCII.GetString($b)
    Assert-True ([regex]::Matches($t, '(?<!\r)\n').Count -eq 0) "$bat uses CRLF"
    $ps1 = $bat -replace '\.bat$', '.ps1'
    Assert-True ($t.Contains("-NoProfile -ExecutionPolicy Bypass -File `"%~dp0$ps1`" %*")) "$bat calls $ps1 with all arguments"
    Assert-True ($t.Contains('if "%~1"=="" set "VF_PAUSE=1"')) "$bat pauses only when started without arguments"
}

# ---------------------------------------------------------------------------
# B. Unit tests of install.ps1 functions
# ---------------------------------------------------------------------------
Write-Host '--- unit tests (install.ps1)' -ForegroundColor Cyan
$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('vf-test-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$null = New-Item -ItemType Directory -Path $tmpRoot -Force
& {
    . $InstallPs1 -Yes
    $ErrorActionPreference = 'Stop'

    $smi = @(
        '+-----------------------------------------------------------------------------------------+',
        '| NVIDIA-SMI 576.88                 Driver Version: 576.88         CUDA Version: 12.9     |',
        '|-----------------------------------------+------------------------+----------------------+'
    )
    $nv = ConvertFrom-NvidiaSmi $smi
    Assert-True $nv.Present 'nvidia-smi parse: present'
    Assert-Equal '576.88' $nv.Driver 'nvidia-smi parse: driver'
    Assert-Equal '12.9' "$($nv.Cuda)" 'nvidia-smi parse: cuda'
    $nv2 = ConvertFrom-NvidiaSmi @('| NVIDIA-SMI 590.10   Driver Version: 590.10   CUDA Driver Version: 13.1 |')
    Assert-Equal '13.1' "$($nv2.Cuda)" 'nvidia-smi parse: "CUDA Driver Version" variant'
    $nv3 = ConvertFrom-NvidiaSmi @('NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.')
    Assert-True (-not $nv3.Present) 'nvidia-smi parse: failure text'
    Assert-Equal '12.8' "$(Get-CudaFromDriverVersion '572.16')" 'driver 572.16 -> CUDA 12.8'
    Assert-Equal '12.4' "$(Get-CudaFromDriverVersion '552.22')" 'driver 552.22 -> CUDA 12.4'
    Assert-True ($null -eq (Get-CudaFromDriverVersion '531.18')) 'driver 531 -> none'

    Assert-Equal @('cu128', 'cu126', 'cu124') @(Get-TorchCandidateTags -CudaVersion ([version]'12.8') -ForceCpu $false -Override '') 'tags for CUDA 12.8'
    Assert-Equal @('cu129', 'cu128', 'cu126', 'cu124') @(Get-TorchCandidateTags -CudaVersion ([version]'12.9') -ForceCpu $false -Override '') 'tags for CUDA 12.9'
    Assert-Equal @('cu132', 'cu130', 'cu129', 'cu128') @(Get-TorchCandidateTags -CudaVersion ([version]'13.2') -ForceCpu $false -Override '') 'tags for CUDA 13.2'
    Assert-Equal @('cu126', 'cu124') @(Get-TorchCandidateTags -CudaVersion ([version]'12.6') -ForceCpu $false -Override '') 'tags for CUDA 12.6'
    Assert-Equal @('cpu') @(Get-TorchCandidateTags -CudaVersion ([version]'12.2') -ForceCpu $false -Override '') 'tags for CUDA 12.2'
    Assert-Equal @('cpu') @(Get-TorchCandidateTags -CudaVersion $null -ForceCpu $false -Override '') 'tags without GPU'
    Assert-Equal @('cpu') @(Get-TorchCandidateTags -CudaVersion ([version]'12.8') -ForceCpu $true -Override '') 'tags with -Cpu'
    Assert-Equal @('cu126') @(Get-TorchCandidateTags -CudaVersion ([version]'12.8') -ForceCpu $false -Override 'CU126') 'tags with -TorchIndex'
    $one = @(Get-TorchCandidateTags -CudaVersion $null -ForceCpu $false -Override '')
    Assert-Equal 'cpu' $one[0] 'single tag stays a string inside @()'

    Assert-Equal @('fast', 'balanced', 'best') @(Get-PresetList 'fast,balanced,best') 'preset list'
    Assert-Equal @('fast') @(Get-PresetList ' Fast ') 'preset list trimmed/lowercased'
    $threw = $false; try { $null = Get-PresetList 'fast,b@d' } catch { $threw = $true }
    Assert-True $threw 'invalid preset rejected'

    foreach ($s in @("plain", "it's", "Jörg Müller’s ‘Voice’ Filter", "C:\Users\Jörg Müller\Voice Filter", '$(Get-Date) `n')) {
        $lit = ConvertTo-PsLiteral $s
        $back = Invoke-Expression $lit
        Assert-Equal $s $back "ConvertTo-PsLiteral round trip: $s"
    }

    # MSVCRT command line parser used to check ConvertTo-CmdArgument.
    function Split-MsvcrtArgs([string]$line) {
        $out = New-Object System.Collections.Generic.List[string]
        $i = 0; $n = $line.Length
        while ($i -lt $n) {
            while ($i -lt $n -and ($line[$i] -eq ' ' -or $line[$i] -eq "`t")) { $i++ }
            if ($i -ge $n) { break }
            $sb = New-Object System.Text.StringBuilder; $inQ = $false
            while ($i -lt $n) {
                $c = $line[$i]
                if (($c -eq ' ' -or $c -eq "`t") -and -not $inQ) { break }
                if ($c -eq '\') {
                    $bs = 0; while ($i -lt $n -and $line[$i] -eq '\') { $bs++; $i++ }
                    if ($i -lt $n -and $line[$i] -eq '"') {
                        $null = $sb.Append([char]'\', [int][math]::Floor($bs / 2))
                        if ($bs % 2 -eq 1) { $null = $sb.Append('"'); $i++ }
                    }
                    else { $null = $sb.Append([char]'\', $bs) }
                    continue
                }
                if ($c -eq '"') { $inQ = -not $inQ; $i++; continue }
                $null = $sb.Append($c); $i++
            }
            $out.Add($sb.ToString())
        }
        return , $out.ToArray()
    }
    $argsIn = @('pip', 'install', 'C:\Users\Jörg Müller\Voice Filter\.venv\Scripts\python.exe', 'C:\dir with space\', '', 'a"b', 'x\\"y', 'torch==2.9.1+cu128', '--index', 'https://download.pytorch.org/whl/cu128')
    $line = ($argsIn | ForEach-Object { ConvertTo-CmdArgument $_ }) -join ' '
    Assert-Equal $argsIn (Split-MsvcrtArgs $line) 'ConvertTo-CmdArgument round trip through MSVCRT rules'

    Assert-True (Test-PathInside -Path '/a/b/c' -Root '/a/b') 'inside: child'
    Assert-True (Test-PathInside -Path '/a/b' -Root '/a/b/') 'inside: same'
    Assert-True (-not (Test-PathInside -Path '/a/bc' -Root '/a/b')) 'inside: sibling prefix is not inside'
    Assert-True (-not (Test-PathInside -Path '' -Root '/a')) 'inside: empty'
    Assert-Equal 'runtime/ffmpeg/bin/ffmpeg.exe' (Get-RelativeAppPath -Path '/app x/runtime/ffmpeg/bin/ffmpeg.exe' -AppDir '/app x') 'relative app path'
    Assert-True ($null -eq (Get-RelativeAppPath -Path '/usr/bin/ffmpeg' -AppDir '/app x')) 'relative app path outside -> null'

    Assert-True (Test-IsAscii "abc`r`n") 'ascii'
    Assert-True (-not (Test-IsAscii 'Jörg')) 'non-ascii'

    $j = Get-JsonFromOutput @('some log line', '{"torch": "2.9.1+cu128", "cuda": true}')
    Assert-Equal '2.9.1+cu128' $j.torch 'json from output'
    $j2 = Get-JsonFromOutput @('{', '  "a": {"b": 1}', '}')
    Assert-Equal 1 $j2.a.b 'multi-line json from output'
    Assert-True ($null -eq (Get-JsonFromOutput @('no json here'))) 'no json -> null'

    # Native command runner: stderr lines must not throw under ErrorActionPreference=Stop.
    $r = Invoke-Native -FilePath '/bin/sh' -Arguments @('-c', 'echo out; echo err 1>&2; exit 3') -Quiet
    Assert-Equal 3 $r.ExitCode 'Invoke-Native exit code'
    Assert-True ($r.Output -contains 'out' -and $r.Output -contains 'err') 'Invoke-Native captures stdout and stderr'
    $r = Invoke-Native -FilePath '/bin/sh' -Arguments @('-c', 'exit 0') -Live
    Assert-Equal 0 $r.ExitCode 'Invoke-Native -Live exit code 0'
    $r = Invoke-Native -FilePath '/bin/sh' -Arguments @('-c', 'echo "$1|$2"; exit 4', 'x', 'a b', 'Jörg "Q"') -Live
    Assert-Equal 4 $r.ExitCode 'Invoke-Native -Live exit code 4'

    # Environment / start scripts with a nasty path
    $app = Join-Path $tmpRoot "Jörg Müller's (x86) & Voice Filter 100%"
    $null = New-Item -ItemType Directory -Path $app -Force
    $envList = Get-AppEnvironment -PortNumber 8765 -Device 'cuda' -FfmpegPath (Join-Path $app 'runtime/ffmpeg/bin/ffmpeg.exe') -FfprobePath 'C:\Users\Jörg\scoop\shims\ffprobe.exe' -AppDir $app
    $names = @($envList | ForEach-Object { $_.Name })
    foreach ($n in @('VF_DATA_DIR', 'VF_MODELS_DIR', 'HF_HOME', 'TORCH_HOME', 'TEMP', 'TMP', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR',
                     'XDG_CACHE_HOME', 'PYTHONUTF8', 'HF_HUB_DISABLE_TELEMETRY', 'VF_PORT', 'VF_DEVICE', 'VF_FFMPEG', 'VF_FFPROBE')) {
        Assert-True ($names -contains $n) "app environment sets $n"
    }
    $bat = New-StartBatText -EnvList $envList -PortNumber 8765 -FfmpegBinRel 'runtime\ffmpeg\bin'
    Assert-True (Test-IsAscii $bat) 'start.bat text is ASCII even with a non-ASCII ffprobe path'
    Assert-True ([regex]::Matches($bat, '(?<!\r)\n').Count -eq 0) 'start.bat text uses CRLF'
    Assert-True ($bat.Contains('rem VF_FFPROBE: path contains non-ASCII')) 'start.bat skips non-ASCII literal paths'
    Assert-True ($bat.Contains('"%VF_PYTHON%" -m voicefilter')) 'start.bat runs python -m voicefilter'
    Assert-True ($bat.Contains('set "VF_PORT=8765"')) 'start.bat sets the port'
    Assert-True ($bat.Contains('set "HF_HOME=%VF_APP_DIR%\models\hf"')) 'start.bat sets HF_HOME relative to %~dp0'
    Assert-True ($bat.Contains('http://localhost:8765/')) 'start.bat opens the browser'
    Assert-True (-not ($bat -match '(?m)^\s*\(|\(\s*\r?$')) 'start.bat has no parenthesised blocks (paths with ")" would break them)'
    $ps1 = New-StartPs1Text -EnvList $envList -PortNumber 8765 -FfmpegBinRel 'runtime\ffmpeg\bin'
    $t = $null; $e = $null
    $null = [System.Management.Automation.Language.Parser]::ParseInput($ps1, [ref]$t, [ref]$e)
    Assert-True (@($e).Count -eq 0) "generated start.ps1 parses: $(@($e) | ForEach-Object { $_.Message })"
    Assert-True ($ps1.Contains("'C:\Users\Jörg\scoop\shims\ffprobe.exe'")) 'start.ps1 keeps non-ASCII paths'
    # Generated browser poller command must parse too.
    $m = [regex]::Match($bat, '-Command "([^"]+)"')
    Assert-True $m.Success 'start.bat contains the browser poller'
    $null = [System.Management.Automation.Language.Parser]::ParseInput($m.Groups[1].Value, [ref]$t, [ref]$e)
    Assert-True (@($e).Count -eq 0) 'browser poller parses'
    Assert-True (-not $m.Groups[1].Value.Contains('%')) 'browser poller has no % (cmd would expand it)'

    # Set-ProcessEnvironment
    Set-ProcessEnvironment -EnvList $envList -AppDir $app
    Assert-Equal ([System.IO.Path]::Combine($app, 'models\hf')) $env:HF_HOME 'process HF_HOME'
    Assert-Equal $app $env:VF_APP_DIR 'process VF_APP_DIR'

    # Elevated helper script
    $res = Join-Path $app "res ’x’.json"
    $txt = New-ElevatedScriptText -RuleName 'Voice Filter (TCP 8765)' -RemovePatterns @('Voice Filter (TCP *)', 'VoiceFilter (TCP *)') -PortNumber 8765 -CreateRule $true `
        -ProfileChanges @([pscustomobject]@{ InterfaceIndex = 7; To = 'Private' }) -ResultFile $res
    $null = [System.Management.Automation.Language.Parser]::ParseInput($txt, [ref]$t, [ref]$e)
    Assert-True (@($e).Count -eq 0) 'elevated firewall script parses'
    Assert-True ($txt.Contains('-Profile Private') -and $txt.Contains('-LocalPort 8765') -and $txt.Contains('-Direction Inbound')) 'elevated script creates the right rule'
    $enc = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($txt))
    Assert-True ($enc.Length -lt 30000) 'encoded elevated command fits the command line limit'

    # Manifest merge
    $old = [ordered]@{ installed_at = '2026-01-01'; created = @('a', 'b'); shortcuts = @('s1'); network_profile_changed = [ordered]@{ interface = 'WLAN'; from = 'Public'; to = 'Private' }; firewall_rules_created = @('Voice Filter (TCP 8765)') }
    $new = [ordered]@{ installed_at = '2026-02-02'; created = @('b', 'c'); shortcuts = @('s1', 's2'); network_profile_changed = $null; firewall_rules_created = @('Voice Filter (TCP 9000)') }
    $m = Merge-Manifest -Old $old -New $new
    Assert-Equal @('a', 'b', 'c') @($m['created']) 'manifest: created union'
    Assert-Equal @('s1', 's2') @($m['shortcuts']) 'manifest: shortcuts union'
    Assert-Equal '2026-01-01' $m['first_installed_at'] 'manifest: first install time kept'
    Assert-Equal '2026-02-02' $m['installed_at'] 'manifest: installed_at updated'
    Assert-Equal 'WLAN' $m['network_profile_changed']['interface'] 'manifest: network change kept'
    Assert-Equal @('Voice Filter (TCP 8765)', 'Voice Filter (TCP 9000)') @($m['firewall_rules_created']) 'manifest: firewall rules union'
    $m2 = Merge-Manifest -Old $null -New $new
    Assert-Equal '2026-02-02' $m2['first_installed_at'] 'manifest: first install without old'

    # JSON round trip with non-ASCII through the same helpers
    $jf = Join-Path $app 'x.json'
    Write-Utf8File -Path $jf -Text (([ordered]@{ app_dir = $app; list = @('one') } | ConvertTo-Json -Depth 4))
    $back = Read-JsonFile $jf
    Assert-Equal $app $back['app_dir'] 'json round trip keeps non-ASCII path'
    Assert-Equal @('one') @($back['list']) 'json round trip keeps one-element arrays'
    $raw = [System.IO.File]::ReadAllBytes($jf)
    Assert-True (-not ($raw[0] -eq 0xEF)) 'json files are written without BOM (Python json.load would fail)'

    # Steps: soft vs fatal
    $ok = Invoke-Step 'soft test' { throw 'boom' } -Soft
    Assert-True (-not $ok) 'soft step returns false'
    $threw = $false
    try { $null = Invoke-Step 'fatal test' { throw 'boom' } } catch { $threw = ("$($_.Exception.Message)" -like 'STEP_FAILED:*') }
    Assert-True $threw 'fatal step throws STEP_FAILED'
    Assert-True ($script:State.Problems.Count -eq 2) 'failed steps are recorded as problems'
}

Write-Host '--- unit tests (uninstall.ps1)' -ForegroundColor Cyan
& {
    . $UninstallPs1 -Yes
    $ErrorActionPreference = 'Stop'
    $app = Join-Path $tmpRoot 'uninst app'
    $null = New-Item -ItemType Directory -Path (Join-Path $app 'sub') -Force
    [System.IO.File]::WriteAllText((Join-Path $app 'sub/f.txt'), 'x')
    $threw = $false
    try { Remove-AppItem -Path $tmpRoot -Label 'outside' -AppDir $app } catch { $threw = $true }
    Assert-True $threw 'Remove-AppItem refuses a path outside the app folder'
    $threw = $false
    try { Remove-AppItem -Path $app -Label 'self' -AppDir $app } catch { $threw = $true }
    Assert-True $threw 'Remove-AppItem refuses the app folder itself'
    Remove-AppItem -Path (Join-Path $app 'sub') -Label 'sub' -AppDir $app
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $app 'sub'))) 'Remove-AppItem removes a folder'
    Remove-AppItem -Path (Join-Path $app 'sub') -Label 'sub' -AppDir $app
    Assert-True ($script:Failed.Count -eq 0) 'Remove-AppItem on a missing path is fine (idempotent)'
    Assert-True ($null -ne (Test-SafeFolderToDelete '/')) 'refuses to delete a drive root'
    Assert-True ($null -ne (Test-SafeFolderToDelete $app)) 'refuses a folder without uninstall.ps1'
    $txt = New-ElevatedUninstallText -Patterns @('Voice Filter (TCP *)') -ProfileChanges @([pscustomobject]@{ InterfaceIndex = 3; To = 'Public' }) -ResultFile '/tmp/x y.json'
    $t = $null; $e = $null
    $null = [System.Management.Automation.Language.Parser]::ParseInput($txt, [ref]$t, [ref]$e)
    Assert-True (@($e).Count -eq 0) 'elevated uninstall script parses'
    $txt = New-FolderRemovalText -Folder "/tmp/Jörg's ‘dir’" -WaitIds @(1, 2)
    $null = [System.Management.Automation.Language.Parser]::ParseInput($txt, [ref]$t, [ref]$e)
    Assert-True (@($e).Count -eq 0) 'folder removal script parses'

    # Run the folder removal script for real on a scratch folder (waits for a short-lived process).
    $victim = Join-Path $tmpRoot "victim Jörg's dir"
    $null = New-Item -ItemType Directory -Path (Join-Path $victim 'a/b') -Force
    [System.IO.File]::WriteAllText((Join-Path $victim 'a/b/c.txt'), 'x')
    $sleeper = Start-Process -FilePath '/bin/sleep' -ArgumentList '2' -PassThru
    $txt = New-FolderRemovalText -Folder $victim -WaitIds @($sleeper.Id)
    $env:SystemRoot = '/'
    & ([scriptblock]::Create(($txt -replace "Join-Path \`$env:SystemRoot 'System32\\cmd.exe'", "'/bin/true'")))
    Assert-True (-not (Test-Path -LiteralPath $victim)) 'folder removal script deletes the folder after the process exits'
}

# ---------------------------------------------------------------------------
# C. Mocked end-to-end runs
# ---------------------------------------------------------------------------
if (-not $SkipE2E) {
    Write-Host '--- mocked end-to-end runs' -ForegroundColor Cyan
    . (Join-Path $PSScriptRoot 'Invoke-MockedE2E.ps1') -TmpRoot $tmpRoot
}

Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ''
if ($script:Failures.Count -gt 0) {
    Write-Host "$($script:Failures.Count) check(s) FAILED, $($script:Passed) passed." -ForegroundColor Red
    exit 1
}
Write-Host "All $($script:Passed) checks passed." -ForegroundColor Green
exit 0
