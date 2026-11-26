# install-earnings-watch.ps1 - make the earnings panel run by itself, forever.
#
# Run this ONCE. After that the watcher starts at every logon, has no console
# window, restarts itself if it ever crashes, and survives reboots.
#
#     powershell -ExecutionPolicy Bypass -File tools\install-earnings-watch.ps1
#     powershell -ExecutionPolicy Bypass -File tools\install-earnings-watch.ps1 -Uninstall
#     powershell -ExecutionPolicy Bypass -File tools\install-earnings-watch.ps1 -NoChart
#
# -NoChart installs it in collect-only mode: it still catches every filing and
# caches the result, but never writes to the chart. Use that if you want the data
# without anything touching TradingView.
#
# WHY A SCHEDULED TASK AND NOT A SERVICE. A Windows service runs in session 0 and
# cannot see the TradingView desktop app's debug port the way a logon-session
# process can. A logon-triggered task runs as the owner, in the owner's session, which is
# the only place the chart is reachable from.

param([switch]$Uninstall, [switch]$NoChart)

$ErrorActionPreference = "Stop"
$TaskName = "EPD Earnings Watch"
$Root     = Split-Path -Parent $PSScriptRoot
$Script   = Join-Path $Root "tools\earnings_watch.py"

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask  -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "removed '$TaskName'"
    } else {
        Write-Host "'$TaskName' was not installed"
    }
    Get-Process pythonw -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.CommandLine -like "*earnings_watch*" } |
        Stop-Process -Force -ErrorAction SilentlyContinue
    exit 0
}

if (-not (Test-Path $Script)) { throw "cannot find $Script" }

# pythonw.exe is python without a console window - that is the whole point here.
$py = (Get-Command python -ErrorAction Stop).Source
$pyw = Join-Path (Split-Path -Parent $py) "pythonw.exe"
if (-not (Test-Path $pyw)) { $pyw = $py }   # fall back rather than fail to install

$argline = "`"$Script`""
if ($NoChart) { $argline += " --no-chart" }

$action  = New-ScheduledTaskAction -Execute $pyw -Argument $argline -WorkingDirectory $Root
# TWO TRIGGERS. At logon, and every five minutes forever.
#
# With MultipleInstances=IgnoreNew the five-minute trigger does nothing while the
# watcher is alive, and starts it again when it is not. That is what recovers it
# after its own watchdog exits a hung process, and after a plain crash - which
# RestartOnFailure does NOT cover, because the scheduler only counts a failure to
# LAUNCH, not a program that launched and later died. The watcher went dark from
# 23 to 29 September 2026 with nothing to bring it back.
#
# It has to be a SEPARATE time trigger. Repetition set on the logon trigger only
# starts counting at the next logon, so it did nothing for the session it was
# added in - measured: killed at 12:51, still dead at 12:59. With this trigger,
# a watcher killed at 13:00:27 was back on its own at 13:04:49.
$trigger = @(
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME),
    (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 5))
)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive `
             -RunLevel Limited

# StartWhenAvailable + a restart policy is what makes this survive a reboot that
# happens while the machine is asleep, and a crash at 4am nobody is awake for.
# Priority 4 = NORMAL. The Task Scheduler default (7) runs the watcher BELOW normal,
# and on a busy desktop Windows starved it: on 2026-09-30 a 6-second EDGAR sweep
# took 36-79 seconds and a 0.04-second chart read took 27, so passes landed 3.5
# minutes apart instead of 25 seconds - a print could reach the panel minutes late.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) -MultipleInstances IgnoreNew -Priority 4

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Watches SEC EDGAR for Item 2.02 8-Ks and renders the result in the EPD chart panel." | Out-Null

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3

$t = Get-ScheduledTask -TaskName $TaskName
$i = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Host ""
Write-Host "installed and started: '$TaskName'"
Write-Host "  state        $($t.State)"
Write-Host "  last result  $($i.LastTaskResult)   (267009 = currently running)"
Write-Host "  runs         $pyw $argline"
Write-Host "  log          $Root\tools\.watch\watch.log"
Write-Host ""
Write-Host "check what it has found:   python tools\earnings_watch.py --status"
