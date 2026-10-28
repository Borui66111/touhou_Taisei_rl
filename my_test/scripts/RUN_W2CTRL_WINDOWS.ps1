<#
Validated Windows skill-v2 policy export + (opt-in) live-control bridge.
Needs the project's real local rl_skill_v2 Taisei build and the main branch
run_live_ppo_windows.py. No game is launched unless -Launch is specified.
#>
param(
  [Parameter(Mandatory=$true)][string]$Checkpoint,
  [string]$Policy = "",
  [switch]$Launch,
  [string]$Renderer = "gl33",
  [string]$Resources = "",
  [string]$ExpectedSha256 = ""
)
$ErrorActionPreference = 'Stop'
$scriptFile = Join-Path $PSScriptRoot 'w2ctrl.py'
$root = Split-Path $PSScriptRoot -Parent
$checkpointFull = (Resolve-Path -LiteralPath $Checkpoint).Path
if (-not $Policy) {
  $name = [IO.Path]::GetFileNameWithoutExtension($checkpointFull)
  $Policy = Join-Path (Join-Path $root 'results') "${name}_TSKILL2.bin"
}
$policyFull = [IO.Path]::GetFullPath($Policy)
if (Get-Command py -ErrorAction SilentlyContinue) {
  $python = 'py'
  $prefix = @('-3.9')
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
  $python = 'python'
  $prefix = @()
} else {
  throw 'Python not found; install Python with PyTorch and NumPy.'
}
$argv = @($scriptFile, 'prepare', $checkpointFull, $policyFull)
if ($ExpectedSha256) {$argv += @('--expected-sha256', $ExpectedSha256)}
& $python @prefix @argv
if ($LASTEXITCODE -ne 0) {throw 'Policy export/verification failed; no game launched.'}
if ($Launch) {
  $runArgs = @($scriptFile, 'run', $policyFull, '--renderer', $Renderer)
  if ($Resources) {$runArgs += @('--resources', $Resources)}
  & $python @prefix @runArgs
  if ($LASTEXITCODE -ne 0) {throw 'No verified live-control attribution; inspect the new launcher manifest and game logs.'}
} else {
  Write-Host ('Validated export: ' + $policyFull)
  Write-Host 'To launch the existing Windows skill-v2 game build, rerun with -Launch.'
}
