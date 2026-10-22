# Windows PowerShell 5.1+ helper. Requires WSL2 Ubuntu installed beforehand.
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Write-Host 'Taisei RL: using Ubuntu on WSL2 for the native Linux simulation library.'
Write-Host 'If WSL is missing: open an Administrator PowerShell and run: wsl --install -d Ubuntu; then restart.'
& wsl.exe -- bash -lc 'uname -s >/dev/null && command -v python3 >/dev/null'
if ($LASTEXITCODE -ne 0) { throw 'WSL2 Ubuntu is not ready. Run wsl --install -d Ubuntu first, reboot if prompted.' }
$linuxPath = (& wsl.exe -- wslpath -u $project | Select-Object -First 1).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxPath) { throw 'Cannot translate Windows path to WSL path' }
# Quote a path that may contain spaces (single quotes in directory names supported).
$escaped = $linuxPath.Replace("'", "'\''")
$command = "set -e; mkdir -p ~/touhou-taisei-rl; cp -a '$escaped'/.' ~/touhou-taisei-rl/; cd ~/touhou-taisei-rl; bash scripts/setup_ubuntu_wsl.sh"
Write-Host 'Copying this project into the faster WSL Linux filesystem, then building Taisei.'
& wsl.exe -- bash -lc $command
if ($LASTEXITCODE -ne 0) { throw "WSL2 setup failed ($LASTEXITCODE). Review the Linux error above." }
Write-Host 'Native setup complete. Launch Ubuntu and run: cd ~/touhou-taisei-rl'
