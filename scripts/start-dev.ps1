$ErrorActionPreference = "Stop"
if (Test-Path Env:ELECTRON_RUN_AS_NODE) {
  Remove-Item Env:ELECTRON_RUN_AS_NODE
}

$repoRoot = Resolve-Path "$PSScriptRoot\.."
$pythonEngineDir = Join-Path $repoRoot "python-engine"
$electronDir = Join-Path $repoRoot "electron-app"
$venvPython = Join-Path $pythonEngineDir ".venv\Scripts\python.exe"
$devPorts = @(5173, 8765)
$pythonImportCheck = "import numpy, websockets, faster_whisper, langdetect"
$pythonPackages = @(
  "numpy==2.2.1",
  "websockets==12.0",
  "faster-whisper==1.1.1",
  "langdetect==1.0.9"
)

if (-not (Test-Path $venvPython)) {
  Write-Host "Python venv not found. Running bootstrap..."
  & (Join-Path $repoRoot "scripts\bootstrap.ps1")
}

Push-Location $pythonEngineDir
$needsPythonDeps = $false
try {
  & $venvPython -c $pythonImportCheck | Out-Null
} catch {
  $needsPythonDeps = $true
}

if ($needsPythonDeps) {
  Write-Host "Python dependencies missing. Installing..."
  & $venvPython -m pip install --upgrade pip
  & $venvPython -m pip install @pythonPackages
  & $venvPython -m pip install -e .
  & $venvPython -c $pythonImportCheck | Out-Null
}
Pop-Location

Push-Location $electronDir
if (-not (Test-Path "node_modules")) {
  Write-Host "Electron dependencies missing. Installing..."
  npm install
}

foreach ($devPort in $devPorts) {
  $listeners = Get-NetTCPConnection -LocalPort $devPort -State Listen -ErrorAction SilentlyContinue
  if ($listeners) {
    $processIds = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $processIds) {
      if ($procId -and $procId -ne 0) {
        Write-Host "Port $devPort is in use by PID $procId. Stopping process..."
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
      }
    }
  }
}

npm run dev
Pop-Location
