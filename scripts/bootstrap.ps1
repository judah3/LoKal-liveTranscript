param(
  [string]$Python = ""
)

$ErrorActionPreference = "Stop"

function Require-Command {
  param([string]$Name, [string]$Hint)
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "$Name is not available in PATH. $Hint"
  }
}

function Resolve-PythonLauncher {
  param([string]$Preferred)

  if ($Preferred) {
    $preferredCmd = Get-Command $Preferred -ErrorAction SilentlyContinue
    if ($preferredCmd) {
      return @{ Bin = $Preferred; Args = @() }
    }
  }

  $compatibleVersions = @("3.12", "3.11", "3.10")
  $pyCmd = Get-Command py -ErrorAction SilentlyContinue
  if ($pyCmd) {
    foreach ($version in $compatibleVersions) {
      & py "-$version" -c "import sys" 2>$null | Out-Null
      if ($LASTEXITCODE -eq 0) {
        return @{ Bin = "py"; Args = @("-$version") }
      }
    }
  }

  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCmd) {
    $version = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null).Trim()
    if ($LASTEXITCODE -eq 0 -and $compatibleVersions -contains $version) {
      return @{ Bin = "python"; Args = @() }
    }
  }

  throw "Compatible Python launcher not found. Install Python 3.10-3.12 and ensure 'py -3.12' or 'python' is available in PATH."
}

Require-Command -Name npm -Hint "Install Node.js 20+."
Require-Command -Name dotnet -Hint "Install .NET SDK 8.0+."

$pythonLauncher = Resolve-PythonLauncher -Preferred $Python
$pythonBin = [string]$pythonLauncher.Bin
$pythonArgs = [string[]]$pythonLauncher.Args
$pythonImportCheck = "import numpy, websockets, faster_whisper, langdetect"
$pythonPackages = @(
  "numpy==2.2.1",
  "websockets==12.0",
  "faster-whisper==1.1.1",
  "langdetect==1.0.9"
)

Write-Host "[1/3] Installing Electron dependencies..."
Push-Location "$PSScriptRoot\..\electron-app"
npm install
Pop-Location

Write-Host "[2/3] Installing Python engine dependencies..."
Push-Location "$PSScriptRoot\..\python-engine"
if (-not (Test-Path .venv)) {
  Write-Host "Creating Python venv with '$pythonBin'..."
  & $pythonBin @pythonArgs -m venv .venv
}

$venvPython = Join-Path (Resolve-Path ".venv").Path "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Python venv is missing '$venvPython'."
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install @pythonPackages
& $venvPython -m pip install -e .
& $venvPython -c $pythonImportCheck
Pop-Location

Write-Host "[3/3] Restoring C# dependencies..."
Push-Location "$PSScriptRoot\..\audio-service-csharp\src"
dotnet restore
Pop-Location

Write-Host "Bootstrap complete."
