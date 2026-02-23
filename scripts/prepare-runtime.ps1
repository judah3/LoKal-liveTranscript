param(
  [switch]$SkipPython,
  [switch]$SkipAudio
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExecutable {
  $compatibleVersions = @("3.12", "3.11", "3.10")
  $pyCmd = Get-Command py -ErrorAction SilentlyContinue
  if ($pyCmd) {
    foreach ($version in $compatibleVersions) {
      $resolved = (& py "-$version" -c "import sys; print(sys.executable)" 2>$null).Trim()
      if ($LASTEXITCODE -eq 0 -and $resolved -and (Test-Path $resolved)) {
        return $resolved
      }
    }
  }

  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCmd) {
    $resolved = (& python -c "import sys; print(sys.executable)" 2>$null).Trim()
    if ($LASTEXITCODE -eq 0 -and $resolved -and (Test-Path $resolved)) {
      $resolvedVersion = (& $resolved -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null).Trim()
      if ($LASTEXITCODE -eq 0 -and $compatibleVersions -contains $resolvedVersion) {
        return $resolved
      }
    }
  }

  throw "Python 3.10-3.12 is required to prepare runtime (3.13 is not supported by current pinned deps). Install Python 3.12 and ensure 'py -3.12' works."
}

function Invoke-Checked {
  param(
    [string]$StepName,
    [scriptblock]$Command
  )

  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$StepName failed with exit code $LASTEXITCODE."
  }
}

$repoRoot = Resolve-Path "$PSScriptRoot\.."
$runtimeRoot = Join-Path $repoRoot "electron-app\build-runtime"
$pythonRuntimeDir = Join-Path $runtimeRoot "python-engine"
$bundledPythonDir = Join-Path $pythonRuntimeDir "python"
$audioRuntimeDir = Join-Path $runtimeRoot "audio-service"
$sourcePythonDir = Join-Path $repoRoot "python-engine"
$sourceAudioProj = Join-Path $repoRoot "audio-service-csharp\src\AudioCaptureService.csproj"

Write-Host "Preparing runtime directory..."
if (Test-Path $runtimeRoot) {
  Remove-Item -Recurse -Force $runtimeRoot
}
New-Item -ItemType Directory -Path $pythonRuntimeDir | Out-Null
New-Item -ItemType Directory -Path $audioRuntimeDir | Out-Null

if (-not $SkipPython) {
  $hostPythonExe = Resolve-PythonExecutable
  $hostPythonHome = (& $hostPythonExe -c "import os,sys; print(os.path.abspath(sys.base_prefix))").Trim()
  if (-not (Test-Path $hostPythonHome)) {
    throw "[Python] Failed to resolve host Python home: $hostPythonHome"
  }

  Write-Host "[Python] Copying portable Python runtime from: $hostPythonHome"
  Copy-Item -Recurse -Force $hostPythonHome $bundledPythonDir

  $bundledPythonExe = Join-Path $bundledPythonDir "python.exe"
  if (-not (Test-Path $bundledPythonExe)) {
    throw "[Python] Bundled python.exe not found at: $bundledPythonExe"
  }

  Write-Host "[Python] Copying engine source..."
  Copy-Item -Recurse -Force (Join-Path $sourcePythonDir "app") (Join-Path $pythonRuntimeDir "app")

  Write-Host "[Python] Installing engine dependencies into bundled runtime..."
  Invoke-Checked -StepName "[Python] pip upgrade" -Command { & $bundledPythonExe -m pip install --upgrade pip }
  Invoke-Checked -StepName "[Python] pip install engine" -Command { & $bundledPythonExe -m pip install $sourcePythonDir }
}

if (-not $SkipAudio) {
  Write-Host "[Audio] Publishing self-contained service..."
  Invoke-Checked -StepName "[Audio] dotnet publish" -Command {
    dotnet publish $sourceAudioProj `
      -c Release `
      -r win-x64 `
      --self-contained true `
      /p:PublishSingleFile=true `
      /p:IncludeNativeLibrariesForSelfExtract=true `
      -o $audioRuntimeDir
  }
}

Write-Host "Runtime prepared at: $runtimeRoot"
