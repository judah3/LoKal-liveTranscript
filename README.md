# LoKal liveTranscript

Desktop app for local, real-time transcription on Windows.

The app captures system output audio (WASAPI loopback), streams it to a local Python transcription engine, and renders live transcript updates in an Electron UI.

## Features

- Local-first transcription pipeline (no cloud service required for core flow)
- Real-time partial and finalized transcript updates
- Output-device audio capture on Windows
- Configurable source language, Whisper model, compute mode, and VAD
- Optional raw Whisper stream view for debugging

## Tech Stack

- Electron + React + TypeScript (`electron-app`)
- Python transcription engine with `faster-whisper` (`python-engine`)
- C# audio loopback capture service with `NAudio` (`audio-service-csharp`)
- WebSocket IPC between audio service, Python engine, and UI

## Prerequisites

- Windows 10/11
- Node.js 20+
- Python 3.10, 3.11, or 3.12
- .NET SDK 8.0+

Optional:

- NVIDIA GPU (for faster transcription when CUDA mode is selected)

## Quick Start (Development)

From repository root:

```powershell
./scripts/bootstrap.ps1
./scripts/start-dev.ps1
```

What this does:

- Installs Electron dependencies
- Creates/updates Python virtual environment in `python-engine/.venv`
- Installs Python engine dependencies
- Restores C# project dependencies
- Starts Vite + Electron dev runtime

## Manual Development Commands

If you want to run pieces manually:

```powershell
cd electron-app
npm install
npm run dev
```

Build check:

```powershell
cd electron-app
npm run build
```

## Windows Installer Build

From `electron-app`:

```powershell
npm run dist:win
```

This runs:

1. Renderer + Electron build
2. Runtime bundling via `scripts/prepare-runtime.ps1`
3. NSIS installer packaging with `electron-builder`

Expected installer output:

```text
electron-app/release/Suggest-AI-Setup-<version>.exe
```

## Runtime Architecture

1. Electron starts Python engine and C# audio service.
2. Audio service captures selected output device and sends float32 mono audio to `ws://127.0.0.1:8765/audio`.
3. Python engine performs streaming transcription and publishes events over `ws://127.0.0.1:8765/ws`.
4. Renderer consumes transcript events and updates the live panel.

## Project Structure

```text
suggest_ai/
  electron-app/
    electron/                # Main/preload process and child-process orchestration
    src/                     # React renderer
  python-engine/
    app/                     # Transcription websocket server + engine code
  audio-service-csharp/
    src/                     # WASAPI loopback capture service
  scripts/                   # Bootstrap/dev/runtime prep scripts
```

## Important Scripts

- `scripts/bootstrap.ps1`: initial dependency setup
- `scripts/start-dev.ps1`: start full local dev app
- `scripts/prepare-runtime.ps1`: prepare bundled runtime assets for packaging
- `electron-app/package.json`:
  - `npm run dev`
  - `npm run build`
  - `npm run dist:win`

## Common Troubleshooting

- Python version error:
  - Ensure `py -3.12` (or 3.11/3.10) works in terminal.
- Backend port conflicts (`5173` / `8765`):
  - `start-dev.ps1` attempts to stop conflicting processes automatically; rerun if needed.
- No audio devices listed:
  - Verify Windows output devices are enabled and active.
- Engine startup failure:
  - Re-run `./scripts/bootstrap.ps1`, then `./scripts/start-dev.ps1`.
- Slow first run:
  - Initial model download/load can take longer.

## Notes

- This project is transcription-only (translation code paths removed).
- The app currently targets Windows because audio capture relies on WASAPI loopback.
