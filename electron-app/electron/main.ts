import { app, BrowserWindow, ipcMain, Menu } from "electron";
import path from "node:path";
import { AudioServiceProcess } from "./process/audioService";
import { PythonEngineProcess } from "./process/pythonEngine";

let mainWindow: BrowserWindow | null = null;
let python: PythonEngineProcess | null = null;
let audio: AudioServiceProcess | null = null;
let backendsRunning = false;
let backendStartPromise: Promise<void> | null = null;
let backendStopPromise: Promise<void> | null = null;
let backendLeaseCount = 0;

function resolveRepoRoot(): string {
  // Dev main process runs from electron-app/dist-electron.
  return path.resolve(__dirname, "../..");
}

function resolveRuntimeServices(): void {
  const runtimeRoot = path.join(process.resourcesPath, "runtime");
  const repoRoot = resolveRepoRoot();

  python = new PythonEngineProcess({
    workingDir: app.isPackaged ? path.join(runtimeRoot, "python-engine") : path.join(repoRoot, "python-engine")
  });

  audio = new AudioServiceProcess(
    app.isPackaged
      ? {
          executablePath: path.join(runtimeRoot, "audio-service", "AudioCaptureService.exe")
        }
      : {
          projectPath: path.join(repoRoot, "audio-service-csharp", "src", "AudioCaptureService.csproj")
        }
  );
}

function resolveAppIconPath(): string {
  const repoRoot = resolveRepoRoot();
  return app.isPackaged
    ? path.join(process.resourcesPath, "runtime", "juda_logo.png")
    : path.join(repoRoot, "electron-app", "src", "images", "juda_logo.png");
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 980,
    icon: resolveAppIconPath(),
    autoHideMenuBar: true,
    backgroundColor: "#e2e1eb",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  mainWindow.setMenuBarVisibility(false);

  if (process.env.NODE_ENV === "development") {
    if (process.env.DEBUG_ELECTRON === "1") {
      mainWindow.webContents.on("console-message", (_event, level, message, line, sourceId) => {
        // eslint-disable-next-line no-console
        console.log(`[renderer:${level}] ${sourceId}:${line} ${message}`);
      });
      mainWindow.webContents.on("did-fail-load", (_event, errorCode, errorDescription, validatedURL) => {
        // eslint-disable-next-line no-console
        console.error(`[renderer-load-fail] code=${errorCode} url=${validatedURL} error=${errorDescription}`);
      });
      mainWindow.webContents.on("render-process-gone", (_event, details) => {
        // eslint-disable-next-line no-console
        console.error(`[renderer-gone] reason=${details.reason} exitCode=${details.exitCode}`);
      });
    }
    void mainWindow.loadURL("http://localhost:5173");
    if (process.env.OPEN_DEVTOOLS === "1") {
      mainWindow.webContents.openDevTools({ mode: "detach" });
    }
  } else {
    void mainWindow.loadFile(path.resolve(__dirname, "../dist/index.html"));
  }

  audio?.onStatus((status) => {
    mainWindow?.webContents.send("audio:status", status);
  });
}

app.whenReady().then(() => {
  resolveRuntimeServices();
  Menu.setApplicationMenu(null);
  createWindow();

  ipcMain.handle("backend:start", async () => {
    if (!python || !audio) {
      throw new Error("Runtime services not initialized");
    }
    backendLeaseCount += 1;
    if (backendsRunning) {
      return;
    }
    if (backendStartPromise) {
      await backendStartPromise;
      return;
    }
    if (backendStopPromise) {
      await backendStopPromise;
    }

    backendStartPromise = (async () => {
      await python.start();
      await audio.start();
      backendsRunning = true;
    })();

    try {
      await backendStartPromise;
    } catch (error) {
      backendsRunning = false;
      backendLeaseCount = Math.max(0, backendLeaseCount - 1);
      try {
        await audio.stop();
      } catch {
        // Ignore cleanup failures after startup failure.
      }
      try {
        await python.stop();
      } catch {
        // Ignore cleanup failures after startup failure.
      }
      throw error;
    } finally {
      backendStartPromise = null;
    }
  });

  ipcMain.handle("backend:stop", async () => {
    if (!python || !audio) {
      return;
    }
    if (backendLeaseCount > 0) {
      backendLeaseCount -= 1;
    }
    if (backendLeaseCount > 0) {
      return;
    }
    if (!backendsRunning && !backendStartPromise) {
      return;
    }
    if (backendStopPromise) {
      await backendStopPromise;
      return;
    }

    backendStopPromise = (async () => {
      if (backendStartPromise) {
        try {
          await backendStartPromise;
        } catch {
          // Continue shutdown after a failed startup attempt.
        }
      }
      await audio.stop();
      await python.stop();
      backendsRunning = false;
      backendLeaseCount = 0;
    })();

    try {
      await backendStopPromise;
    } finally {
      backendStopPromise = null;
    }
  });

  ipcMain.handle("audio:listDevices", async () => {
    if (!audio) {
      return [];
    }
    return audio.listDevices();
  });

  ipcMain.handle("audio:startCapture", async (_event, args: { deviceId: string }) => {
    if (!audio) {
      throw new Error("Audio service not initialized");
    }
    await audio.startCapture(args.deviceId);
  });

  ipcMain.handle("audio:stopCapture", async () => {
    if (!audio) {
      return;
    }
    await audio.stopCapture();
  });

  ipcMain.handle("engine:updateSettings", async (_event, settings) => {
    if (!python) {
      throw new Error("Python engine not initialized");
    }
    await python.sendControl({ type: "update_config", payload: settings });
  });

  ipcMain.handle("window:setAlwaysOnTop", async (_event, args: { enabled: boolean }) => {
    if (!mainWindow) {
      return;
    }
    const enabled = Boolean(args?.enabled);
    mainWindow.setAlwaysOnTop(enabled, "screen-saver");
    if (enabled) {
      mainWindow.show();
      mainWindow.focus();
    }
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", async () => {
  backendLeaseCount = 0;
  if (backendStartPromise) {
    try {
      await backendStartPromise;
    } catch {
      // Continue shutdown path.
    }
  }
  if (audio) {
    await audio.stop();
  }
  if (python) {
    await python.stop();
  }
  backendsRunning = false;
  if (process.platform !== "darwin") {
    app.quit();
  }
});
