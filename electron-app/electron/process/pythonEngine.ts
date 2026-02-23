import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import type WebSocket from "ws";

interface PythonEngineOptions {
  workingDir: string;
}

interface PythonLauncher {
  bin: string;
  args: string[];
  label: string;
}

export class PythonEngineProcess {
  private child: ChildProcessWithoutNullStreams | null = null;
  private usingExternalServer = false;
  private startPromise: Promise<void> | null = null;
  private stopPromise: Promise<void> | null = null;

  constructor(private readonly options: PythonEngineOptions) {}

  async start(): Promise<void> {
    if (this.startPromise) {
      return this.startPromise;
    }
    if (this.stopPromise) {
      await this.stopPromise;
    }
    this.startPromise = this.startInternal().finally(() => {
      this.startPromise = null;
    });
    return this.startPromise;
  }

  private async startInternal(): Promise<void> {
    if (this.child) {
      return;
    }

    if (await this.isServerReady()) {
      this.usingExternalServer = true;
      return;
    }

    const launchers = this.resolvePythonLaunchers();
    const errors: string[] = [];

    for (const launcher of launchers) {
      try {
        await this.startWithLauncher(launcher);
        return;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        errors.push(`${launcher.label}: ${message}`);
      }
    }

    throw new Error(`Unable to start Python engine. Tried: ${errors.join(" | ")}`);
  }

  async stop(): Promise<void> {
    if (this.stopPromise) {
      return this.stopPromise;
    }
    this.stopPromise = this.stopInternal().finally(() => {
      this.stopPromise = null;
    });
    return this.stopPromise;
  }

  private async stopInternal(): Promise<void> {
    if (this.startPromise) {
      try {
        await this.startPromise;
      } catch {
        // Ignore startup failures while stopping.
      }
    }
    if (this.usingExternalServer) {
      this.usingExternalServer = false;
      return;
    }
    if (!this.child) {
      return;
    }
    this.child.kill();
    this.child = null;
  }

  async sendControl(message: Record<string, unknown>): Promise<void> {
    const ws = await import("ws");
    let lastError: Error | null = null;
    for (let attempt = 0; attempt < 12; attempt++) {
      try {
        await new Promise<void>((resolve, reject) => {
          const socket = new ws.WebSocket("ws://127.0.0.1:8765/ws");
          socket.on("open", () => {
            socket.send(JSON.stringify(message));
            socket.close();
            resolve();
          });
          socket.on("error", (err: Error) => reject(err));
        });
        return;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error("Unknown websocket error");
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
    }
    throw lastError ?? new Error("Failed to send control message");
  }

  private async waitUntilReady(): Promise<void> {
    if (await this.isServerReady()) {
      return;
    }

    const ws = await import("ws");
    const start = Date.now();
    while (Date.now() - start < 30000) {
      try {
        await new Promise<void>((resolve, reject) => {
          const socket: WebSocket = new ws.WebSocket("ws://127.0.0.1:8765/ws");
          socket.on("open", () => {
            socket.close();
            resolve();
          });
          socket.on("error", reject);
        });
        return;
      } catch {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }
    throw new Error("Python engine did not become ready in time.");
  }

  private async isServerReady(): Promise<boolean> {
    const ws = await import("ws");
    try {
      await new Promise<void>((resolve, reject) => {
        const socket: WebSocket = new ws.WebSocket("ws://127.0.0.1:8765/ws");
        const timer = setTimeout(() => {
          try {
            socket.close();
          } catch {
            // best effort
          }
          reject(new Error("websocket readiness timeout"));
        }, 1200);
        socket.on("open", () => {
          clearTimeout(timer);
          socket.close();
          resolve();
        });
        socket.on("error", (err) => {
          clearTimeout(timer);
          reject(err);
        });
      });
      return true;
    } catch {
      return false;
    }
  }

  private resolvePythonLaunchers(): PythonLauncher[] {
    const venvPython = path.join(this.options.workingDir, ".venv", "Scripts", "python.exe");
    const bundledPython = path.join(this.options.workingDir, "python", "python.exe");
    const launchers: PythonLauncher[] = [];

    if (existsSync(bundledPython)) {
      launchers.push({
        bin: bundledPython,
        args: ["-m", "app.main"],
        label: "bundled python"
      });
    }

    if (existsSync(venvPython)) {
      launchers.push({
        bin: venvPython,
        args: ["-m", "app.main"],
        label: "venv python"
      });
    }

    if (process.platform === "win32") {
      launchers.push({
        bin: "py",
        args: ["-3", "-m", "app.main"],
        label: "py launcher"
      });
    }

    launchers.push({
      bin: "python",
      args: ["-m", "app.main"],
      label: "python in PATH"
    });

    return launchers;
  }

  private async startWithLauncher(launcher: PythonLauncher): Promise<void> {
    const child = spawn(launcher.bin, launcher.args, {
      cwd: this.options.workingDir,
      env: process.env,
      stdio: "pipe"
    });

    this.child = child;

    child.stderr.on("data", (chunk) => {
      const text = String(chunk);
      if (text.trim()) {
        process.stderr.write(`[python] ${text}`);
      }
    });

    child.stdout.on("data", (chunk) => {
      const text = String(chunk);
      if (text.trim()) {
        process.stdout.write(`[python] ${text}`);
      }
    });

    const started = await new Promise<boolean>((resolve, reject) => {
      let settled = false;

      const onError = (err: Error) => {
        if (settled) {
          return;
        }
        settled = true;
        this.child = null;
        reject(err);
      };

      const onExit = (code: number | null) => {
        if (settled) {
          return;
        }
        settled = true;
        this.child = null;
        reject(new Error(`Python process exited early (code=${code ?? "unknown"})`));
      };

      child.once("error", onError);
      child.once("exit", onExit);

      this.waitUntilReady()
        .then(() => {
          if (settled) {
            return;
          }
          settled = true;
          child.off("error", onError);
          child.off("exit", onExit);
          child.on("exit", () => {
            this.child = null;
          });
          resolve(true);
        })
        .catch((error) => {
          if (settled) {
            return;
          }
          settled = true;
          child.off("error", onError);
          child.off("exit", onExit);
          try {
            child.kill();
          } catch {
            // best effort
          }
          this.child = null;
          reject(error);
        });
    });

    if (!started) {
      throw new Error("Python did not start");
    }
  }
}
