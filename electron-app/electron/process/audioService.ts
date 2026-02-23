import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface } from "node:readline";

interface AudioDevice {
  id: string;
  name: string;
}

interface AudioServiceOptions {
  projectPath?: string;
  executablePath?: string;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
  type: string;
}

export class AudioServiceProcess {
  private child: ChildProcessWithoutNullStreams | null = null;
  private pending = new Map<string, PendingRequest>();
  private statusListeners: Array<(status: { listening: boolean; error?: string }) => void> = [];
  private startPromise: Promise<void> | null = null;
  private stopPromise: Promise<void> | null = null;

  constructor(private readonly options: AudioServiceOptions) {}

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

    if (this.options.executablePath) {
      this.child = spawn(this.options.executablePath, [], {
        env: process.env,
        stdio: "pipe"
      });
    } else if (this.options.projectPath) {
      this.child = spawn("dotnet", ["run", "--project", this.options.projectPath], {
        env: process.env,
        stdio: "pipe"
      });
    } else {
      throw new Error("Audio service path is not configured");
    }

    const rl = createInterface({ input: this.child.stdout });
    rl.on("line", (line) => this.handleLine(line));

    this.child.stderr.on("data", (chunk) => {
      this.emitStatus({ listening: false, error: String(chunk).trim() });
    });

    this.child.on("exit", () => {
      this.child = null;
      this.pending.clear();
      this.emitStatus({ listening: false, error: "Audio service stopped" });
    });

    try {
      await this.send("ping", {});
    } catch (error) {
      try {
        this.child.kill();
      } catch {
        // Best effort cleanup.
      }
      this.child = null;
      throw error;
    }
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
        // Continue shutdown even if startup failed.
      }
    }
    if (!this.child) {
      return;
    }

    try {
      await this.send("shutdown", {});
    } catch {
      // If service isn't responsive, force kill below.
    }
    try {
      this.child.kill();
    } catch {
      // Best effort cleanup.
    }
    this.child = null;
  }

  onStatus(listener: (status: { listening: boolean; error?: string }) => void): void {
    this.statusListeners.push(listener);
  }

  async listDevices(): Promise<AudioDevice[]> {
    const devices = await this.send("list_devices", {});
    return devices as AudioDevice[];
  }

  async startCapture(deviceId: string): Promise<void> {
    await this.send("start", { deviceId, backendUrl: "ws://127.0.0.1:8765/audio" });
    this.emitStatus({ listening: true });
  }

  async stopCapture(): Promise<void> {
    await this.send("stop", {});
    this.emitStatus({ listening: false });
  }

  private async send(type: string, payload: Record<string, unknown>): Promise<unknown> {
    if (!this.child) {
      throw new Error("Audio service not running");
    }

    const requestId = `${type}-${Date.now()}-${Math.random()}`;
    const message = JSON.stringify({ requestId, type, payload });

    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject, type });
      this.child?.stdin.write(`${message}\n`);

      setTimeout(() => {
        if (this.pending.has(requestId)) {
          this.pending.delete(requestId);
          reject(new Error(`Audio request timeout: ${type}`));
        }
      }, 60000);
    });
  }

  private handleLine(line: string): void {
    let msg: any;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }

    if (msg.type === "status") {
      this.emitStatus({ listening: Boolean(msg.payload?.listening), error: msg.payload?.error });
    }

    const pending = this.pending.get(msg.requestId);
    if (pending) {
      this.pending.delete(msg.requestId);
      if (msg.error) {
        pending.reject(new Error(msg.error));
      } else {
        pending.resolve(msg.payload);
      }
    }
  }

  private emitStatus(status: { listening: boolean; error?: string }): void {
    for (const listener of this.statusListeners) {
      listener(status);
    }
  }
}
