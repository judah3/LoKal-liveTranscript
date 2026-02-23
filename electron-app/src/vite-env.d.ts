import type { AudioDevice, AppSettings } from "./types/contracts";

export interface ElectronBridge {
  listAudioDevices: () => Promise<AudioDevice[]>;
  startCapture: (input: { deviceId: string }) => Promise<void>;
  stopCapture: () => Promise<void>;
  setAlwaysOnTop: (enabled: boolean) => Promise<void>;
  startBackends: () => Promise<void>;
  stopBackends: () => Promise<void>;
  updateEngineSettings: (settings: AppSettings) => Promise<void>;
  onAudioStatus: (listener: (event: { listening: boolean; error?: string }) => void) => () => void;
}

declare global {
  interface Window {
    suggestAi: ElectronBridge;
  }
}

export {};
