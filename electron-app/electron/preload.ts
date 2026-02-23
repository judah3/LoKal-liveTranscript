import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("suggestAi", {
  startBackends: () => ipcRenderer.invoke("backend:start"),
  stopBackends: () => ipcRenderer.invoke("backend:stop"),
  listAudioDevices: () => ipcRenderer.invoke("audio:listDevices"),
  startCapture: (input: { deviceId: string }) => ipcRenderer.invoke("audio:startCapture", input),
  stopCapture: () => ipcRenderer.invoke("audio:stopCapture"),
  setAlwaysOnTop: (enabled: boolean) => ipcRenderer.invoke("window:setAlwaysOnTop", { enabled }),
  updateEngineSettings: (settings: unknown) => ipcRenderer.invoke("engine:updateSettings", settings),
  onAudioStatus: (listener: (event: { listening: boolean; error?: string }) => void) => {
    const wrapped = (_event: unknown, payload: { listening: boolean; error?: string }) => listener(payload);
    ipcRenderer.on("audio:status", wrapped);
    return () => ipcRenderer.removeListener("audio:status", wrapped);
  }
});
