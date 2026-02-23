import React from "react";
import type { AppSettings, ComputeMode, TranscriptionPreset } from "../types/contracts";

interface SettingsModalProps {
  open: boolean;
  settings: AppSettings;
  transcriptionModelStatus?: "not_downloaded" | "downloading" | "ready";
  onChange: (next: AppSettings) => void;
  onClose: () => void;
}

const whisperModels = ["tiny", "base", "small", "medium", "large-v3"];

const presetToModel: Record<Exclude<TranscriptionPreset, "custom">, string> = {
  ultra_fast: "tiny",
  balanced: "base",
  high_accuracy: "small"
};

export function SettingsModal({
  open,
  settings,
  transcriptionModelStatus,
  onChange,
  onClose
}: SettingsModalProps) {
  if (!open) {
    return null;
  }

  const set = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    onChange({ ...settings, [key]: value });
  };

  const setPreset = (preset: TranscriptionPreset) => {
    if (preset === "custom") {
      onChange({ ...settings, transcriptionPreset: "custom" });
      return;
    }
    onChange({
      ...settings,
      transcriptionPreset: preset,
      whisperModel: presetToModel[preset]
    });
  };

  const setWhisperModel = (whisperModel: string) => {
    onChange({ ...settings, whisperModel, transcriptionPreset: "custom" });
  };

  const modelStatusLabel = (status?: "not_downloaded" | "downloading" | "ready"): string => {
    if (status === "downloading") {
      return "Downloading...";
    }
    if (status === "ready") {
      return "Ready (cached)";
    }
    return "Not downloaded";
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Engine Settings</h2>
        <label>
          Transcription Preset
          <select
            value={settings.transcriptionPreset}
            onChange={(e) => setPreset(e.target.value as TranscriptionPreset)}
          >
            <option value="ultra_fast">Ultra Fast</option>
            <option value="balanced">Balanced</option>
            <option value="high_accuracy">High Accuracy</option>
            <option value="custom">Custom</option>
          </select>
        </label>
        <div className="model-status">Transcription model: {modelStatusLabel(transcriptionModelStatus)}</div>
        <label className="toggle-row">
          <span>Enable NLP Processing</span>
          <input
            type="checkbox"
            checked={settings.enableNlp}
            onChange={(e) => set("enableNlp", e.target.checked)}
          />
        </label>
        <div className="model-status">NLP is currently {settings.enableNlp ? "On" : "Off"}</div>
        <label className="toggle-row">
          <span>Show Raw Whisper Stream</span>
          <input
            type="checkbox"
            checked={settings.enableRawWhisper}
            onChange={(e) => set("enableRawWhisper", e.target.checked)}
          />
        </label>
        <div className="model-status">
          Raw stream is {settings.enableRawWhisper ? "On (debug stream, less stable)" : "Off (sentence-aligned view)"}
        </div>
        <label className="toggle-row">
          <span>Enable VAD</span>
          <input
            type="checkbox"
            checked={settings.enableVad}
            onChange={(e) => set("enableVad", e.target.checked)}
          />
        </label>
        <div className="model-status">
          VAD is {settings.enableVad ? "On (better silence trimming)" : "Off (full audio decoding)"}
        </div>
        <details className="advanced">
          <summary>Advanced</summary>
          <label>
            Compute Mode
            <select value={settings.computeMode} onChange={(e) => set("computeMode", e.target.value as ComputeMode)}>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA GPU</option>
            </select>
          </label>
          <label>
            Whisper Model
            <select value={settings.whisperModel} onChange={(e) => setWhisperModel(e.target.value)}>
              {whisperModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
        </details>
        <button className="btn close" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
