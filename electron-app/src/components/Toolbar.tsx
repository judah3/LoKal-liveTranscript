import React from "react";
import type { AudioDevice } from "../types/contracts";

interface ToolbarProps {
  devices: AudioDevice[];
  selectedDevice: string;
  sourceLanguage: string;
  listening: boolean;
  busy?: boolean;
  busyLabel?: string;
  liveCaptionMode: boolean;
  onDeviceChange: (deviceId: string) => void;
  onSourceChange: (lang: string) => void;
  onToggleLiveCaptionMode: () => void;
  onToggleListening: () => void;
  onOpenSettings: () => void;
  onClear: () => void;
}

const languages = [
  { value: "en", label: "en" },
  { value: "es", label: "es" },
  { value: "fr", label: "fr" },
  { value: "de", label: "de" },
  { value: "it", label: "it" },
  { value: "pt", label: "pt" },
  { value: "ja", label: "ja" },
  { value: "ko", label: "ko" },
  { value: "zh", label: "zh" },
  { value: "tl", label: "tl (Tagalog/Filipino)" }
];

export function Toolbar(props: ToolbarProps) {
  const busy = Boolean(props.busy);

  return (
    <div className="toolbar">
      {props.busyLabel ? <div className="toolbar-status">{props.busyLabel}</div> : null}
      <label>
        Output Device
        <select
          value={props.selectedDevice}
          onChange={(e) => props.onDeviceChange(e.target.value)}
          disabled={busy}
        >
          {props.devices.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Source Language
        <select
          value={props.sourceLanguage}
          onChange={(e) => props.onSourceChange(e.target.value)}
          disabled={busy}
        >
          {languages.map((lang) => (
            <option key={lang.value} value={lang.value}>
              {lang.label}
            </option>
          ))}
        </select>
      </label>
      <button
        className={`btn ${props.listening ? "stop" : "start"}`}
        onClick={props.onToggleListening}
        disabled={busy}
      >
        {busy ? (props.listening ? "Stopping..." : "Starting...") : props.listening ? "Stop Listening" : "Start Listening"}
      </button>
      <button className="btn settings" onClick={props.onOpenSettings} disabled={busy}>
        Settings
      </button>
      <button className="btn clear" onClick={props.onClear} disabled={busy}>
        Clear
      </button>
      <button className={`btn ${props.liveCaptionMode ? "caption-on" : "caption-off"}`} onClick={props.onToggleLiveCaptionMode} disabled={busy}>
        {props.liveCaptionMode ? "Live Caption: ON" : "Live Caption: OFF"}
      </button>
      {props.liveCaptionMode ? (
        <div className="caption-flags">
          <span className="caption-flag">Word-by-word</span>
          <span className="caption-flag">No history</span>
        </div>
      ) : null}
    </div>
  );
}
