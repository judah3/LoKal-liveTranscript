import { useEffect, useRef, useState } from "react";
import { Toolbar } from "./components/Toolbar";
import { LivePanel } from "./components/LivePanel";
import { SettingsModal } from "./components/SettingsModal";
import { useEngineSocket } from "./hooks/useEngineSocket";
import type { AppSettings, AudioDevice } from "./types/contracts";

const SETTINGS_STORAGE_KEY = "suggest_ai.transcribe.settings.v1";

const defaultSettings: AppSettings = {
  sourceLanguage: "en",
  whisperModel: "medium",
  computeMode: "cuda",
  transcriptionPreset: "custom",
  enableNlp: true,
  enableRawWhisper: false,
  enableVad: true
};

export function App() {
  const [settings, setSettings] = useState<AppSettings>(() => loadPersistedSettings());
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [audioListening, setAudioListening] = useState(false);
  const [audioError, setAudioError] = useState<string | undefined>();
  const [isBooting, setIsBooting] = useState(true);
  const [loadingInfo, setLoadingInfo] = useState("Preparing runtime...");
  const [isTogglingCapture, setIsTogglingCapture] = useState(false);
  const [uiActiveAudio, setUiActiveAudio] = useState(false);
  const [liveCaptionMode, setLiveCaptionMode] = useState(false);
  const idleHoldTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { state, clear } = useEngineSocket(settings, { liveCaptionMode });

  useEffect(() => {
    persistSettings(settings);
  }, [settings]);

  useEffect(() => {
    if (!window.suggestAi) {
      setAudioError("Electron bridge unavailable. Restart the app.");
      setLoadingInfo("Startup failed.");
      setIsBooting(false);
      return;
    }
    let cancelled = false;
    let startedBackends = false;
    const startBackendsPromise = (async () => {
      await window.suggestAi.startBackends();
      startedBackends = true;
    })();

    const run = async () => {
      try {
        setIsBooting(true);
        setLoadingInfo("Starting backend services...");
        await startBackendsPromise;
        if (cancelled) {
          return;
        }
        setLoadingInfo("Loading audio devices...");
        const available = await window.suggestAi.listAudioDevices();
        if (cancelled) {
          return;
        }
        setDevices(available);
        setAudioError(undefined);
        if (available.length > 0) {
          setDeviceId(available[0].id);
          setLoadingInfo(`Ready. Found ${available.length} audio device${available.length === 1 ? "" : "s"}.`);
        } else {
          setLoadingInfo("Ready. No output audio devices detected.");
        }
      } catch (error) {
        setAudioError(toDisplayErrorMessage(error));
        setLoadingInfo("Startup failed.");
      } finally {
        setIsBooting(false);
      }
    };

    void run();

    const off = window.suggestAi.onAudioStatus((event) => {
      setAudioListening(event.listening);
      setAudioError(event.error);
    });

    return () => {
      cancelled = true;
      off();
      void (async () => {
        try {
          await startBackendsPromise;
        } catch {
          // Startup failed; no running backends to stop.
        }
        if (startedBackends) {
          await window.suggestAi.stopBackends();
        }
      })();
    };
  }, []);

  const listening = audioListening || state.listening;
  const activeAudio = Boolean(state.activeAudio);
  useEffect(() => {
    if (activeAudio) {
      if (idleHoldTimerRef.current) {
        clearTimeout(idleHoldTimerRef.current);
        idleHoldTimerRef.current = null;
      }
      setUiActiveAudio(true);
      return;
    }
    if (!listening) {
      setUiActiveAudio(false);
      return;
    }
    idleHoldTimerRef.current = setTimeout(() => {
      setUiActiveAudio(false);
      idleHoldTimerRef.current = null;
    }, 2200);
    return () => {
      if (idleHoldTimerRef.current) {
        clearTimeout(idleHoldTimerRef.current);
        idleHoldTimerRef.current = null;
      }
    };
  }, [activeAudio, listening]);

  useEffect(() => {
    if (!window.suggestAi?.setAlwaysOnTop) {
      return;
    }
    void window.suggestAi.setAlwaysOnTop(listening);
    return () => {
      void window.suggestAi.setAlwaysOnTop(false);
    };
  }, [listening]);

  const controlsBusy = isBooting || isTogglingCapture;
  const busyLabel = isBooting
    ? "Starting services..."
    : isTogglingCapture
      ? listening
        ? "Stopping capture..."
        : "Starting capture..."
      : undefined;
  const transcriptLines = settings.enableRawWhisper ? state.rawTranscriptLines : state.transcriptLines;
  const transcriptionPartialLabel = formatTranscriptionPartialLabel(state.partialPhase, state.partialConfidence);
  const currentTranscriptFinal = String(transcriptLines[transcriptLines.length - 1] ?? "").trim();
  const retainedTranscriptFinal = useRetainedCaptionLine(currentTranscriptFinal, liveCaptionMode);
  const statusDetail =
    audioError ||
    state.lastError ||
    (isBooting
      ? `Initializing services... ${loadingInfo}`
      : isTogglingCapture
        ? (listening ? "Stopping capture..." : "Starting capture...")
        : "") ||
    `${state.runtimeDevice ? `Mode: ${state.runtimeDevice}/${state.runtimeComputeType || "auto"} | ` : ""}Latency: ${state.latencyMs}ms | RMS: ${(state.rms ?? 0).toFixed(4)}`;
  const hasError = Boolean(audioError || state.lastError);

  const handleToggleListening = async () => {
    if (controlsBusy) {
      return;
    }
    setAudioError(undefined);
    setIsTogglingCapture(true);
    try {
      if (listening) {
        await window.suggestAi.stopCapture();
      } else {
        await window.suggestAi.startCapture({ deviceId });
      }
    } catch (error) {
      setAudioError(error instanceof Error ? error.message : "Failed to toggle capture");
    } finally {
      setIsTogglingCapture(false);
    }
  };

  return (
    <main className="app-shell">
      <Toolbar
        devices={devices}
        selectedDevice={deviceId}
        sourceLanguage={settings.sourceLanguage}
        listening={listening}
        busy={controlsBusy}
        busyLabel={busyLabel}
        liveCaptionMode={liveCaptionMode}
        onDeviceChange={setDeviceId}
        onSourceChange={(sourceLanguage) => setSettings((prev) => ({ ...prev, sourceLanguage }))}
        onToggleLiveCaptionMode={() => setLiveCaptionMode((prev) => !prev)}
        onToggleListening={() => void handleToggleListening()}
        onOpenSettings={() => setSettingsOpen(true)}
        onClear={clear}
      />
      <div className="grid compact-grid">
        <LivePanel
          title="Live Transcription"
          lines={transcriptLines}
          partial={state.partialText}
          partialLoadingLabel={transcriptionPartialLabel}
          captionMode={liveCaptionMode}
          speechActive={activeAudio}
          currentFinalText={currentTranscriptFinal}
          retainedFinalText={retainedTranscriptFinal.text}
          retainedFinalFading={retainedTranscriptFinal.fading}
        />
      </div>

      {isBooting ? (
        <section className="startup-info">
          <strong>Loading...</strong>
          <span>{loadingInfo}</span>
        </section>
      ) : null}

      <footer className="status-bar">
        <span className={listening ? "status listening" : "status"}>
          {listening ? (uiActiveAudio ? "Listening..." : "Idle (no audio)") : "Idle"}
        </span>
        <span className={hasError ? "status-detail error" : "status-detail"}>{statusDetail}</span>
      </footer>

      <SettingsModal
        open={settingsOpen}
        settings={settings}
        transcriptionModelStatus={state.transcriptionModelStatus}
        onChange={setSettings}
        onClose={() => setSettingsOpen(false)}
      />
    </main>
  );
}

function useRetainedCaptionLine(
  currentFinal: string,
  enabled: boolean,
  holdMs = 3000,
  fadeMs = 700
): { text: string; fading: boolean } {
  const [retained, setRetained] = useState<{ text: string; fading: boolean }>({ text: "", fading: false });
  const lastSeenFinalRef = useRef("");
  const holdUntilRef = useRef(0);
  const fadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const clearTimers = () => {
      if (fadeTimerRef.current) {
        clearTimeout(fadeTimerRef.current);
        fadeTimerRef.current = null;
      }
      if (clearTimerRef.current) {
        clearTimeout(clearTimerRef.current);
        clearTimerRef.current = null;
      }
    };

    if (!enabled) {
      clearTimers();
      lastSeenFinalRef.current = "";
      holdUntilRef.current = 0;
      setRetained({ text: "", fading: false });
      return;
    }

    const next = currentFinal.trim();
    if (!next || next === lastSeenFinalRef.current) {
      return;
    }
    const previous = lastSeenFinalRef.current.trim();
    lastSeenFinalRef.current = next;
    if (!previous || previous === next) {
      return;
    }

    const now = Date.now();
    if (now < holdUntilRef.current) {
      return;
    }

    clearTimers();
    holdUntilRef.current = now + holdMs + fadeMs;
    setRetained({ text: previous, fading: false });

    fadeTimerRef.current = setTimeout(() => {
      setRetained((prev) => (prev.text === previous ? { ...prev, fading: true } : prev));
    }, holdMs);
    clearTimerRef.current = setTimeout(() => {
      setRetained((prev) => (prev.text === previous ? { text: "", fading: false } : prev));
      holdUntilRef.current = 0;
    }, holdMs + fadeMs);

    return () => {
      clearTimers();
    };
  }, [currentFinal, enabled, holdMs, fadeMs]);

  return retained;
}

function loadPersistedSettings(): AppSettings {
  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) {
      return defaultSettings;
    }
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    const computeMode = parsed.computeMode === "cpu" || parsed.computeMode === "cuda" ? parsed.computeMode : defaultSettings.computeMode;
    const transcriptionPreset =
      parsed.transcriptionPreset === "ultra_fast" ||
      parsed.transcriptionPreset === "balanced" ||
      parsed.transcriptionPreset === "high_accuracy" ||
      parsed.transcriptionPreset === "custom"
        ? parsed.transcriptionPreset
        : defaultSettings.transcriptionPreset;
    return {
      sourceLanguage: typeof parsed.sourceLanguage === "string" && parsed.sourceLanguage ? parsed.sourceLanguage : defaultSettings.sourceLanguage,
      whisperModel: typeof parsed.whisperModel === "string" && parsed.whisperModel ? parsed.whisperModel : defaultSettings.whisperModel,
      computeMode,
      transcriptionPreset,
      enableNlp: typeof parsed.enableNlp === "boolean" ? parsed.enableNlp : defaultSettings.enableNlp,
      enableRawWhisper:
        typeof parsed.enableRawWhisper === "boolean" ? parsed.enableRawWhisper : defaultSettings.enableRawWhisper,
      enableVad: typeof parsed.enableVad === "boolean" ? parsed.enableVad : defaultSettings.enableVad
    };
  } catch {
    return defaultSettings;
  }
}

function persistSettings(settings: AppSettings): void {
  try {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Best-effort only.
  }
}

function toDisplayErrorMessage(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error ?? "Failed to start services");
  let cleaned = raw.replace(/^Error invoking remote method '[^']+':\s*/, "");
  cleaned = cleaned.replace(/^Error:\s*/, "");

  if (!cleaned.startsWith("Unable to start Python engine. Tried:")) {
    return cleaned;
  }

  const attempts = cleaned
    .replace("Unable to start Python engine. Tried:", "")
    .split("|")
    .map((part) => part.trim())
    .filter(Boolean);

  const formattedAttempts = attempts.length > 0 ? attempts.map((part) => `- ${part}`).join("\n") : "- No launch attempts reported.";
  const hasBundledAttempt = attempts.some((part) => part.toLowerCase().startsWith("bundled python:"));
  const recoveryHint = hasBundledAttempt
    ? "Repair or reinstall the app using the installer to restore bundled runtime files."
    : "Run ./scripts/bootstrap.ps1 to install Python dependencies.";
  return [
    "Unable to start Python engine.",
    "Attempts:",
    formattedAttempts,
    recoveryHint
  ].join("\n");
}

function formatTranscriptionPartialLabel(
  phase?: "interim" | "refined",
  confidence?: number
): string {
  const score = typeof confidence === "number" && Number.isFinite(confidence)
    ? `${Math.round(Math.max(0, Math.min(1, confidence)) * 100)}%`
    : "";
  if (phase === "interim") {
    return score ? `Interim ${score}` : "Interim";
  }
  if (phase === "refined") {
    return score ? `Refined ${score}` : "Refined";
  }
  return "Transcribing...";
}
