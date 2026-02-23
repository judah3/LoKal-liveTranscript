import { useEffect, useMemo, useRef, useState } from "react";
import type { AppSettings, CaptureState, EngineEvent } from "../types/contracts";

const STREAM_FLUSH_MS = 50;
const RECONNECT_BASE_MS = 900;
const RECONNECT_MAX_MS = 8000;
const MAX_CONVERSATION_LINES = 14;
const CAPTION_MODE_MAX_LINES = 1;
const URGENT_EVENTS = new Set<EngineEvent["type"]>([
  "status",
  "model_status",
  "transcription_partial",
  "transcription_final",
  "error"
]);

const initialState: CaptureState = {
  listening: false,
  activeAudio: false,
  rms: 0,
  latencyMs: 0,
  partialText: "",
  partialPhase: undefined,
  partialConfidence: undefined,
  rawTranscriptLines: [],
  transcriptLines: []
};

export function useEngineSocket(settings: AppSettings, options?: { liveCaptionMode?: boolean }) {
  const [state, setState] = useState<CaptureState>(initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptRef = useRef(0);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flushDueAtRef = useRef<number>(0);
  const pendingEventsRef = useRef<EngineEvent[]>([]);
  const closedRef = useRef(false);
  const liveCaptionMode = Boolean(options?.liveCaptionMode);
  const liveCaptionModeRef = useRef(liveCaptionMode);

  useEffect(() => {
    liveCaptionModeRef.current = liveCaptionMode;
    setState((prev) => trimConversationLines(prev, liveCaptionMode ? CAPTION_MODE_MAX_LINES : MAX_CONVERSATION_LINES));
  }, [liveCaptionMode]);

  useEffect(() => {
    closedRef.current = false;

    const flushBufferedEvents = () => {
      if (pendingEventsRef.current.length === 0) {
        return;
      }
      const events = pendingEventsRef.current;
      pendingEventsRef.current = [];
      setState((prev) => events.reduce((acc, msg) => reduceState(acc, msg), prev));
    };

    const scheduleFlush = (delayMs: number) => {
      if (closedRef.current) {
        return;
      }
      const dueAt = Date.now() + delayMs;
      if (flushTimerRef.current && flushDueAtRef.current <= dueAt) {
        return;
      }
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
      }
      flushDueAtRef.current = dueAt;
      flushTimerRef.current = setTimeout(() => {
        flushTimerRef.current = null;
        flushDueAtRef.current = 0;
        flushBufferedEvents();
      }, delayMs);
    };

    const enqueueEvent = (msg: EngineEvent) => {
      pendingEventsRef.current.push(msg);
      scheduleFlush(URGENT_EVENTS.has(msg.type) ? 0 : STREAM_FLUSH_MS);
    };

    const connect = () => {
      if (closedRef.current) {
        return;
      }
      const ws = new WebSocket("ws://127.0.0.1:8765/ws");
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectAttemptRef.current = 0;
        setState((prev) => ({ ...prev, lastError: undefined }));
        ws.send(JSON.stringify({ type: "update_config", payload: settings }));
      };

      ws.onmessage = (event) => {
        try {
          const msg: EngineEvent = JSON.parse(String(event.data));
          enqueueEvent(msg);
        } catch {
          setState((prev) => ({ ...prev, lastError: "Invalid engine payload" }));
        }
      };

      ws.onerror = () => {
        setState((prev) => ({ ...prev, lastError: "Engine websocket error" }));
      };

      ws.onclose = () => {
        if (closedRef.current) {
          return;
        }
        wsRef.current = null;
        reconnectAttemptRef.current += 1;
        const delayMs = Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * (2 ** (reconnectAttemptRef.current - 1)));
        setState((prev) => ({ ...prev, lastError: "Engine disconnected. Reconnecting..." }));
        reconnectTimer.current = setTimeout(connect, delayMs);
      };
    };

    connect();
    return () => {
      closedRef.current = true;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
      }
      pendingEventsRef.current = [];
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "update_config", payload: settings }));
    }
  }, [settings]);

  const api = useMemo(
    () => ({
      clear: () => {
        if (flushTimerRef.current) {
          clearTimeout(flushTimerRef.current);
          flushTimerRef.current = null;
          flushDueAtRef.current = 0;
        }
        pendingEventsRef.current = [];
        setState(initialState);
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "clear" }));
        }
      }
    }),
    []
  );

  return { state: trimConversationLines(state, liveCaptionMode ? CAPTION_MODE_MAX_LINES : MAX_CONVERSATION_LINES), ...api };
}

function reduceState(prev: CaptureState, msg: EngineEvent): CaptureState {
  switch (msg.type) {
    case "status":
      return {
        ...prev,
        listening: Boolean(msg.payload.listening),
        activeAudio: Boolean(msg.payload.activeAudio ?? prev.activeAudio),
        rms: Number(msg.payload.rms ?? prev.rms ?? 0),
        latencyMs: Number(msg.payload.latencyMs ?? prev.latencyMs),
        runtimeDevice: String(msg.payload.device ?? prev.runtimeDevice ?? ""),
        runtimeComputeType: String(msg.payload.computeType ?? prev.runtimeComputeType ?? "")
      };
    case "transcription_partial":
      return {
        ...prev,
        partialText: String(msg.payload.text ?? ""),
        partialPhase: toPartialPhase(msg.payload.phase, prev.partialPhase),
        partialConfidence: toFiniteNumber(msg.payload.confidence, prev.partialConfidence),
        latencyMs: Number(msg.payload.latencyMs ?? prev.latencyMs)
      };
    case "transcription_raw": {
      const text = String(msg.payload.text ?? "").trim();
      if (!text) {
        return prev;
      }
      const nextLines = mergeRawLines(prev.rawTranscriptLines, text);
      const clipped = nextLines.length > 500 ? nextLines.slice(nextLines.length - 500) : nextLines;
      return {
        ...prev,
        rawTranscriptLines: clipped,
        latencyMs: Number(msg.payload.latencyMs ?? prev.latencyMs)
      };
    }
    case "model_status":
      if (String(msg.payload.component) !== "transcription") {
        return prev;
      }
      return {
        ...prev,
        transcriptionModelStatus: toModelStatus(msg.payload.status, prev.transcriptionModelStatus)
      };
    case "transcription_final": {
      const text = String(msg.payload.text ?? "");
      if (!text.trim() || isDuplicateFinalLine(prev.transcriptLines, text)) {
        return {
          ...prev,
          partialText: "",
          partialPhase: undefined,
          partialConfidence: undefined,
          latencyMs: Number(msg.payload.latencyMs ?? prev.latencyMs)
        };
      }
      return {
        ...prev,
        transcriptLines: [...prev.transcriptLines, text],
        partialText: "",
        partialPhase: undefined,
        partialConfidence: undefined,
        latencyMs: Number(msg.payload.latencyMs ?? prev.latencyMs)
      };
    }
    case "error":
      return { ...prev, lastError: String(msg.payload.message ?? "Unknown error") };
    default:
      return prev;
  }
}

function trimConversationLines(state: CaptureState, maxLines: number): CaptureState {
  if (state.transcriptLines.length <= maxLines) {
    return state;
  }
  return {
    ...state,
    transcriptLines: state.transcriptLines.slice(state.transcriptLines.length - maxLines)
  };
}

function mergeRawLines(lines: string[], value: string): string[] {
  if (lines.length === 0) {
    return [value];
  }

  const lastIndex = lines.length - 1;
  const previous = lines[lastIndex];
  if (previous === value) {
    return lines;
  }

  if (shouldReplaceRawLine(previous, value)) {
    const next = [...lines];
    next[lastIndex] = value;
    return next;
  }

  return [...lines, value];
}

function shouldReplaceRawLine(previous: string, next: string): boolean {
  if (next.startsWith(previous) || previous.startsWith(next)) {
    return true;
  }

  const previousWords = wordCount(previous);
  const nextWords = wordCount(next);
  if (previousWords <= 3 || nextWords <= 3) {
    return true;
  }

  const commonPrefixRatio = longestCommonPrefixLength(previous, next) / Math.max(1, Math.min(previous.length, next.length));
  return commonPrefixRatio >= 0.45;
}

function wordCount(value: string): number {
  return value.trim().split(/\s+/).filter(Boolean).length;
}

function longestCommonPrefixLength(a: string, b: string): number {
  const max = Math.min(a.length, b.length);
  let i = 0;
  while (i < max && a[i] === b[i]) {
    i += 1;
  }
  return i;
}

function isDuplicateFinalLine(lines: string[], candidate: string): boolean {
  const normalizedCandidate = normalizeFinalLine(candidate);
  if (!normalizedCandidate) {
    return true;
  }

  const recent = lines.slice(Math.max(0, lines.length - 8));
  for (const line of recent) {
    const normalizedExisting = normalizeFinalLine(line);
    if (!normalizedExisting) {
      continue;
    }
    if (normalizedCandidate === normalizedExisting) {
      return true;
    }
    if (
      normalizedCandidate.length >= 24 &&
      (normalizedCandidate.includes(normalizedExisting) || normalizedExisting.includes(normalizedCandidate))
    ) {
      return true;
    }
  }
  return false;
}

function normalizeFinalLine(value: string): string {
  const compact = value.trim().toLowerCase().replace(/\s+/g, " ");
  return compact.replace(/[^a-z0-9 ]/g, "");
}

function toModelStatus(
  value: unknown,
  fallback?: "not_downloaded" | "downloading" | "ready"
): "not_downloaded" | "downloading" | "ready" | undefined {
  const v = String(value ?? "");
  if (v === "not_downloaded" || v === "downloading" || v === "ready") {
    return v;
  }
  return fallback;
}

function toPartialPhase(
  value: unknown,
  fallback?: "interim" | "refined"
): "interim" | "refined" | undefined {
  const v = String(value ?? "");
  if (v === "interim" || v === "refined") {
    return v;
  }
  return fallback;
}

function toFiniteNumber(value: unknown, fallback?: number): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}
