export type ComputeMode = "cpu" | "cuda";
export type TranscriptionPreset = "ultra_fast" | "balanced" | "high_accuracy" | "custom";

export interface AppSettings {
  sourceLanguage: string;
  whisperModel: string;
  computeMode: ComputeMode;
  transcriptionPreset: TranscriptionPreset;
  enableNlp: boolean;
  enableRawWhisper: boolean;
  enableVad: boolean;
}

export interface AudioDevice {
  id: string;
  name: string;
}

export interface EngineEvent {
  type:
    | "status"
    | "model_status"
    | "transcription_raw"
    | "transcription_partial"
    | "transcription_final"
    | "error";
  payload: Record<string, unknown>;
}

export interface CaptureState {
  listening: boolean;
  activeAudio?: boolean;
  rms?: number;
  latencyMs: number;
  partialText: string;
  partialPhase?: "interim" | "refined";
  partialConfidence?: number;
  rawTranscriptLines: string[];
  transcriptLines: string[];
  transcriptionModelStatus?: "not_downloaded" | "downloading" | "ready";
  runtimeDevice?: string;
  runtimeComputeType?: string;
  lastError?: string;
}
