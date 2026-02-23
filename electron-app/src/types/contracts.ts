export type ComputeMode = "cpu" | "cuda";
export type TranscriptionPreset = "ultra_fast" | "balanced" | "high_accuracy" | "custom";
export type AiProvider = "ollama" | "ollama_cloud" | "openai";

export interface AppSettings {
  sourceLanguage: string;
  whisperModel: string;
  computeMode: ComputeMode;
  transcriptionPreset: TranscriptionPreset;
  enableNlp: boolean;
  enableRawWhisper: boolean;
  enableVad: boolean;
  aiProvider: AiProvider;
  aiModel: string;
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
    | "question_status"
    | "question_suggestion"
    | "error";
  payload: Record<string, unknown>;
}

export interface TranscriptLine {
  text: string;
  isQuestion?: boolean;
  questionScore?: number;
}

export interface CaptureState {
  listening: boolean;
  activeAudio?: boolean;
  rms?: number;
  latencyMs: number;
  partialText: string;
  partialIsQuestion?: boolean;
  partialQuestionScore?: number;
  partialPhase?: "interim" | "refined";
  partialConfidence?: number;
  rawTranscriptLines: string[];
  transcriptLines: TranscriptLine[];
  answerLines: TranscriptLine[];
  latestQuestionSuggestion?: string;
  questionStatus?: "idle" | "queued" | "processing" | "answered" | "error";
  latestQuestionText?: string;
  latestQuestionError?: string;
  transcriptionModelStatus?: "not_downloaded" | "downloading" | "ready";
  runtimeDevice?: string;
  runtimeComputeType?: string;
  lastError?: string;
}
