from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol, serve

from app.core import ConfigStore, RuntimeConfig
from app.nlp import NlpPipeline, StreamingTranscriptionStabilizer
from app.transcription import TranscriptionEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_RING_SECONDS = 8
MICRO_INFER_WINDOW_SECONDS = 0.45
CONTEXT_INFER_WINDOW_SECONDS = 2.4
MIN_INFER_SECONDS = 0.18
MICRO_INFER_INTERVAL_SECONDS = 0.16
CONTEXT_INFER_INTERVAL_SECONDS = 0.4
AUDIO_ACTIVE_ON_RMS_THRESHOLD = 0.0009
AUDIO_ACTIVE_OFF_RMS_THRESHOLD = 0.00045
RMS_EMA_ALPHA = 0.2
IDLE_TIMEOUT_SECONDS = 1.4
MAX_QUEUE_CHUNKS = 64
MAX_PENDING_CHUNKS_PER_PASS = 48
STATUS_EMIT_INTERVAL_SECONDS = 0.08
FORCED_FINALIZE_INTERVAL_SECONDS = 5.5
FORCED_FINALIZE_MIN_WORDS = 18
PARTIAL_STALL_FINALIZE_SECONDS = 1.6
WS_PING_INTERVAL_SECONDS = 20
WS_PING_TIMEOUT_SECONDS = 90


class EngineServer:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port

        self._config_store = ConfigStore()
        self._nlp = NlpPipeline()
        self._transcriber = TranscriptionEngine()

        self._ui_clients: set[WebSocketServerProtocol] = set()
        self._audio_clients: set[WebSocketServerProtocol] = set()
        self._audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=MAX_QUEUE_CHUNKS)

        self._last_audio_ts: float = 0.0
        self._last_active_audio_ts: float = 0.0
        self._stabilizer = StreamingTranscriptionStabilizer()
        self._latest_stable_partial: str = ""
        self._next_sentence_id = 0
        self._last_runtime_signature = ""
        self._audio_active = False
        self._rms_ema = 0.0
        self._last_status_emit_ts = 0.0
        self._last_forced_finalize_ts = 0.0
        self._last_partial_change_ts = 0.0
        self._latest_segment_timestamps: list[dict[str, object]] = []
        self._latest_partial_confidence: float = 0.0
        self._recent_finalized: deque[str] = deque(maxlen=24)
        self._ready_transcription_models: set[str] = set()
        self._warmup_task: asyncio.Task[None] | None = None

    async def run_forever(self) -> None:
        processor_task = asyncio.create_task(self._audio_processor(), name="audio-processor")
        idle_task = asyncio.create_task(self._idle_monitor(), name="idle-monitor")
        async with serve(
            self._route_handler,
            self._host,
            self._port,
            ping_interval=WS_PING_INTERVAL_SECONDS,
            ping_timeout=WS_PING_TIMEOUT_SECONDS,
        ):
            logger.info("Engine server listening at ws://%s:%s", self._host, self._port)
            await asyncio.Future()

        processor_task.cancel()
        idle_task.cancel()
        if self._warmup_task is not None:
            self._warmup_task.cancel()
        await self._transcriber.close()

    async def _route_handler(self, websocket: WebSocketServerProtocol, path: str) -> None:
        if path == "/audio":
            await self._handle_audio_socket(websocket)
            return
        if path == "/ws":
            await self._handle_ui_socket(websocket)
            return
        await websocket.close(code=1008, reason="Unsupported path")

    async def _handle_audio_socket(self, websocket: WebSocketServerProtocol) -> None:
        self._audio_clients.add(websocket)
        await self._broadcast({"type": "status", "payload": {"listening": True, "activeAudio": False, "latencyMs": 0, "rms": self._rms_ema}})

        try:
            async for message in websocket:
                if not isinstance(message, (bytes, bytearray)):
                    continue
                chunk = np.frombuffer(message, dtype=np.float32)
                if chunk.size == 0:
                    continue

                self._last_audio_ts = time.perf_counter()
                rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float32)))
                self._rms_ema = (RMS_EMA_ALPHA * rms) + ((1.0 - RMS_EMA_ALPHA) * self._rms_ema)
                speech_detected = (
                    rms >= AUDIO_ACTIVE_ON_RMS_THRESHOLD
                    or self._rms_ema >= AUDIO_ACTIVE_ON_RMS_THRESHOLD
                    or (self._audio_active and (rms >= AUDIO_ACTIVE_OFF_RMS_THRESHOLD or self._rms_ema >= AUDIO_ACTIVE_OFF_RMS_THRESHOLD))
                )
                if speech_detected:
                    self._last_active_audio_ts = self._last_audio_ts
                    if not self._audio_active:
                        self._audio_active = True
                        self._last_forced_finalize_ts = self._last_audio_ts
                        await self._broadcast(
                            {
                                "type": "status",
                                "payload": {"listening": True, "activeAudio": True, "latencyMs": 0, "rms": self._rms_ema},
                            }
                        )
                if (self._last_audio_ts - self._last_status_emit_ts) >= STATUS_EMIT_INTERVAL_SECONDS:
                    self._last_status_emit_ts = self._last_audio_ts
                    await self._broadcast(
                        {
                            "type": "status",
                            "payload": {"listening": True, "activeAudio": self._audio_active, "latencyMs": 0, "rms": self._rms_ema},
                        }
                    )

                try:
                    self._audio_queue.put_nowait(chunk.copy())
                except asyncio.QueueFull:
                    _ = self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait(chunk.copy())
        except ConnectionClosed:
            logger.info("Audio client disconnected")
        finally:
            self._audio_clients.discard(websocket)
            self._audio_active = False
            self._rms_ema = 0.0
            self._last_status_emit_ts = 0.0
            self._last_partial_change_ts = 0.0
            if not self._audio_clients:
                await self._broadcast({"type": "status", "payload": {"listening": False, "activeAudio": False, "latencyMs": 0, "rms": self._rms_ema}})

    async def _handle_ui_socket(self, websocket: WebSocketServerProtocol) -> None:
        self._ui_clients.add(websocket)
        await self._send_json(
            websocket,
            {
                "type": "status",
                "payload": {
                    "listening": bool(self._audio_clients),
                    "activeAudio": self._audio_active,
                    "latencyMs": 0,
                    "rms": self._rms_ema,
                },
            },
        )

        try:
            async for raw in websocket:
                if not isinstance(raw, str):
                    continue
                await self._handle_ui_message(raw)
        except ConnectionClosed:
            logger.info("UI client disconnected")
        finally:
            self._ui_clients.discard(websocket)

    async def _handle_ui_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._broadcast_error("Invalid JSON control message")
            return

        msg_type = msg.get("type")
        payload = msg.get("payload") or {}

        if msg_type == "update_config":
            normalized = {
                "source_language": payload.get("sourceLanguage", "en"),
                "whisper_model": payload.get("whisperModel", "medium"),
                "compute_mode": payload.get("computeMode", "cuda"),
                "enable_nlp": bool(payload.get("enableNlp", True)),
                "enable_raw_whisper": bool(payload.get("enableRawWhisper", False)),
                "enable_vad": bool(payload.get("enableVad", True)),
            }
            config = await self._config_store.update(normalized)
            await self._emit_model_status(
                component="transcription",
                model=config.whisper_model,
                status="ready" if config.whisper_model in self._ready_transcription_models else "not_downloaded",
            )
            if self._warmup_task is not None:
                self._warmup_task.cancel()
            self._warmup_task = asyncio.create_task(self._warmup_models(config), name="model-warmup")
            logger.info("Engine config updated")
            return

        if msg_type == "clear":
            self._stabilizer = StreamingTranscriptionStabilizer()
            self._latest_stable_partial = ""
            self._next_sentence_id = 0
            self._last_forced_finalize_ts = 0.0
            self._last_partial_change_ts = 0.0
            self._latest_segment_timestamps = []
            self._latest_partial_confidence = 0.0
            self._recent_finalized.clear()
            return

    async def _warmup_models(self, config: RuntimeConfig) -> None:
        try:
            await self._emit_model_status("transcription", config.whisper_model, "downloading")
            await self._transcriber.warmup(config)
            self._ready_transcription_models.add(config.whisper_model)
            await self._emit_model_status("transcription", config.whisper_model, "ready")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Model warmup failed")
            await self._broadcast_error(str(exc))

    async def _emit_model_status(self, component: str, model: str, status: str) -> None:
        await self._broadcast(
            {
                "type": "model_status",
                "payload": {
                    "component": component,
                    "model": model,
                    "status": status,
                },
            }
        )

    async def _audio_processor(self) -> None:
        ring = np.zeros(0, dtype=np.float32)
        last_micro_infer = 0.0
        last_context_infer = 0.0

        while True:
            chunk = await self._audio_queue.get()
            pending = [chunk]
            backlog = self._audio_queue.qsize()
            if backlog > MAX_PENDING_CHUNKS_PER_PASS:
                drop_count = backlog - MAX_PENDING_CHUNKS_PER_PASS
                for _ in range(drop_count):
                    _ = self._audio_queue.get_nowait()
            while not self._audio_queue.empty():
                pending.append(self._audio_queue.get_nowait())

            merged = np.concatenate(pending)
            ring = np.concatenate((ring, merged))
            max_samples = SAMPLE_RATE * MAX_RING_SECONDS
            if ring.size > max_samples:
                ring = ring[-max_samples:]

            now = time.perf_counter()
            if ring.size < int(SAMPLE_RATE * MIN_INFER_SECONDS):
                continue

            config = await self._config_store.get()

            try:
                if now - last_micro_infer >= MICRO_INFER_INTERVAL_SECONDS:
                    micro_samples = int(SAMPLE_RATE * MICRO_INFER_WINDOW_SECONDS)
                    micro_audio = ring[-micro_samples:] if ring.size > micro_samples else ring
                    micro_result = await self._transcriber.transcribe(micro_audio.copy(), config)
                    await self._emit_runtime_status()
                    await self._process_transcript(micro_result, config, phase="interim")
                    last_micro_infer = now

                if now - last_context_infer >= CONTEXT_INFER_INTERVAL_SECONDS:
                    context_samples = int(SAMPLE_RATE * CONTEXT_INFER_WINDOW_SECONDS)
                    context_audio = ring[-context_samples:] if ring.size > context_samples else ring
                    context_result = await self._transcriber.transcribe(context_audio.copy(), config)
                    await self._emit_runtime_status()
                    await self._process_transcript(context_result, config, phase="refined")
                    last_context_infer = now
            except Exception as exc:
                logger.exception("Processing error")
                await self._broadcast_error(str(exc))

    async def _emit_runtime_status(self) -> None:
        runtime = self._transcriber.runtime_info()
        signature = f"{runtime['device']}:{runtime['computeType']}"
        if signature == self._last_runtime_signature:
            return
        self._last_runtime_signature = signature
        await self._broadcast(
            {
                "type": "status",
                "payload": {
                    "listening": bool(self._audio_clients),
                    "activeAudio": self._audio_active,
                    "latencyMs": 0,
                    "rms": self._rms_ema,
                    "device": runtime["device"],
                    "computeType": runtime["computeType"],
                },
            }
        )

    async def _idle_monitor(self) -> None:
        while True:
            await asyncio.sleep(0.2)
            if not self._audio_active:
                continue
            if not self._audio_clients:
                self._audio_active = False
                self._rms_ema = 0.0
                continue
            now = time.perf_counter()
            if now - self._last_active_audio_ts >= IDLE_TIMEOUT_SECONDS:
                self._audio_active = False
                config = await self._config_store.get()
                await self._finalize_pending_partial(config)
                await self._broadcast(
                    {
                        "type": "status",
                        "payload": {
                            "listening": True,
                            "activeAudio": False,
                            "latencyMs": 0,
                            "rms": self._rms_ema,
                        },
                    }
                )
                continue

            if (
                self._latest_stable_partial.strip()
                and self._last_partial_change_ts > 0
                and (now - self._last_partial_change_ts) >= PARTIAL_STALL_FINALIZE_SECONDS
            ):
                config = await self._config_store.get()
                await self._finalize_pending_partial(config)
                self._last_partial_change_ts = now

    async def _process_transcript(self, transcript_result: dict[str, Any], config: RuntimeConfig, phase: str) -> None:
        raw_text = self._nlp.normalize_text(str(transcript_result.get("text", "")))
        segments = transcript_result.get("segments", [])
        self._latest_segment_timestamps = segments if isinstance(segments, list) else []
        confidence = self._estimate_transcription_confidence(self._latest_segment_timestamps)
        self._latest_partial_confidence = confidence
        if not raw_text:
            return

        latency_ms = int((time.perf_counter() - self._last_audio_ts) * 1000) if self._last_audio_ts else 0
        if phase == "interim":
            await self._broadcast(
                {
                    "type": "transcription_partial",
                    "payload": {
                        "text": raw_text,
                        "latencyMs": latency_ms,
                        "timestamps": self._latest_segment_timestamps,
                        "phase": "interim",
                        "confidence": confidence,
                    },
                }
            )
            return

        segment_finalized = self._extract_stable_segment_sentences(self._latest_segment_timestamps)
        if segment_finalized:
            await self._emit_finalized_sentences(segment_finalized, config, latency_ms)

        stable_partial = self._stabilizer.process_partial(raw_text)
        if stable_partial != self._latest_stable_partial:
            self._latest_stable_partial = stable_partial
            self._last_partial_change_ts = time.perf_counter()
            if config.enable_raw_whisper and stable_partial:
                await self._broadcast(
                    {
                        "type": "transcription_raw",
                        "payload": {
                            "text": stable_partial,
                            "latencyMs": latency_ms,
                            "timestamps": self._latest_segment_timestamps,
                        },
                    }
                )
            await self._broadcast(
                {
                    "type": "transcription_partial",
                    "payload": {
                        "text": stable_partial,
                        "latencyMs": latency_ms,
                        "timestamps": self._latest_segment_timestamps,
                        "phase": "refined",
                        "confidence": confidence,
                    },
                }
            )

        if self._should_force_finalize(stable_partial):
            forced = self._stabilizer.process_final(stable_partial, silence=False, vad_end=False)
            forced_finalized = [str(x) for x in forced.get("finalized", []) if str(x).strip()]
            if forced_finalized:
                self._last_forced_finalize_ts = time.perf_counter()
                await self._emit_finalized_sentences(forced_finalized, config, latency_ms)
            next_partial = str(forced.get("partial", "")).strip()
            if next_partial != self._latest_stable_partial:
                self._latest_stable_partial = next_partial
                self._last_partial_change_ts = time.perf_counter()
                await self._broadcast(
                    {
                        "type": "transcription_partial",
                        "payload": {
                            "text": next_partial,
                            "latencyMs": latency_ms,
                            "timestamps": self._latest_segment_timestamps,
                            "phase": "refined",
                            "confidence": confidence,
                        },
                    }
                )

        finalized = self._stabilizer.consume_pending_finalized()
        if finalized:
            await self._emit_finalized_sentences(finalized, config, latency_ms)

    def _extract_stable_segment_sentences(self, raw_segments: list[dict[str, object]]) -> list[str]:
        if not raw_segments:
            return []

        max_end = 0.0
        for segment in raw_segments:
            end = segment.get("end")
            if isinstance(end, (int, float)) and float(end) > max_end:
                max_end = float(end)

        out: list[str] = []
        for segment in raw_segments:
            text = self._nlp.normalize_text(str(segment.get("text", "")))
            if not text:
                continue

            end = segment.get("end")
            end_f = float(end) if isinstance(end, (int, float)) else None
            ends_with_terminal = text.endswith((".", "?", "!"))
            words = len(text.split())

            is_trailing_unstable = end_f is None or (max_end - end_f) < 0.35
            if is_trailing_unstable and not ends_with_terminal:
                continue
            if words < 3 and not ends_with_terminal:
                continue
            out.append(text)
        return out

    async def _finalize_pending_partial(self, config: RuntimeConfig) -> None:
        if not self._latest_stable_partial.strip():
            return
        latency_ms = int((time.perf_counter() - self._last_audio_ts) * 1000) if self._last_audio_ts else 0
        final_result = self._stabilizer.process_final(self._latest_stable_partial, silence=True, vad_end=False)
        finalized = [str(x) for x in final_result.get("finalized", []) if str(x).strip()]
        if finalized:
            await self._emit_finalized_sentences(finalized, config, latency_ms)
        next_partial = str(final_result.get("partial", "")).strip()
        self._latest_stable_partial = next_partial
        await self._broadcast(
            {
                "type": "transcription_partial",
                "payload": {
                    "text": next_partial,
                    "latencyMs": latency_ms,
                    "phase": "refined",
                    "confidence": self._latest_partial_confidence,
                },
            }
        )

    async def _emit_finalized_sentences(self, sentences: list[str], config: RuntimeConfig, latency_ms: int) -> None:
        for final_sentence in sentences:
            if not final_sentence:
                continue
            if self._is_duplicate_final(final_sentence):
                continue
            sentence_id = self._next_sentence_id
            self._next_sentence_id += 1
            self._remember_final(final_sentence)
            detected = (
                self._nlp.detect_language(final_sentence, default=config.source_language)
                if config.enable_nlp
                else config.source_language
            )
            await self._broadcast(
                {
                    "type": "transcription_final",
                    "payload": {
                        "sentenceId": sentence_id,
                        "text": final_sentence,
                        "language": detected,
                        "latencyMs": latency_ms,
                        "timestamps": self._latest_segment_timestamps,
                    },
                }
            )

    def _remember_final(self, sentence: str) -> None:
        normalized = self._normalize_sentence(sentence)
        if normalized:
            self._recent_finalized.append(normalized)

    def _is_duplicate_final(self, sentence: str) -> bool:
        candidate = self._normalize_sentence(sentence)
        if not candidate:
            return True
        for previous in self._recent_finalized:
            if candidate == previous:
                return True
            ratio = SequenceMatcher(None, candidate, previous).ratio()
            if ratio >= 0.94:
                return True
            if len(candidate) >= 24 and (candidate in previous or previous in candidate):
                return True
        return False

    def _normalize_sentence(self, value: str) -> str:
        lowered = value.strip().lower()
        compact = " ".join(lowered.split())
        return "".join(ch for ch in compact if ch.isalnum() or ch.isspace())

    def _should_force_finalize(self, partial: str) -> bool:
        if not partial:
            return False
        if not self._audio_active:
            return False
        words = len(partial.split())
        if words < FORCED_FINALIZE_MIN_WORDS:
            return False
        now = time.perf_counter()
        return (now - self._last_forced_finalize_ts) >= FORCED_FINALIZE_INTERVAL_SECONDS

    def _estimate_transcription_confidence(self, segments: list[dict[str, object]]) -> float:
        if not segments:
            return 0.0
        weighted_total = 0.0
        weight_sum = 0.0
        for segment in segments:
            conf = self._segment_confidence(segment)
            start = segment.get("start")
            end = segment.get("end")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                duration = max(0.12, float(end) - float(start))
            else:
                duration = max(0.12, len(str(segment.get("text", ""))) / 16.0)
            weighted_total += conf * duration
            weight_sum += duration
        if weight_sum <= 0:
            return 0.0
        value = weighted_total / weight_sum
        return max(0.0, min(1.0, value))

    def _segment_confidence(self, segment: dict[str, object]) -> float:
        avg_logprob_raw = segment.get("avg_logprob")
        no_speech_raw = segment.get("no_speech_prob")
        avg_logprob = float(avg_logprob_raw) if isinstance(avg_logprob_raw, (int, float)) else -1.5
        no_speech_prob = float(no_speech_raw) if isinstance(no_speech_raw, (int, float)) else 1.0

        logprob_score = (avg_logprob + 1.5) / 1.4
        logprob_score = max(0.0, min(1.0, logprob_score))
        speech_score = max(0.0, min(1.0, 1.0 - no_speech_prob))
        return (0.65 * logprob_score) + (0.35 * speech_score)

    async def _broadcast_error(self, message: str) -> None:
        await self._broadcast({"type": "error", "payload": {"message": message}})

    async def _broadcast(self, data: dict[str, Any]) -> None:
        if not self._ui_clients:
            return
        await asyncio.gather(
            *[self._send_json(ws, data) for ws in list(self._ui_clients)],
            return_exceptions=True,
        )

    async def _send_json(self, ws: WebSocketServerProtocol, data: dict[str, Any]) -> None:
        try:
            await ws.send(json.dumps(data))
        except Exception:
            self._ui_clients.discard(ws)
