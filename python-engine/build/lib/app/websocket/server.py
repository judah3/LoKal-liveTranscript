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
from app.translation import TranslationEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_RING_SECONDS = 8
INFER_WINDOW_SECONDS = 4.5
MIN_INFER_SECONDS = 0.25
INFER_INTERVAL_SECONDS = 0.3
AUDIO_ACTIVE_ON_RMS_THRESHOLD = 0.0009
AUDIO_ACTIVE_OFF_RMS_THRESHOLD = 0.00045
RMS_EMA_ALPHA = 0.2
IDLE_TIMEOUT_SECONDS = 2.6
PARTIAL_TRANSLATION_INTERVAL_SECONDS = 0.75
MAX_QUEUE_CHUNKS = 220
MAX_PENDING_CHUNKS_PER_PASS = 120
STATUS_EMIT_INTERVAL_SECONDS = 0.15
FORCED_FINALIZE_INTERVAL_SECONDS = 5.5
FORCED_FINALIZE_MIN_WORDS = 18
MAX_PARTIAL_TRANSLATION_CHARS = 320
PARTIAL_STALL_FINALIZE_SECONDS = 3.4
WS_PING_INTERVAL_SECONDS = 20
WS_PING_TIMEOUT_SECONDS = 90


class EngineServer:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port

        self._config_store = ConfigStore()
        self._nlp = NlpPipeline()
        self._transcriber = TranscriptionEngine()
        self._translator = TranslationEngine()

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
        self._last_partial_translation_ts = 0.0
        self._last_forced_finalize_ts = 0.0
        self._last_partial_change_ts = 0.0
        self._latest_segment_timestamps: list[dict[str, object]] = []
        self._final_translation_queue: asyncio.Queue[tuple[int, str, RuntimeConfig, int]] = asyncio.Queue(maxsize=200)
        self._partial_translation_queue: asyncio.Queue[tuple[int, str, RuntimeConfig, int]] = asyncio.Queue(maxsize=1)
        self._partial_translation_seq = 0
        self._recent_finalized: deque[str] = deque(maxlen=24)
        self._ready_transcription_models: set[str] = set()
        self._ready_translation_models: set[str] = set()
        self._warmup_task: asyncio.Task[None] | None = None

    async def run_forever(self) -> None:
        processor_task = asyncio.create_task(self._audio_processor(), name="audio-processor")
        idle_task = asyncio.create_task(self._idle_monitor(), name="idle-monitor")
        partial_translation_task = asyncio.create_task(self._partial_translation_worker(), name="partial-translation-worker")
        final_translation_task = asyncio.create_task(self._final_translation_worker(), name="final-translation-worker")
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
        partial_translation_task.cancel()
        final_translation_task.cancel()
        if self._warmup_task is not None:
            self._warmup_task.cancel()
        await self._transcriber.close()
        await self._translator.close()

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
                "target_language": payload.get("targetLanguage", "es"),
                "whisper_model": payload.get("whisperModel", "medium"),
                "translation_model": payload.get("translationModel", "Helsinki-NLP/opus-mt-en-es"),
                "compute_mode": payload.get("computeMode", "cuda"),
                "enable_nlp": bool(payload.get("enableNlp", True)),
                "enable_raw_whisper": bool(payload.get("enableRawWhisper", False)),
            }
            config = await self._config_store.update(normalized)
            await self._emit_model_status(
                component="transcription",
                model=config.whisper_model,
                status="ready" if config.whisper_model in self._ready_transcription_models else "not_downloaded",
            )
            await self._emit_model_status(
                component="translation",
                model=config.translation_model,
                status="ready" if config.translation_model in self._ready_translation_models else "not_downloaded",
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
            self._last_partial_translation_ts = 0.0
            self._last_forced_finalize_ts = 0.0
            self._last_partial_change_ts = 0.0
            self._latest_segment_timestamps = []
            self._partial_translation_seq = 0
            self._recent_finalized.clear()
            while not self._partial_translation_queue.empty():
                _ = self._partial_translation_queue.get_nowait()
            while not self._final_translation_queue.empty():
                _ = self._final_translation_queue.get_nowait()
            return

    async def _warmup_models(self, config: RuntimeConfig) -> None:
        try:
            await self._emit_model_status("transcription", config.whisper_model, "downloading")
            await self._transcriber.warmup(config)
            self._ready_transcription_models.add(config.whisper_model)
            await self._emit_model_status("transcription", config.whisper_model, "ready")

            await self._emit_model_status("translation", config.translation_model, "downloading")
            await self._translator.warmup(config)
            self._ready_translation_models.add(config.translation_model)
            await self._emit_model_status("translation", config.translation_model, "ready")
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
        last_infer = 0.0

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
            if now - last_infer < INFER_INTERVAL_SECONDS or ring.size < int(SAMPLE_RATE * MIN_INFER_SECONDS):
                continue

            last_infer = now
            config = await self._config_store.get()

            try:
                infer_samples = int(SAMPLE_RATE * INFER_WINDOW_SECONDS)
                infer_audio = ring[-infer_samples:] if ring.size > infer_samples else ring
                transcript_result = await self._transcriber.transcribe(infer_audio.copy(), config)
                await self._emit_runtime_status()
                await self._process_transcript(transcript_result, config)
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

    async def _process_transcript(self, transcript_result: dict[str, Any], config: RuntimeConfig) -> None:
        raw_text = self._nlp.normalize_text(str(transcript_result.get("text", "")))
        segments = transcript_result.get("segments", [])
        self._latest_segment_timestamps = segments if isinstance(segments, list) else []
        if not raw_text:
            return

        latency_ms = int((time.perf_counter() - self._last_audio_ts) * 1000) if self._last_audio_ts else 0
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
                    },
                }
            )
            await self._enqueue_partial_translation(stable_partial, config, latency_ms)

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
                        },
                    }
                )
                await self._enqueue_partial_translation(next_partial, config, latency_ms)

        finalized = self._stabilizer.consume_pending_finalized()
        if finalized:
            await self._emit_finalized_sentences(finalized, config, latency_ms)

        # Do not finalize on every overlapping decode pass; this causes repeated
        # short fragments to be emitted as new lines. Finalization is handled on
        # silence/idle boundaries.

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
                "payload": {"text": next_partial, "latencyMs": latency_ms},
            }
        )
        await self._broadcast(
            {
                "type": "translation_partial",
                "payload": {"text": "" if not next_partial else str(final_result.get("partial", "")), "latencyMs": latency_ms},
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
            if self._final_translation_queue.full():
                _ = self._final_translation_queue.get_nowait()
            await self._final_translation_queue.put((sentence_id, final_sentence, config, latency_ms))

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

    async def _enqueue_partial_translation(
        self, cleaned: str, config: RuntimeConfig, latency_ms: int
    ) -> None:
        clipped = self._clip_partial_for_translation(cleaned)
        if not clipped:
            return
        now = time.perf_counter()
        if now - self._last_partial_translation_ts < PARTIAL_TRANSLATION_INTERVAL_SECONDS:
            return
        self._last_partial_translation_ts = now
        self._partial_translation_seq += 1
        seq = self._partial_translation_seq
        while not self._partial_translation_queue.empty():
            _ = self._partial_translation_queue.get_nowait()
        await self._partial_translation_queue.put((seq, clipped, config, latency_ms))

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

    def _clip_partial_for_translation(self, text: str) -> str:
        value = text.strip()
        if not value:
            return ""
        if len(value) <= MAX_PARTIAL_TRANSLATION_CHARS:
            return value
        clipped = value[-MAX_PARTIAL_TRANSLATION_CHARS:]
        boundary = clipped.find(" ")
        return clipped[boundary + 1 :].strip() if boundary >= 0 else clipped

    async def _partial_translation_worker(self) -> None:
        while True:
            seq, text, config, latency_ms = await self._partial_translation_queue.get()
            try:
                detected = (
                    self._nlp.detect_language(text, default=config.source_language)
                    if config.enable_nlp
                    else config.source_language
                )
                translated = await self._translator.translate(text, config, detected_source=detected)
                if seq != self._partial_translation_seq:
                    continue
                await self._broadcast(
                    {
                        "type": "translation_partial",
                        "payload": {"text": translated, "latencyMs": latency_ms},
                    }
                )
            except Exception as exc:
                logger.exception("Partial translation worker error")
                await self._broadcast_error(str(exc))

    async def _final_translation_worker(self) -> None:
        while True:
            sentence_id, text, config, latency_ms = await self._final_translation_queue.get()
            try:
                detected = (
                    self._nlp.detect_language(text, default=config.source_language)
                    if config.enable_nlp
                    else config.source_language
                )
                translated = await self._translator.translate(text, config, detected_source=detected)
                await self._broadcast(
                    {
                        "type": "translation_final",
                        "payload": {"sentenceId": sentence_id, "text": translated, "latencyMs": latency_ms},
                    }
                )
            except Exception as exc:
                logger.exception("Final translation worker error")
                await self._broadcast_error(str(exc))

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
