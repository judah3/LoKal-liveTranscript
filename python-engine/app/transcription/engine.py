from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from app.core import RuntimeConfig

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._model: WhisperModel | None = None
        self._loaded_key: tuple[str, str] | None = None
        self._model_lock = asyncio.Lock()
        self._runtime_device = "uninitialized"
        self._runtime_compute_type = "unknown"

    async def transcribe(self, audio: np.ndarray, config: RuntimeConfig) -> dict[str, Any]:
        await self._ensure_model(config)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe_sync, audio, config)

    async def warmup(self, config: RuntimeConfig) -> None:
        await self._ensure_model(config)

    def runtime_info(self) -> dict[str, str]:
        return {
            "device": self._runtime_device,
            "computeType": self._runtime_compute_type,
        }

    async def _ensure_model(self, config: RuntimeConfig) -> None:
        async with self._model_lock:
            key = (config.whisper_model, config.compute_mode)
            if self._model is not None and self._loaded_key == key:
                return

            logger.info("Loading whisper model=%s compute=%s", config.whisper_model, config.compute_mode)
            device = "cuda" if config.compute_mode == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            loop = asyncio.get_running_loop()

            def loader() -> WhisperModel:
                return WhisperModel(
                    config.whisper_model,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4,
                    num_workers=1,
                )

            try:
                self._model = await loop.run_in_executor(self._executor, loader)
            except Exception as exc:
                if device == "cuda":
                    raise RuntimeError(
                        "CUDA load failed for faster-whisper/CTranslate2. "
                        "Install CUDA-enabled runtime deps (ctranslate2/cudnn) or switch to CPU."
                    ) from exc
                raise
            self._loaded_key = key
            self._runtime_device = device
            self._runtime_compute_type = compute_type

    def _transcribe_sync(self, audio: np.ndarray, config: RuntimeConfig) -> dict[str, Any]:
        assert self._model is not None
        segments, _ = self._model.transcribe(
            audio,
            language=None if config.source_language == "auto" else config.source_language,
            beam_size=1,
            best_of=1,
            vad_filter=config.enable_vad,
            temperature=0,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        items = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            items.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                    "avg_logprob": float(getattr(segment, "avg_logprob", -1.5)),
                    "no_speech_prob": float(getattr(segment, "no_speech_prob", 1.0)),
                }
            )
        full_text = " ".join(item["text"] for item in items).strip()
        return {"text": full_text, "segments": items}

    async def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
