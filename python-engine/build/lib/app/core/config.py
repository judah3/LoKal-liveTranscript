from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class RuntimeConfig:
    source_language: str = "en"
    target_language: str = "es"
    whisper_model: str = "medium"
    translation_model: str = "Helsinki-NLP/opus-mt-en-es"
    compute_mode: str = "cuda"
    enable_nlp: bool = True
    enable_raw_whisper: bool = False


class ConfigStore:
    def __init__(self) -> None:
        self._config = RuntimeConfig()
        self._lock = asyncio.Lock()

    async def get(self) -> RuntimeConfig:
        async with self._lock:
            return RuntimeConfig(**asdict(self._config))

    async def update(self, raw: dict[str, Any]) -> RuntimeConfig:
        async with self._lock:
            for key, value in raw.items():
                if not hasattr(self._config, key):
                    continue
                setattr(self._config, key, value)
            return RuntimeConfig(**asdict(self._config))
