from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class RuntimeConfig:
    source_language: str = "en"
    whisper_model: str = "medium"
    compute_mode: str = "cuda"
    enable_nlp: bool = True
    enable_raw_whisper: bool = False
    enable_vad: bool = True


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
