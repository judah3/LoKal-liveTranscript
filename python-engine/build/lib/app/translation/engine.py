from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from app.core import RuntimeConfig

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

NLLB_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
}


class TranslationEngine:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translator")
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None
        self._loaded_key: tuple[str, str] | None = None
        self._model_lock = asyncio.Lock()
        self._runtime_device = "cpu"
        self._fallback: tuple[PreTrainedTokenizerBase, PreTrainedModel] | None = None

    async def translate(self, text: str, config: RuntimeConfig, detected_source: str | None = None) -> str:
        if not text.strip():
            return ""

        await self._ensure_model(config.translation_model, config.compute_mode)
        loop = asyncio.get_running_loop()

        source = detected_source or config.source_language
        target = config.target_language
        return await loop.run_in_executor(self._executor, self._translate_sync, text, source, target)

    async def warmup(self, config: RuntimeConfig) -> None:
        await self._ensure_model(config.translation_model, config.compute_mode)

    async def _ensure_model(self, model_name: str, compute_mode: str) -> None:
        async with self._model_lock:
            cuda_ok = torch is not None and torch.cuda.is_available()
            device = "cuda" if compute_mode == "cuda" and cuda_ok else "cpu"
            key = (model_name, device)
            if self._model is not None and self._loaded_key == key:
                return

            loop = asyncio.get_running_loop()

            def load_main() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model.to(device)
                model.eval()
                return tokenizer, model

            try:
                logger.info("Loading translation model=%s device=%s", model_name, device)
                tokenizer, model = await loop.run_in_executor(self._executor, load_main)
                self._tokenizer = tokenizer
                self._model = model
                self._loaded_key = key
                self._runtime_device = device
            except Exception as exc:
                logger.exception("Failed to load selected model, using Marian fallback: %s", exc)
                await self._ensure_fallback(device)

    async def _ensure_fallback(self, device: str) -> None:
        if self._fallback is not None and self._loaded_key == ("Helsinki-NLP/opus-mt-en-es", device):
            self._tokenizer, self._model = self._fallback
            return

        loop = asyncio.get_running_loop()

        def load_fallback() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
            name = "Helsinki-NLP/opus-mt-en-es"
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
            model.to(device)
            model.eval()
            return tokenizer, model

        tokenizer, model = await loop.run_in_executor(self._executor, load_fallback)
        self._fallback = (tokenizer, model)
        self._tokenizer = tokenizer
        self._model = model
        self._loaded_key = ("Helsinki-NLP/opus-mt-en-es", device)
        self._runtime_device = device

    def _translate_sync(self, text: str, source: str, target: str) -> str:
        assert self._tokenizer is not None and self._model is not None

        tokenizer = cast(PreTrainedTokenizerBase, self._tokenizer)
        model = cast(PreTrainedModel, self._model)
        device = self._runtime_device

        if "nllb" in ((self._loaded_key or ("", ""))[0]).lower():
            src_lang = NLLB_CODES.get(source, "eng_Latn")
            tgt_lang = NLLB_CODES.get(target, "spa_Latn")
            tokenizer.src_lang = src_lang
            encoded = cast(Any, tokenizer)(text, return_tensors="pt", truncation=True)
            if hasattr(encoded, "to"):
                encoded = encoded.to(device)
            bos_id_raw = tokenizer.convert_tokens_to_ids(tgt_lang)
            forced_bos_token_id = bos_id_raw if isinstance(bos_id_raw, int) else None
            inference_ctx = torch.inference_mode() if torch is not None else nullcontext()
            with inference_ctx:
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=128,
                    num_beams=1,
                    do_sample=False,
                )
            decoded = cast(list[str], tokenizer.batch_decode(generated, skip_special_tokens=True))
            return decoded[0] if decoded else ""

        encoded = cast(Any, tokenizer)(text, return_tensors="pt", truncation=True)
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        inference_ctx = torch.inference_mode() if torch is not None else nullcontext()
        with inference_ctx:
            generated = model.generate(
                **encoded,
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
            )
        decoded = cast(list[str], tokenizer.batch_decode(generated, skip_special_tokens=True))
        return decoded[0] if decoded else ""

    async def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
