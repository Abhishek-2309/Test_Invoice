import asyncio
import logging
from dataclasses import dataclass
from typing import Dict

import torch
from unsloth import FastModel

log = logging.getLogger("uvicorn.error")

@dataclass
class WarmupConfig:
    model: str = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    max_model_len: int = 8192
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    device_map: str = "auto"  # "auto" or explicit mapping

# Global singleton
_engine_lock = asyncio.Lock()
_engine = None
_engine_info: Dict[str, str] = {}

class UnslothEngine:
    def __init__(self, cfg: WarmupConfig):
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True

        log.info(f"[Unsloth] Loading extractor model: {cfg.model}")
        # EXACTLY like your snippet
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=cfg.model,
            max_seq_length=cfg.max_model_len,
            load_in_4bit=cfg.load_in_4bit,
            load_in_8bit=cfg.load_in_8bit,
            device_map=cfg.device_map,
        )
        self.model.eval()
        self.model_name = cfg.model
        self.max_new_tokens = 1024

    async def generate(self, prompt: str) -> str:
        def _run():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # If the model echoes the prompt, trim it
            return text[len(prompt):] if text.startswith(prompt) else text

        return await asyncio.to_thread(_run)

async def load_global_engine(cfg: WarmupConfig):
    global _engine, _engine_info
    async with _engine_lock:
        if _engine is not None:
            return
        _engine = UnslothEngine(cfg)
        _engine_info = {
            "backend": "unsloth",
            "model": _engine.model_name,
            "max_model_len": str(cfg.max_model_len),
            "load_in_4bit": str(cfg.load_in_4bit),
            "load_in_8bit": str(cfg.load_in_8bit),
            "device_map": cfg.device_map,
        }
        log.info(f"[Unsloth] Engine ready: {_engine_info}")

async def unload_global_engine():
    global _engine
    async with _engine_lock:
        _engine = None

def get_engine():
    if _engine is None:
        raise RuntimeError("Engine not initialized. Did you miss app startup?")
    return _engine

def get_engine_info():
    return dict(_engine_info)
