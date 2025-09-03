"""
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
        self.max_new_tokens = 2048

    async def generate(self, prompt: str) -> str:

        import torch

        def _run():
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,     
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                input_ids = inputs
                attention_mask = None
            else:
                # assume dict-like
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", None)

            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            gen_kwargs = dict(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask

            with torch.no_grad():
                out = self.model.generate(**gen_kwargs)

            text = self.tokenizer.decode(out[0], skip_special_tokens=True)

            idx = text.find("{")
            if idx > 0:
                text = text[idx:]
            return text

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
"""
# llm_engine.py — AWQ (4-bit) backend for Qwen3-8B on ARM (no Triton needed)

import os
import asyncio
import warnings
from typing import Dict, Any

# Make 100% sure nothing tries to pull Triton / xFormers paths.
os.environ.setdefault("PYTORCH_DISABLE_TRITON", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_XFORMERS", "1")
os.environ.setdefault("UNSLOTH_DISABLE_XFORMERS", "1")

import torch
from transformers import AutoTokenizer

try:
    # Package name is "autoawq" (pip), import path is "awq"
    from awq import AutoAWQForCausalLM
except Exception as e:
    raise RuntimeError(
        "AutoAWQ is not installed. Run: pip install autoawq autoawq-kernels"
    ) from e


def _maybe_bool(env: str, default: bool) -> bool:
    v = os.getenv(env)
    if v is None:
        return default
    return v.lower() in ("1", "true", "t", "yes", "y")


def _apply_qwen_chat_template(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_ctx: int,
    disable_thinking: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Wraps the plain prompt into Qwen-style chat messages and returns tokenized tensors.
    We try to pass enable_thinking=False when the tokenizer supports it.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_ctx,
        return_tensors="pt",
    )

    # Some Qwen tokenizers accept enable_thinking; others don't — guard it.
    if disable_thinking:
        try:
            kwargs["enable_thinking"] = False  # Qwen3 supports this; safe to ignore if not
        except Exception:
            pass

    try:
        input_ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Older tokenizers without the arg — retry without it
        kwargs.pop("enable_thinking", None)
        input_ids = tokenizer.apply_chat_template(messages, **kwargs)
    return {"input_ids": input_ids}


class AWQEngine:
    """
    Minimal, robust AWQ engine:
      - Loads an *already* AWQ-quantized Qwen3-8B (recommended), e.g. a repo/path with awq_config.json.
      - Or, if you point to a base FP16 model, you should quantize offline once and then load the quantized dir.
    """

    def __init__(self) -> None:
        # ---- Config from env ----
        self.model_id = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
        self.device_map = os.getenv("LLM_DEVICE_MAP", "cuda:0")  # pin to GPU1 via "cuda:1" in .env
        self.max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))
        self.max_ctx = int(os.getenv("MAX_MODEL_LEN", "8192"))
        self.reserve = int(os.getenv("LLM_GEN_RESERVE_TOKENS", "384"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("LLM_TOP_P", "1.0"))
        self.do_sample = _maybe_bool("LLM_DO_SAMPLE", False)

        # ---- Tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            trust_remote_code=True,
        )

        # Pad token handling to avoid "pad token == eos" warnings
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if getattr(self.tokenizer, "chat_template", None) is None:
            warnings.warn(
                "Tokenizer has no chat_template; generation may be less controlled."
            )

        # ---- Model (AWQ-quantized) ----
        # Expecting LLM_MODEL to point to an AWQ-quantized repo/folder.
        # If you pass a base model here, AWQ will try to load quantized weights if present.
        self.model = AutoAWQForCausalLM.from_quantized(
            self.model_id,
            trust_remote_code=True,
            fuse_layers=True,
            safetensors=True,
            device_map=self.device_map,
            # You can set max_seq_len to trim KV cache usage on smaller VRAM if needed:
            # max_seq_len=self.max_ctx,
        ).eval()

        # Ensure eos id known
        self.eos_id = self.tokenizer.eos_token_id

    async def generate(self, prompt: str) -> str:
        """
        Deterministic (default do_sample=False) generation of the JSON string.
        Keeps a context headroom so we don't overflow the model's window.
        """
        def _run() -> str:
            # Keep room for output tokens
            max_len = max(1024, self.max_ctx - self.reserve)

            toks = _apply_qwen_chat_template(
                self.tokenizer,
                prompt_text=prompt,
                max_ctx=max_len,
                disable_thinking=True,
            )

            input_ids = toks["input_ids"]
            # Always pass attention_mask to avoid pad/eos issues
            attention_mask = torch.ones_like(input_ids)

            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            gen_kwargs: Dict[str, Any] = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.eos_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            with torch.no_grad():
                out = self.model.generate(**gen_kwargs)

            text = self.tokenizer.decode(out[0], skip_special_tokens=True)

            # Return only from the first JSON brace if present
            i = text.find("{")
            return text[i:] if i >= 0 else text

        return await asyncio.to_thread(_run)


# ---- Module-level singletons to match your app's current imports ----
_engine: AWQEngine | None = None

def get_engine() -> AWQEngine:
    global _engine
    if _engine is None:
        _engine = AWQEngine()
    return _engine

def get_engine_info() -> Dict[str, Any]:
    return {
        "backend": "awq",
        "model": os.getenv("LLM_MODEL", "unknown"),
        "device_map": os.getenv("LLM_DEVICE_MAP", "cuda:0"),
        "max_model_len": os.getenv("MAX_MODEL_LEN", "8192"),
        "max_new_tokens": os.getenv("LLM_MAX_NEW_TOKENS", "1024"),
    }

