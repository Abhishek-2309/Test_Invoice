import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_engine import get_engine, get_engine_info
from prompts import kv_prompt

@dataclass
class LLMResult:
    fields: Dict[str, Any]
    model_info: Dict[str, Any]

# Remove hidden reasoning if any (some models add it)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

# code fences (we will consider **all** matches, newest first)
_CODEFENCE_TILDE = re.compile(r"~~~\s*json\s*([\s\S]*?)\s*~~~", re.IGNORECASE)
_CODEFENCE_TICK  = re.compile(r"```+\s*json\s*([\s\S]*?)\s*```+", re.IGNORECASE)

# raw JSON (fallback) — consider all blocks, newest first
_JSON_ANY = re.compile(r"\{[\s\S]*?\}", re.MULTILINE)

def _clean_model_text(text: str) -> str:
    # strip optional <think>...</think> blocks
    return _THINK_RE.sub("", text)

def _json_safe_parse(blob: str) -> dict:
    # strict first
    try:
        return json.loads(blob)
    except Exception:
        pass
    # light repairs (commas before } or ])
    blob2 = blob.replace("\u00A0", " ").replace("\t", " ")
    blob2 = re.sub(r",\s*}", "}", blob2)
    blob2 = re.sub(r",\s*]", "]", blob2)
    return json.loads(blob2)

def _pick_last_valid_json(text: str) -> str:
    """
    Strategy:
    1) Try ALL fenced json blocks (~~~json / ```json), newest-last → newest-first.
       Return the first that parses.
    2) Else, try ALL raw {...} blocks, newest-first; return first that parses.
    3) Else, raise.
    """
    cleaned = _clean_model_text(text)

    # 1) all fenced (tilde + tick)
    candidates: List[str] = []
    candidates += _CODEFENCE_TILDE.findall(cleaned)
    candidates += _CODEFENCE_TICK.findall(cleaned)

    # newest-first (take from the end)
    for blob in reversed(candidates):
        jb = blob.strip()
        if not (jb.startswith("{") and jb.endswith("}")):
            continue
        try:
            _ = _json_safe_parse(jb)
            return jb
        except Exception:
            continue

    # 2) raw {...} blocks (may include instruction examples; pick last one that parses)
    raw_blocks = _JSON_ANY.findall(cleaned)
    for jb in reversed(raw_blocks):
        jb = jb.strip()
        try:
            _ = _json_safe_parse(jb)
            return jb
        except Exception:
            continue

    raise ValueError("No valid JSON object found in model output")

async def _one_shot(prompt: str, timeout_s: int) -> str:
    engine = get_engine()
    async def _run():
        return await engine.generate(prompt)
    return await asyncio.wait_for(_run(), timeout=timeout_s)

async def extract_structured_json(
    ocr_markdown: str,
    timeout_s: int = 90,
) -> LLMResult:
    base_prompt = kv_prompt.format(doc_body=ocr_markdown or "")

    # First pass
    model_out = await _one_shot(base_prompt, timeout_s)
    try:
        json_blob = _pick_last_valid_json(model_out)
        fields = _json_safe_parse(json_blob)
        return LLMResult(fields=fields, model_info=get_engine_info())
    except Exception:
        pass

    # Retry once with strict tail (keeps your kv_prompt intact)
    strict_suffix = "\nReturn ONLY a valid JSON object, with no extra text."
    model_out2 = await _one_shot(base_prompt + strict_suffix, timeout_s)
    json_blob = _pick_last_valid_json(model_out2)
    fields = _json_safe_parse(json_blob)
    return LLMResult(fields=fields, model_info=get_engine_info())
