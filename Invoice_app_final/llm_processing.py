import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict

from llm_engine import get_engine, get_engine_info
from prompts import kv_prompt

@dataclass
class LLMResult:
    fields: Dict[str, Any]
    model_info: Dict[str, Any]

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}", re.MULTILINE)

def _extract_first_json_block(text: str) -> str:
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError("No JSON object found in model output")
    return m.group(0)

def _json_safe_parse(blob: str) -> dict:
    try:
        return json.loads(blob)
    except Exception:
        pass
    blob2 = blob.replace("\n", " ").replace("\t", " ")
    blob2 = re.sub(r",\s*}", "}", blob2)
    blob2 = re.sub(r",\s*]", "]", blob2)
    return json.loads(blob2)

async def extract_structured_json(
    ocr_markdown: str,
    timeout_s: int = 90,
) -> LLMResult:
    engine = get_engine()
    prompt = kv_prompt.format(doc_body=ocr_markdown or "")

    async def _run():
        return await engine.generate(prompt)

    model_out = await asyncio.wait_for(_run(), timeout=timeout_s)
    json_blob = _extract_first_json_block(model_out)
    fields = _json_safe_parse(json_blob)

    return LLMResult(fields=fields, model_info=get_engine_info())
