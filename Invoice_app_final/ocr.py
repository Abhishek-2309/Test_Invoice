import asyncio
import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF
from PIL import Image
import torch

from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
)

log = logging.getLogger("uvicorn.error")

@dataclass
class OCRResult:
    markdown: str

# ---- Config ----
_OCR_MODEL_ID = os.getenv("OCR_MODEL_ID", "nanonets/Nanonets-OCR-s")
_OCR_DTYPE = os.getenv("OCR_DTYPE", "auto")             # "auto" | "float16" | "bfloat16"
_OCR_MAX_NEW_TOKENS = int(os.getenv("OCR_MAX_NEW_TOKENS", "4000"))
_OCR_DPI = int(os.getenv("OCR_DPI", "180"))
_OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "1000"))

# EXACT prompt (your text, verbatim)
_PROMPT2 = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

def strip_prompt_from_output(text: str) -> str:
    split_pattern = r"(?:^|\n)assistant\s*\n"
    parts = re.split(split_pattern, text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()

# ---- Singleton engine ----
_engine_lock = asyncio.Lock()
_engine_loaded = False
_ocr_model = None
_ocr_tokenizer = None
_ocr_processor = None

def _dtype_from_str(name: str):
    name = (name or "auto").lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return "auto"

async def _lazy_load_engine():
    global _engine_loaded, _ocr_model, _ocr_tokenizer, _ocr_processor
    async with _engine_lock:
        if _engine_loaded:
            return

        log.info(f"[OCR] Loading '{_OCR_MODEL_ID}' (dtype={_OCR_DTYPE})")
        _ocr_model = AutoModelForImageTextToText.from_pretrained(
            _OCR_MODEL_ID, torch_dtype=_dtype_from_str(_OCR_DTYPE), device_map="auto"
        ).eval()
        try:
            _ocr_model = torch.compile(_ocr_model)
            log.info("[OCR] torch.compile enabled")
        except Exception:
            log.info("[OCR] torch.compile not enabled")

        _ocr_tokenizer = AutoTokenizer.from_pretrained(_OCR_MODEL_ID)
        _ocr_processor = AutoProcessor.from_pretrained(_OCR_MODEL_ID)

        _engine_loaded = True
        log.info("[OCR] Engine ready.")

def _is_pdf(filename: str, blob: bytes) -> bool:
    if filename.lower().endswith(".pdf"):
        return True
    return blob[:4] == b"%PDF"

def _pdf_to_temp_images(blob: bytes, dpi: int) -> List[str]:
    """
    Render PDF pages to temp PNGs and return file paths (so we can use file:// in chat msgs).
    """
    paths: List[str] = []
    with fitz.open(stream=blob, filetype="pdf") as doc:
        pages = min(len(doc), _OCR_MAX_PAGES)
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            fd, path = tempfile.mkstemp(suffix=".png", prefix=f"ocr_page_{i}_")
            os.close(fd)
            img.save(path)
            paths.append(path)
    return paths

def _bytes_to_temp_image(blob: bytes) -> str:
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    fd, path = tempfile.mkstemp(suffix=".png", prefix="ocr_img_")
    os.close(fd)
    img.save(path)
    return path

def _ocr_page_with_nanonets(image_path: str, max_new_tokens: int) -> str:
    """
    EXACTLY your working function, just non-async. Called via to_thread.
    """
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": _PROMPT2}
        ]}
    ]
    text = _ocr_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _ocr_processor(text=[text], images=[image], return_tensors="pt", padding=False).to(_ocr_model.device)
    with torch.no_grad():
        outputs = _ocr_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    markdown = _ocr_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(markdown)
    # free temp tensors
    del inputs, outputs
    return strip_prompt_from_output(markdown)

async def _ocr_page_with_nanonets_async(image_path: str, max_new_tokens: int) -> str:
    return await asyncio.to_thread(_ocr_page_with_nanonets, image_path, max_new_tokens)

async def ocr_file_to_markdown(
    filename: str,
    file_bytes: bytes,
    timeout_s: int = 180,
) -> str:
    """
    OCR that returns a single markdown string (with <table> tags, etc.) by concatenating page outputs.
    Uses your exact nanonets-ocr-s prompt and invocation semantics.
    """
    await _lazy_load_engine()

    if _is_pdf(filename, file_bytes):
        page_paths = _pdf_to_temp_images(file_bytes, dpi=_OCR_DPI)
        if not page_paths:
            return ""
        outputs: List[str] = []
        for p in page_paths:
            out = await asyncio.wait_for(_ocr_page_with_nanonets_async(p, _OCR_MAX_NEW_TOKENS), timeout=timeout_s)
            outputs.append(out)
        # optional: cleanup temp files
        for p in page_paths:
            try: os.remove(p)
            except Exception: pass
        return "\n\n".join(outputs).strip()

    # Single image
    pth = _bytes_to_temp_image(file_bytes)
    out = await asyncio.wait_for(_ocr_page_with_nanonets_async(pth, _OCR_MAX_NEW_TOKENS), timeout=timeout_s)
    try: os.remove(pth)
    except Exception: pass
    return out.strip()
