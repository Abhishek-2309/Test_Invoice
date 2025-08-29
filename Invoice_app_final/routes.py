import asyncio
import logging
import time
from typing import List, Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ocr import ocr_file_to_markdown
from llm_processing import extract_structured_json
from schemas import ExtractionResponse, BatchExtractionResponse

router = APIRouter()
log = logging.getLogger("uvicorn.error")

# Global concurrency cap for heavy work
_CONCURRENCY_LIMIT = int(__import__("os").getenv("UVICORN_CONCURRENCY", "4"))
_sema = asyncio.Semaphore(_CONCURRENCY_LIMIT)

class Health(BaseModel):
    status: str = "ok"

@router.get("/healthz", response_model=Health)
async def healthz():
    return Health()

async def _process_one(
    filename: str,
    content: bytes,
    return_ocr: bool,
    ocr_timeout_s: int,
    llm_timeout_s: int,
) -> Dict[str, Any]:
    started = time.time()
    async with _sema:
        try:
            # OCR -> markdown with <table> tags preserved
            ocr_markdown = await ocr_file_to_markdown(
                filename=filename,
                file_bytes=content,
                timeout_s=ocr_timeout_s,
            )
        except Exception as e:
            log.exception("OCR failed")
            raise HTTPException(status_code=400, detail=f"OCR failed: {e}")

        try:
            # Pass OCR markdown AS-IS to kv_prompt
            result = await extract_structured_json(
                ocr_markdown=ocr_markdown,
                timeout_s=llm_timeout_s,
            )
        except Exception as e:
            log.exception("LLM extraction failed")
            raise HTTPException(status_code=500, detail=f"LLM extraction failed: {e}")

    elapsed = round(time.time() - started, 3)
    payload: Dict[str, Any] = {
        "filename": filename,
        "elapsed_seconds": elapsed,
        "fields": result.fields,
        "model_info": result.model_info,
    }
    if return_ocr:
        payload["ocr_markdown"] = ocr_markdown
    return payload

@router.post("/extract", response_model=ExtractionResponse)
async def extract(
    file: UploadFile = File(...),
    return_ocr: bool = Query(False, description="Include raw OCR markdown"),
    ocr_timeout_s: int = Query(120),
    llm_timeout_s: int = Query(90),
):
    content = await file.read()
    filename = file.filename or "uploaded"
    payload = await _process_one(
        filename, content, return_ocr, ocr_timeout_s, llm_timeout_s
    )
    return JSONResponse(content=payload)

@router.post("/extract_batch", response_model=BatchExtractionResponse)
async def extract_batch(
    files: List[UploadFile] = File(...),
    return_ocr: bool = Query(False),
    ocr_timeout_s: int = Query(120),
    llm_timeout_s: int = Query(90),
):
    """Process multiple files concurrently under the global semaphore."""
    async def _runner(f: UploadFile):
        content = await f.read()
        fname = f.filename or "uploaded"
        return await _process_one(
            fname, content, return_ocr, ocr_timeout_s, llm_timeout_s
        )

    tasks = [_runner(f) for f in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_raw: List[str] = []
    errors = 0
    for res in results:
        if isinstance(res, Exception):
            errors += 1
            final_raw.append(f'{{"error": "{str(res)}"}}')
        else:
            final_raw.append(__import__("json").dumps(res))

    return BatchExtractionResponse(total=len(files), errors=errors, results_raw=final_raw)
