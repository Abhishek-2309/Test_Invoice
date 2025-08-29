"""
Optional helper for CLI-like folder processing (not used by FastAPI directly).
"""
import asyncio
from pathlib import Path
from typing import List
from routes import _process_one

async def process_folder(path: str):
    p = Path(path)
    tasks = []
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}:
            with open(f, "rb") as fh:
                content = fh.read()
            tasks.append(
                _process_one(
                    filename=f.name,
                    content=content,
                    return_ocr=False,
                    ocr_timeout_s=180,
                    llm_timeout_s=90,
                )
            )
    return await asyncio.gather(*tasks, return_exceptions=True)
