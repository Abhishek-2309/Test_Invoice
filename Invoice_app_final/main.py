import logging
import os
from fastapi import FastAPI
from routes import router
from llm_engine import WarmupConfig, load_global_engine, unload_global_engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("uvicorn.error")

app = FastAPI(title="Invoice Structuring API", version="3.0.0")
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """
    Warm up the extractor LLM (Qwen3-8B via Unsloth FastModel) once.
    OCR (nanonets-ocr-s) is lazy-loaded on first request.
    """
    cfg = WarmupConfig(
        model=os.getenv("LLM_MODEL", "unsloth/Qwen3-8B-unsloth-bnb-4bit"),
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "8192")),
        load_in_4bit=os.getenv("LOAD_IN_4BIT", "1") == "1",
        load_in_8bit=os.getenv("LOAD_IN_8BIT", "0") == "1",
        device_map=os.getenv("LLM_DEVICE_MAP", "auto"),
    )
    await load_global_engine(cfg)
    log.info("Extractor LLM warmed up.")

@app.on_event("shutdown")
async def shutdown_event():
    await unload_global_engine()
    log.info("Shutdown complete.")
