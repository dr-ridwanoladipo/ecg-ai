"""
🫀 ECG Diagnosis API - FastAPI REST API
API for serving precomputed ECG cardiac analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from fastapi import FastAPI, Request
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y | %I:%M%p",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ECG Classification API",
    description="API for serving precomputed cardiac ECG analysis results",
    version="1.0.0",
)

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} completed in {process_time:.2f} ms")
    return response

@app.get("/", summary="ECG Diagnosis API Overview")
async def root():
    return {"message": "ECG Diagnosis API is running successfully."}
