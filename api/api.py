"""
🫀 ECG Diagnosis API - FastAPI REST API
API for serving precomputed ECG cardiac analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from datetime import datetime
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y | %I:%M%p",
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="ECG Classification API",
    description="API for serving precomputed cardiac ECG analysis results",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} in {(time.time() - start) * 1000:.2f} ms")
    return response

data_loaded = False
startup_time = None

def current_time_iso():
    return datetime.now().isoformat()

@app.on_event("startup")
async def startup_event():
    global startup_time
    startup_time = current_time_iso()
    logger.info("Starting ECG Classification API...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ECG Classification API...")

@app.get("/", summary="ECG Diagnosis API Overview")
async def root():
    return {
        "app": "ECG Classification API",
        "purpose": "Serve precomputed cardiac ECG analysis results.",
        "author": "Ridwan Oladipo, MD | AI Specialist",
        "version": "1.0.0",
        "documentation": "/docs",
    }
