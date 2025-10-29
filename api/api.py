"""
🫀 ECG Diagnosis API - FastAPI REST API
API for serving precomputed ECG cardiac analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from datetime import datetime
import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .ecg_api_helpers import initialize_data_service, data_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y | %I:%M%p",
    handlers=[logging.FileHandler("ecg_api.log"), logging.StreamHandler()],
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
    global data_loaded, startup_time
    startup_time = current_time_iso()
    logger.info("Starting ECG Classification API...")
    try:
        data_loaded = initialize_data_service()
        logger.info("Data service ready" if data_loaded else "Data service failed to load")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ECG Classification API...")

@app.get("/", summary="ECG Diagnosis API Overview", tags=["App Info"])
async def root():
    return {
        "app": "ECG Classification API",
        "purpose": "Serve precomputed cardiac ECG analysis results with clinical-grade precision.",
        "author": "Ridwan Oladipo, MD | AI Specialist",
        "version": "1.0.0",
        "documentation": "/docs",
    }

@app.get("/health", summary="Service Health Check", tags=["Health"])
async def health_check():
    return {
        "status": "ok" if data_loaded else "error",
        "data_loaded": data_loaded,
        "version": "1.0.0",
        "startup_time": startup_time,
        "timestamp": current_time_iso(),
    }

@app.get("/cases", summary="Get All Curated Cases", tags=["Cases"])
@limiter.limit("10/minute")
async def get_cases(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("Cases list requested")
        cases = data_service.get_demo_cases()
        return cases
    except Exception as e:
        logger.error(f"Error getting cases: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve cases")

@app.get("/case/{case_id}", summary="Get Case Details", tags=["Cases"])
@limiter.limit("10/minute")
async def get_case_details(request: Request, case_id: int):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Case details requested: {case_id}")
        case_data = data_service.get_case_details(case_id)
        if not case_data:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Case {case_id} not found")
        return case_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting case details: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve case details")

@app.get("/case/{case_id}/prediction", summary="Get Case Prediction", tags=["Cases"])
@limiter.limit("10/minute")
async def get_case_prediction(request: Request, case_id: int):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Prediction requested: {case_id}")
        prediction = data_service.get_case_prediction(case_id)
        if not prediction:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Prediction for case {case_id} not found")
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve prediction")

@app.get("/clinical-report/{case_id}", summary="Get Clinical Report", tags=["Clinical"])
@limiter.limit("10/minute")
async def get_clinical_report(request: Request, case_id: int):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Clinical report requested: {case_id}")
        report = data_service.get_clinical_report(case_id)
        if not report:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Clinical report for case {case_id} not found")
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting clinical report: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve clinical report")

@app.post("/generate-report/{case_id}", summary="Generate Clinical Report", tags=["Clinical"])
@limiter.limit("5/minute")
async def generate_clinical_report(request: Request, case_id: int):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Report generation requested: {case_id}")
        report = data_service.generate_clinical_report(case_id)
        if not report:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Cannot generate report for case {case_id}")
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating clinical report: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate clinical report")

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(status_code=422, content={"detail": f"Validation error: {exc}", "time": current_time_iso()})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Unexpected error occurred", "time": current_time_iso()})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
