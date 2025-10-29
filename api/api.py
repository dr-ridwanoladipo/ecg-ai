"""
🫀 ECG Diagnosis API - FastAPI REST API
API for serving precomputed ECG cardiac analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from datetime import datetime
import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .ecg_api_helpers import initialize_data_service, data_service


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y | %I:%M%p",
    handlers=[logging.FileHandler("ecg_api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Rate Limiting
# ------------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ------------------------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# State / Utilities
# ------------------------------------------------------------------------------
data_loaded = False
startup_time = None


def current_time_iso():
    return datetime.now().isoformat()


# ------------------------------------------------------------------------------
# Response Models
# ------------------------------------------------------------------------------
class CaseInfo(BaseModel):
    case_id: int
    description: str
    true_class: str
    predicted_class: str
    confidence: float


class Demographics(BaseModel):
    age: float
    sex: str
    heart_rate: Optional[int]
    rhythm: Optional[str]
    height: float
    weight: float


class CasePrediction(BaseModel):
    case_id: int
    predicted_class: str
    confidence: float
    predictions: Dict[str, float]
    true_class: str


class ClinicalReport(BaseModel):
    case_id: int
    demographics: Demographics
    predicted_class: str
    confidence: float
    clinical_note: str
    shap_importance: Dict[str, float]


class MIMetrics(BaseModel):
    sensitivity: float
    specificity: float
    ppv: float
    npv: float


class PerformanceMetrics(BaseModel):
    test_accuracy: float
    macro_f1: float
    class_f1_scores: Dict[str, float]
    mi_clinical_metrics: MIMetrics


class MetricsSummary(BaseModel):
    model_name: str
    version: str
    architecture: str
    performance_metrics: PerformanceMetrics
    test_cases: int
    timestamp: str


class RobustnessSummary(BaseModel):
    jitter_levels: List[float]
    jitter_performance: List[float]
    scale_factors: List[float]
    scale_performance: List[float]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    data_loaded: bool
    startup_time: str
    timestamp: str
    version: str


class CaseImages(BaseModel):
    case_id: int
    ecg_single_clean: Optional[str]
    ecg_12lead_clean: Optional[str]
    gradcam_single: Optional[str]
    gradcam_12lead: Optional[str]
    shap: Optional[str]
    message: str


# ------------------------------------------------------------------------------
# Startup / Shutdown
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/", summary="ECG Diagnosis API Overview", tags=["App Info"])
async def root():
    return {
        "app": "ECG Classification API",
        "purpose": "Serve precomputed cardiac ECG analysis results with clinical-grade precision.",
        "model": {
            "type": "ResNet-1D + Dense Multimodal Network",
            "performance": {
                "MI_sensitivity": "96.2%",
                "MI_specificity": "100.0%",
                "overall_accuracy": "87.4%",
            },
            "training_data": "PTB-XL Dataset (22,000 ECG records, 19,000 patients)",
        },
        "author": "Ridwan Oladipo, MD | AI Specialist",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, summary="Service Health Check", tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok" if data_loaded else "error",
        data_loaded=data_loaded,
        version="1.0.0",
        startup_time=startup_time,
        timestamp=current_time_iso(),
    )


@app.get("/cases", response_model=List[CaseInfo], summary="Get All Curated Cases", tags=["Cases"])
@limiter.limit("10/minute")
async def get_cases(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("Cases list requested")
        cases = data_service.get_demo_cases()
        return [CaseInfo(**{k: v for k, v in case.items() if k in CaseInfo.__fields__}) for case in cases]
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


@app.get("/case/{case_id}/prediction", response_model=CasePrediction, summary="Get Case Prediction", tags=["Cases"])
@limiter.limit("10/minute")
async def get_case_prediction(request: Request, case_id: int):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Prediction requested: {case_id}")
        prediction = data_service.get_case_prediction(case_id)
        if not prediction:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Prediction for case {case_id} not found")
        return CasePrediction(**prediction)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve prediction")


@app.get("/clinical-report/{case_id}", response_model=ClinicalReport, summary="Get Clinical Report", tags=["Clinical"])
@limiter.limit("10/minute")
async def get_clinical_report(request: Request, case_id: int):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Clinical report requested: {case_id}")
        report = data_service.get_clinical_report(case_id)
        if not report:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Clinical report for case {case_id} not found")
        return ClinicalReport(
            demographics=Demographics(**report["demographics"]),
            **{k: v for k, v in report.items() if k != "demographics"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting clinical report: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve clinical report")


@app.post("/generate-report/{case_id}", response_model=ClinicalReport, summary="Generate Clinical Report", tags=["Clinical"])
@limiter.limit("5/minute")
async def generate_clinical_report(request: Request, case_id: int):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Report generation requested: {case_id}")
        report = data_service.generate_clinical_report(case_id)
        if not report:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Cannot generate report for case {case_id}")
        return ClinicalReport(
            demographics=Demographics(**report["demographics"]),
            **{k: v for k, v in report.items() if k != "demographics"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating clinical report: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate clinical report")


@app.get("/metrics-summary", response_model=MetricsSummary, summary="Get Model Performance Summary", tags=["Performance"])
@limiter.limit("10/minute")
async def get_metrics_summary(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("Metrics summary requested")
        metrics = data_service.get_metrics_summary()
        performance = metrics["performance_metrics"]
        return MetricsSummary(
            model_name=metrics["model_name"],
            version=metrics["version"],
            architecture=metrics["architecture"],
            performance_metrics=PerformanceMetrics(
                test_accuracy=performance["test_accuracy"],
                macro_f1=performance["macro_f1"],
                class_f1_scores=performance["class_f1_scores"],
                mi_clinical_metrics=MIMetrics(**performance["mi_clinical_metrics"]),
            ),
            test_cases=metrics["test_cases"],
            timestamp=current_time_iso(),
        )
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve metrics summary")


@app.get("/robustness-summary", response_model=RobustnessSummary, summary="Get Robustness Analysis", tags=["Performance"])
@limiter.limit("10/minute")
async def get_robustness_summary(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("Robustness summary requested")
        robustness = data_service.get_robustness_summary()
        return RobustnessSummary(
            jitter_levels=robustness.get("jitter_levels", []),
            jitter_performance=robustness.get("jitter_performance", []),
            scale_factors=robustness.get("scale_factors", []),
            scale_performance=robustness.get("scale_performance", []),
            timestamp=current_time_iso(),
        )
    except Exception as e:
        logger.error(f"Error getting robustness summary: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve robustness summary")


@app.get("/calibration", summary="Get Calibration Data", tags=["Performance"])
@limiter.limit("10/minute")
async def get_calibration_data(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("Calibration data requested")
        calibration = data_service.get_calibration_data()
        if not calibration:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Calibration data not found")
        return calibration
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting calibration data: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve calibration data")


@app.get("/roc-pr-curves", summary="Get ROC and PR Curve Data", tags=["Performance"])
@limiter.limit("10/minute")
async def get_roc_pr_curves(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("ROC/PR curves requested")
        curves = data_service.get_roc_pr_data()
        return curves
    except Exception as e:
        logger.error(f"Error getting ROC/PR curves: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve curve data")


@app.get("/demographic-analysis", summary="Get Demographic Performance Analysis", tags=["Performance"])
@limiter.limit("10/minute")
async def get_demographic_analysis(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("Demographic analysis requested")
        demographics = data_service.get_demographic_analysis()
        if not demographics:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Demographic analysis not found")
        return demographics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting demographic analysis: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve demographic analysis")


@app.get("/case/{case_id}/images", response_model=CaseImages, summary="Get Image File Paths", tags=["Media"])
@limiter.limit("10/minute")
async def get_case_images(request: Request, case_id: int):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Image paths requested: {case_id}")
        images = data_service.get_case_images(case_id)
        if not images:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Images for case {case_id} not found")
        return CaseImages(**images)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting images: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve image paths")


# ------------------------------------------------------------------------------
# Exception Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(status_code=422, content={"detail": f"Validation error: {exc}", "time": current_time_iso()})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Unexpected error occurred", "time": current_time_iso()})


# ------------------------------------------------------------------------------
# Uvicorn Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
