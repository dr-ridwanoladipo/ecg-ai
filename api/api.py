"""
🫀 ECG Diagnosis API - FastAPI REST API
API for serving precomputed ECG cardiac analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from fastapi import FastAPI

app = FastAPI(
    title="ECG Classification API",
    description="API for serving precomputed cardiac ECG analysis results",
    version="1.0.0",
)
