"""
🫀 ECG Diagnosis API Package
FastAPI backend for serving precomputed cardiac analysis.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from .api import app
from .ecg_api_helpers import data_service, initialize_data_service, get_data_service

__all__ = ["app", "data_service", "initialize_data_service", "get_data_service"]
