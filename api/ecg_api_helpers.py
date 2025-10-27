"""
🫀 ECG Diagnosis API - FastAPI REST API
API for serving precomputed ECG cardiac analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import json
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECGDataService:
    """ECG Classification Data Service - Serves precomputed results."""

    def __init__(self):
        """Initialize empty placeholders; populate via load_data()."""
        self.curated_cases = None
        self.model_card = None
        self.performance_data = None
        self.data_path = Path("evaluation_results")

    def load_data(self) -> bool:
        """Placeholder: load all required data files."""
        logger.info("Loading ECG classification data...")
        return True

data_service = ECGDataService()

def initialize_data_service() -> bool:
    """Initialize the global data service instance."""
    return data_service.load_data()

def get_data_service() -> ECGDataService:
    """Get the global data service instance."""
    return data_service
