"""
🫀 ECG Diagnosis API - FastAPI REST API
API for serving precomputed ECG cardiac analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import json
import logging
import warnings
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

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
        """Load all data files."""
        try:
            logger.info("Loading ECG classification data...")

            with open(self.data_path / 'curated_cases.json', 'r') as f:
                self.curated_cases = json.load(f)

            with open(self.data_path / 'model_card.json', 'r') as f:
                self.model_card = json.load(f)

            with open(self.data_path / 'performance_data.json', 'r') as f:
                self.performance_data = json.load(f)

            logger.info("All data loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return False

    def get_demo_cases(self) -> List[Dict[str, Any]]:
        """Return list of all curated cases."""
        if not self.curated_cases:
            return []
        return self.curated_cases

    def get_case_details(self, case_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific case."""
        if not self.curated_cases:
            return None
        for case in self.curated_cases:
            if case['case_id'] == case_id:
                return case
        return None

    def validate_data(self) -> Dict[str, bool]:
        """Validate that all required data is loaded."""
        return {
            'curated_cases_loaded': self.curated_cases is not None,
            'model_card_loaded': self.model_card is not None,
            'performance_data_loaded': self.performance_data is not None
        }

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        if not self.curated_cases:
            return {'error': 'Data not loaded'}
        return {
            'total_cases': len(self.curated_cases),
            'curated_cases': len(self.curated_cases),
            'model_info_available': bool(self.model_card),
            'performance_data_available': bool(self.performance_data),
            'classes': ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        }

data_service = ECGDataService()

def initialize_data_service() -> bool:
    """Initialize the global data service instance."""
    return data_service.load_data()

def get_data_service() -> ECGDataService:
    """Get the global data service instance."""
    return data_service
