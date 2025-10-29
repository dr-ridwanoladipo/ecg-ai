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

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                            Core Data Service Class
# ═══════════════════════════════════════════════════════════════════════════════
class ECGDataService:
    """ECG Classification Data Service - Serves precomputed results."""

    def __init__(self):
        """Initialize empty placeholders; populate via load_data()."""
        self.curated_cases = None
        self.model_card = None
        self.performance_data = None
        self.data_path = Path("evaluation_results")

    # ───────────────────────────────────────────────────────────────────────────
    # Data loading
    # ───────────────────────────────────────────────────────────────────────────
    def load_data(self) -> bool:
        """Load all data files."""
        try:
            logger.info("Loading ECG classification data...")

            # Load curated cases
            with open(self.data_path / 'curated_cases.json', 'r') as f:
                self.curated_cases = json.load(f)

            # Load model card
            with open(self.data_path / 'model_card.json', 'r') as f:
                self.model_card = json.load(f)

            # Load performance data
            with open(self.data_path / 'performance_data.json', 'r') as f:
                self.performance_data = json.load(f)

            logger.info("All data loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return False

    # ───────────────────────────────────────────────────────────────────────────
    # Demo cases
    # ───────────────────────────────────────────────────────────────────────────
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

    def get_case_prediction(self, case_id: int) -> Optional[Dict[str, Any]]:
        """Get prediction results for a specific case."""
        case = self.get_case_details(case_id)
        if not case:
            return None

        return {
            'case_id': case_id,
            'predicted_class': case['predicted_class'],
            'confidence': case['confidence'],
            'predictions': case['predictions'],
            'true_class': case['true_class']
        }

    # ───────────────────────────────────────────────────────────────────────────
    # Clinical reports
    # ───────────────────────────────────────────────────────────────────────────
    def get_clinical_report(self, case_id: int) -> Optional[Dict[str, Any]]:
        """Get clinical report for a specific case."""
        case = self.get_case_details(case_id)
        if not case:
            return None

        return {
            'case_id': case_id,
            'demographics': case['demographics'],
            'predicted_class': case['predicted_class'],
            'confidence': case['confidence'],
            'clinical_note': case['clinical_note'],
            'shap_importance': case.get('shap_importance', {})
        }

    def generate_clinical_report(self, case_id: int) -> Optional[Dict[str, Any]]:
        """Generate clinical report (returns precomputed result)."""
        return self.get_clinical_report(case_id)

    # ───────────────────────────────────────────────────────────────────────────
    # Performance metrics
    # ───────────────────────────────────────────────────────────────────────────
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get model performance summary."""
        if not self.model_card:
            return {}

        return {
            'model_name': self.model_card.get('model_info', {}).get('name', 'ECG ResNet-1D'),
            'version': self.model_card.get('model_info', {}).get('version', '1.0'),
            'architecture': self.model_card.get('model_info', {}).get('architecture', 'ResNet-1D + Dense'),
            'performance_metrics': self.model_card.get('performance', {}),
            'test_cases': len(self.curated_cases) if self.curated_cases else 0
        }

    def get_robustness_summary(self) -> Dict[str, Any]:
        """Get robustness analysis results."""
        if not self.performance_data or 'robustness' not in self.performance_data:
            return {'jitter': {}, 'scaling': {}}

        return self.performance_data['robustness']

    def get_calibration_data(self) -> Dict[str, Any]:
        """Get calibration analysis results."""
        if not self.performance_data or 'calibration' not in self.performance_data:
            return {}

        return self.performance_data['calibration']

    def get_roc_pr_data(self) -> Dict[str, Any]:
        """Get ROC and PR curve data."""
        if not self.performance_data:
            return {'roc_curves': {}, 'pr_curves': {}}

        return {
            'roc_curves': self.performance_data.get('roc_curves', {}),
            'pr_curves': self.performance_data.get('pr_curves', {})
        }

    def get_demographic_analysis(self) -> Dict[str, Any]:
        """Get demographic slice analysis."""
        if not self.performance_data or 'slice_analysis' not in self.performance_data:
            return {}

        return self.performance_data['slice_analysis']

    # ───────────────────────────────────────────────────────────────────────────
    # Media files
    # ───────────────────────────────────────────────────────────────────────────
    def get_case_images(self, case_id: int) -> Optional[Dict[str, Any]]:
        """Get image file paths for a case."""
        # Check if images exist for the case
        precolored_single = self.data_path / 'precolored_ecgs' / f'case_{case_id}_ecg_single_clean.png'
        precolored_12lead = self.data_path / 'precolored_ecgs' / f'case_{case_id}_ecg_12lead_clean.png'
        gradcam_single = self.data_path / 'curated_cases' / f'case_{case_id}_gradcam_single.png'
        gradcam_12lead = self.data_path / 'curated_cases' / f'case_{case_id}_gradcam_12lead.png'
        shap_image = self.data_path / 'curated_cases' / f'case_{case_id}_shap.png'

        images = {'case_id': case_id}

        if precolored_single.exists():
            images['ecg_single_clean'] = f'case_{case_id}_ecg_single_clean.png'

        if precolored_12lead.exists():
            images['ecg_12lead_clean'] = f'case_{case_id}_ecg_12lead_clean.png'

        if gradcam_single.exists():
            images['gradcam_single'] = f'case_{case_id}_gradcam_single.png'

        if gradcam_12lead.exists():
            images['gradcam_12lead'] = f'case_{case_id}_gradcam_12lead.png'

        if shap_image.exists():
            images['shap'] = f'case_{case_id}_shap.png'

        if len(images) == 1:
            return None

        images['message'] = f"Image files available for case {case_id}"
        return images

    # ───────────────────────────────────────────────────────────────────────────
    # Data validation
    # ───────────────────────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
#                        Global instance + Convenience functions
# ═══════════════════════════════════════════════════════════════════════════════
data_service = ECGDataService()


def initialize_data_service() -> bool:
    """Initialize the global data service instance."""
    return data_service.load_data()


def get_data_service() -> ECGDataService:
    """Get the global data service instance."""
    return data_service


def load_demo_data():
    """Load demo data (backwards compatibility)."""
    if not data_service.curated_cases:
        data_service.load_data()
    return (
        data_service.curated_cases,
        data_service.model_card,
        data_service.performance_data
    )


def get_case_data(case_id: int):
    """Get case data (backwards compatibility)."""
    return data_service.get_case_details(case_id)


def get_clinical_report_data(case_id: int):
    """Get clinical report (backwards compatibility)."""
    return data_service.get_clinical_report(case_id)