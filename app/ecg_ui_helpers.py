"""
🫀 ECG Classification AI - Helper Functions
Medical AI interface for cardiac arrhythmia analysis.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
import base64


def load_custom_css():
    """Load custom CSS for professional cardiac interface"""
    st.markdown("""
    <style>
    /* Import medical-grade fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide Streamlit's default chrome */
    #MainMenu, footer, header, .stDeployButton {visibility: hidden;}

    /* Reduce top/bottom padding of main container */
    div.block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        margin-top: 0rem !important;
        margin-bottom: 7rem !important;
    }

    /* Medical header styling */
    .cardiac-header {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .cardiac-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .cardiac-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Status indicators */
    .success-indicator {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 0.5rem 0;
    }

    /* Case selection */
    .case-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .case-card:hover {
        border-color: #dc2626;
        box-shadow: 0 4px 16px rgba(220, 38, 38, 0.2);
    }

    .case-card.selected {
        border-color: #dc2626;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }

    /* Prediction results */
    .prediction-results {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .prediction-results h4 {
        color: #0c4a6e;
        margin-bottom: 1rem;
    }

    /* ECG viewer styling */
    .ecg-viewer {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }

    /* Clinical note */
    .clinical-note {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }

    /* Footer styling */
    .cardiac-footer {
        background: #1f2937;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }

    /* Color-coded diagnosis text */
    .diagnosis-norm { color: #00AA00; font-weight: bold; }
    .diagnosis-mi { color: #FF0000; font-weight: bold; }
    .diagnosis-sttc { color: #0066CC; font-weight: bold; }
    .diagnosis-cd { color: #8A2BE2; font-weight: bold; }
    .diagnosis-hyp { color: #FF8C00; font-weight: bold; }

    /* Performance grid */
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .performance-item {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def load_evaluation_data():
    """Load all evaluation data files"""
    try:
        # Base path to evaluation results
        base_path = Path('evaluation_results')

        # Load curated cases
        with open(base_path / 'curated_cases.json', 'r') as f:
            curated_cases = json.load(f)

        # Load model card
        with open(base_path / 'model_card.json', 'r') as f:
            model_card = json.load(f)

        # Load performance data
        with open(base_path / 'performance_data.json', 'r') as f:
            performance_data = json.load(f)

        return curated_cases, model_card, performance_data
    except FileNotFoundError as e:
        st.error(f"Evaluation data file not found: {e}")
        return None, None, None



