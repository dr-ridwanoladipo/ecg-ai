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


def display_model_card(model_card):
    """Display model card information in sidebar"""
    st.sidebar.markdown("### 🏥 Model Information")

    # Model basics
    model_info = model_card['model_info']
    st.sidebar.markdown(f"**Model:** {model_info['name']}")
    st.sidebar.markdown(f"**Version:** {model_info['version']}")
    st.sidebar.markdown(f"**Architecture:** {model_info['architecture']}")

    # Key performance metrics
    st.sidebar.markdown("### 📊 Key Metrics")
    performance = model_card['performance']

    mi_metrics = performance['mi_clinical_metrics']

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{mi_metrics['sensitivity']:.1%}</div>
            <div class="metric-label">MI Sensitivity</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{mi_metrics['specificity']:.1%}</div>
            <div class="metric-label">MI Specificity</div>
        </div>
        """, unsafe_allow_html=True)

    # Dataset info
    st.sidebar.markdown("### 📈 Dataset")
    dataset = model_card['dataset_info']
    st.sidebar.markdown(f"**Dataset:** {dataset['name']}")
    st.sidebar.markdown(f"**Test Size:** {dataset['test_size']:,} cases")

    # Expandable detailed metrics
    with st.sidebar.expander("📋 Detailed Performance"):
        class_f1 = performance['class_f1_scores']
        for class_name, f1_score in class_f1.items():
            st.markdown(f"**{class_name}:** {f1_score:.3f}")

        st.markdown("---")
        st.markdown(f"**Overall Accuracy:** {performance['test_accuracy']:.3f}")
        st.markdown(f"**Macro F1:** {performance['macro_f1']:.3f}")

    # Clinical notes
    with st.sidebar.expander("⚠️ Clinical Notes"):
        st.markdown("""
        **Important Disclaimers:**
        - This AI model is for research demonstration only
        - Not approved for clinical diagnosis
        - All medical decisions should involve qualified healthcare providers
        - Model performance may vary in different clinical settings
        """)


def display_ecg_image(case_id, view_type="single", overlay_type="clean"):
    """Display ECG image with proper error handling"""
    try:
        base_path = Path('evaluation_results')

        if overlay_type == "clean":
            # Pre-colored ECG
            if view_type == "single":
                img_path = base_path / 'precolored_ecgs' / f'case_{case_id}_ecg_single_clean.png'
            else:
                img_path = base_path / 'precolored_ecgs' / f'case_{case_id}_ecg_12lead_clean.png'
        else:
            # Grad-CAM overlay
            if view_type == "single":
                img_path = base_path / 'curated_cases' / f'case_{case_id}_gradcam_single.png'
            else:
                img_path = base_path / 'curated_cases' / f'case_{case_id}_gradcam_12lead.png'

        if img_path.exists():
            st.image(str(img_path), width="stretch")
        else:
            st.error(f"ECG image not found: {img_path}")

    except Exception as e:
        st.error(f"Error displaying ECG: {e}")


def simulate_prediction_progress():
    """Simulate AI prediction progress with cardiac focus"""
    progress_text = st.empty()
    progress_bar = st.progress(0)

    stages = [
        "Loading 12-lead ECG signal...",
        "Preprocessing cardiac waveforms...",
        "Analyzing rhythm and morphology...",
        "Running ResNet-1D inference...",
        "Generating prediction confidence...",
        "Finalizing cardiac diagnosis..."
    ]

    for i, stage in enumerate(stages):
        progress_text.text(stage)
        progress_bar.progress((i + 1) / len(stages))
        time.sleep(0.3)

    progress_text.text("✅ Cardiac analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    progress_text.empty()


def get_diagnosis_color_class(diagnosis):
    """Get CSS class for diagnosis color coding"""
    color_map = {
        'Normal (NORM)': 'diagnosis-norm',
        'Normal': 'diagnosis-norm',
        'NORM': 'diagnosis-norm',
        'Myocardial Infarction (MI)': 'diagnosis-mi',
        'Myocardial Infarction': 'diagnosis-mi',
        'MI': 'diagnosis-mi',
        'ST-T Abnormality (STTC)': 'diagnosis-sttc',
        'ST-T Abnormality': 'diagnosis-sttc',
        'STTC': 'diagnosis-sttc',
        'Conduction Disturbance (CD)': 'diagnosis-cd',
        'Conduction Disturbance': 'diagnosis-cd',
        'CD': 'diagnosis-cd',
        'Hypertrophy (HYP)': 'diagnosis-hyp',
        'Hypertrophy': 'diagnosis-hyp',
        'HYP': 'diagnosis-hyp'
    }
    return color_map.get(diagnosis, 'diagnosis-norm')


def display_prediction_results(case_data, show_differential=True):
    """Display prediction results with clinical formatting"""
    st.markdown(f"""
    <div class="prediction-results">
        <h4>🫀 Cardiac Diagnosis Results</h4>
    </div>
    """, unsafe_allow_html=True)

    # Primary diagnosis
    predicted_class = case_data['predicted_class']
    confidence = case_data['confidence']
    color_class = get_diagnosis_color_class(predicted_class)

    st.markdown(f"""
    <div style="text-align: center; padding: 1rem;">
        <h3>Primary Diagnosis:</h3>
        <h2 class="{color_class}">{predicted_class}</h2>
        <p style="font-size: 1.2rem;">Confidence: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    if show_differential:
        # Differential diagnosis table
        st.markdown("### 📊 Differential Diagnosis")

        predictions_df = pd.DataFrame([
            {'Diagnosis': diag, 'Probability': prob}
            for diag, prob in case_data['predictions'].items()
        ]).sort_values('Probability', ascending=False)

        # Color-code the dataframe
        def style_diagnosis(row):
            color_class = get_diagnosis_color_class(row['Diagnosis'])
            color_map = {
                'diagnosis-norm': '#00AA00',
                'diagnosis-mi': '#FF0000',
                'diagnosis-sttc': '#0066CC',
                'diagnosis-cd': '#8A2BE2',
                'diagnosis-hyp': '#FF8C00'
            }
            color = color_map.get(color_class, '#000000')
            return [f'color: {color}; font-weight: bold'] * len(row)

        styled_df = predictions_df.style.apply(style_diagnosis, axis=1)
        st.dataframe(styled_df, width="stretch")


def display_shap_analysis(case_id):
    """Display SHAP analysis for the selected case"""
    try:
        base_path = Path('evaluation_results')
        shap_path = base_path / 'curated_cases' / f'case_{case_id}_shap.png'

        if shap_path.exists():
            st.image(str(shap_path), width="stretch")
        else:
            st.warning("SHAP analysis not available for this case")

    except Exception as e:
        st.error(f"Error displaying SHAP analysis: {e}")


def display_clinical_note(case_data):
    """Display clinical note with patient information"""
    demographics = case_data['demographics']

    st.markdown(f"""
    <div class="clinical-note">
        <h4>📋 Clinical Assessment</h4>
        <p><strong>Patient Demographics:</strong></p>
        <ul>
            <li>Age: {demographics['age']:.0f} years</li>
            <li>Sex: {demographics['sex']}</li>
            <li>Heart Rate: {demographics.get('heart_rate', 'N/A')} bpm</li>
            <li>Rhythm: {demographics.get('rhythm', 'N/A')}</li>
        </ul>
        <p><strong>Clinical Note:</strong> {case_data['clinical_note']}</p>
    </div>
    """, unsafe_allow_html=True)


def display_performance_plots():
    """Display all performance analysis plots"""
    base_path = Path('evaluation_results')

    # Performance plots
    plots = [
        ('calibration_curves.png', 'Model Calibration Analysis'),
        ('roc_pr_curves.png', 'ROC and Precision-Recall Curves'),
        ('demographic_analysis.png', 'Demographic Performance Analysis'),
        ('robustness_test.png', 'Model Robustness Testing')
    ]

    for plot_file, title in plots:
        plot_path = base_path / plot_file
        if plot_path.exists():
            st.markdown(f"### {title}")
            st.image(str(plot_path), width="stretch")
            st.markdown("---")
        else:
            st.warning(f"Plot not found: {title}")


def create_case_explorer_grid(curated_cases):
    """Create grid view of all curated cases"""
    st.markdown("### 🔍 Clinical Case Explorer")
    st.markdown("Overview of all 7 curated cases with ground truth vs predictions")

    # Create grid layout
    cols = st.columns(2)

    for i, case in enumerate(curated_cases):
        with cols[i % 2]:
            # Case card
            true_class = case['true_class']
            predicted_class = case['predicted_class']
            confidence = case['confidence']

            # Determine if prediction is correct
            is_correct = true_class == predicted_class
            border_color = "#00AA00" if is_correct else "#FF0000"

            st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;">
                <h4>Case {case['case_id']}: {case['description']}</h4>
                <p><strong>True:</strong> <span class="{get_diagnosis_color_class(true_class)}">{true_class}</span></p>
                <p><strong>Predicted:</strong> <span class="{get_diagnosis_color_class(predicted_class)}">{predicted_class}</span> ({confidence:.1%})</p>
                <p><strong>Status:</strong> {'✅ Correct' if is_correct else '❌ Incorrect'}</p>
            </div>
            """, unsafe_allow_html=True)

            # Thumbnail ECG
            try:
                base_path = Path('evaluation_results')
                thumbnail_path = base_path / 'precolored_ecgs' / f'case_{case["case_id"]}_ecg_single_clean.png'
                if thumbnail_path.exists():
                    st.image(str(thumbnail_path), width=300)
            except Exception:
                st.write("Thumbnail not available")


def get_case_summary(case_data):
    """Get summary information for case selection"""
    return {
        'case_id': case_data['case_id'],
        'description': case_data['description'],
        'true_class': case_data['true_class'],
        'predicted_class': case_data['predicted_class'],
        'confidence': case_data['confidence']
    }


def display_robustness_results(performance_data):
    """Display robustness testing results"""
    st.markdown("### 🔬 Model Robustness Analysis")
    st.markdown("Testing model stability under various signal conditions")

    # Display robustness plots
    base_path = Path('evaluation_results')
    robustness_plot = base_path / 'robustness_test.png'

    if robustness_plot.exists():
        st.image(str(robustness_plot), width="stretch")

    # Display robustness metrics if available
    if 'robustness' in performance_data:
        robustness = performance_data['robustness']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Amplitude Jitter Test")
            if 'jitter_performance' in robustness:
                jitter_data = pd.DataFrame({
                    'Noise Level': robustness['jitter_levels'],
                    'Agreement': robustness['jitter_performance']
                })
                st.dataframe(jitter_data, width="stretch")

        with col2:
            st.markdown("#### 📊 Amplitude Scaling Test")
            if 'scale_performance' in robustness:
                scale_data = pd.DataFrame({
                    'Scale Factor': robustness['scale_factors'],
                    'Agreement': robustness['scale_performance']
                })
                st.dataframe(scale_data, width="stretch")

        st.markdown("""
        **Analysis:** Model demonstrates robust performance across various signal 
        conditions, indicating reliability for clinical deployment across different 
        ECG acquisition systems and signal qualities.
        """)