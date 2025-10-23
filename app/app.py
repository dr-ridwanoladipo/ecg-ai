"""
🫀 ECG Classification AI - Streamlit Application
Medical AI interface for cardiac arrhythmia analysis.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from ecg_ui_helpers import *


st.set_page_config(
    page_title="ECG Classification AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    # Load custom CSS
    load_custom_css()

    # Header
    st.markdown("""
    <div class="cardiac-header">
        <h1>🫀 ECG Diagnosis AI</h1>
        <p>AI-assisted interpretation of cardiac electrical activity</p>
        <p><strong>By Ridwan Oladipo, MD | AI Specialist</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading AI model and evaluation data..."):
        curated_cases, model_card, performance_data = load_evaluation_data()

    if curated_cases is None:
        st.error("Failed to load evaluation data. Please ensure all data files are present.")
        return

    # Status indicator
    st.markdown("""
    <div class="success-indicator">
        ✅ AI Model Ready | 96.2% MI Sensitivity | 99.97% MI Specificity | Clinical Precision
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with model card
    with st.sidebar:
        display_model_card(model_card)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🫀 ECG Prediction",
        "📈 Performance Metrics",
        "🏥 Clinical Case Explorer",
        "🔬 Robustness Testing"
    ])

    # TAB 1: ECG Prediction
    with tab1:
        st.markdown("")
        st.markdown("## 🫀 AI-Powered ECG Classification")

        # Case Selection
        st.markdown("")
        case_options = [f"Case {case['case_id']}: {case['description']}" for case in curated_cases]
        selected_case_idx = st.selectbox(
            "Select a patient case below to analyze their ECG with our AI model:",
            range(len(case_options)),
            format_func=lambda x: case_options[x],
            key="case_selector"
        )

        st.markdown("")
        selected_case = curated_cases[selected_case_idx]
        case_id = selected_case['case_id']

        # Session State
        if 'show_prediction' not in st.session_state:
            st.session_state.show_prediction = False
        if 'current_case_id' not in st.session_state:
            st.session_state.current_case_id = None

        # Reset prediction when switching case
        if st.session_state.current_case_id != case_id:
            st.session_state.show_prediction = False
            st.session_state.current_case_id = case_id

        # Demographics
        if not st.session_state.show_prediction:
            demographics = selected_case['demographics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Age", f"{demographics['age']:.0f} years")
            with col2:
                st.metric("Sex", demographics['sex'])
            with col3:
                st.metric("Heart Rate", f"{demographics.get('heart_rate', 'N/A')} bpm")
            with col4:
                st.metric("Rhythm", demographics.get('rhythm', 'N/A'))

        # ECG Display
        st.markdown('<div class="ecg-viewer">', unsafe_allow_html=True)

        view_type = st.radio(
            "ECG View:",
            ["Lead II (Single)", "12-Lead View"],
            horizontal=True,
            key="view_toggle"
        )

        display_view = "single" if view_type == "Lead II (Single)" else "12lead"

        # Before prediction
        if not st.session_state.show_prediction:
            st.markdown("#### Pre-Colored ECG Trace")
            st.info("ECG color-coded by true diagnosis. Click **Run AI Prediction** below to analyze.")
            overlay_type = "clean"

        # After prediction
        else:
            st.markdown("---")
            st.markdown("#### 🧠 AI Cardiac Analysis (Grad-CAM + Prediction)")
            st.info("Interpretability-enhanced AI diagnosis based on 12-lead ECG.")
            st.markdown("")

            # Show original ECG checkbox
            show_original = st.checkbox(
                "Show original ECG for comparison",
                value=False,
                key="show_original_toggle",
                help="Toggle to view the pre-colored ECG instead of Grad-CAM overlay."
            )

            overlay_type = "clean" if show_original else "gradcam"

        # Display ECG image
        display_ecg_image(case_id, display_view, overlay_type)
        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction Trigger
        if not st.session_state.show_prediction:
            st.markdown("---")
            if st.button(" **Run AI Prediction**", key="predict_btn", use_container_width=True):
                simulate_prediction_progress()
                st.success("✅ AI Prediction Complete!")
                st.session_state.show_prediction = True
                st.rerun()

        # Post-Prediction Report
        if st.session_state.show_prediction:
            st.markdown("")

            # Primary + Differential
            display_prediction_results(selected_case)

            # Explainability + Clinical note
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🧩 SHAP Demographic Attribution")
                display_shap_analysis(case_id)
            with col2:
                st.markdown("")
                display_clinical_note(selected_case)

    # TAB 2: Performance Metrics
    with tab2:
        st.markdown("")
        st.markdown("## Model Performance Analysis")
        st.markdown("Comprehensive evaluation results and clinical validation metrics")

        # Key metrics summary
        performance = model_card['performance']
        mi_metrics = performance['mi_clinical_metrics']

        col1, col2, col3, col4 = st.columns(4)

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

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{performance['test_accuracy']:.1%}</div>
                <div class="metric-label">Overall Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{performance['macro_f1']:.3f}</div>
                <div class="metric-label">Macro F1 Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Display all performance plots
        display_performance_plots()

    # TAB 3: Clinical Case Explorer
    with tab3:
        st.markdown("")
        st.markdown("## Clinical Case Explorer")
        create_case_explorer_grid(curated_cases)

    # TAB 4: Robustness Testing
    with tab4:
        st.markdown("")
        st.markdown("## Model Robustness Testing")
        display_robustness_results(performance_data)

    # Footer
    st.markdown("---")
    display_footer()


if __name__ == "__main__":
    main()