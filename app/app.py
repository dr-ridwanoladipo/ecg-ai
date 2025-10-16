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

# ================ 🛠 SIDEBAR TOGGLE ================
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

st.set_page_config(
    page_title="ECG Classification AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state
)

if st.button("🫀", help="Toggle sidebar"):
    st.session_state.sidebar_state = (
        'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
    )
    st.rerun()

st.markdown(
    '<div style="font-size:0.75rem; color:#6b7280; margin-top:-10px;">Menu</div>',
    unsafe_allow_html=True
)


def main():
    # Load custom CSS
    load_custom_css()

    # Header
    st.markdown("""
    <div class="cardiac-header">
        <h1>🫀 ECG Classification AI</h1>
        <p>AI-powered cardiac arrhythmia analysis with clinical precision</p>
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
        st.markdown("## 🫀 AI-Powered ECG Classification")
        st.markdown("Select a patient case below to analyze their ECG with our AI model.")

        # Case selection dropdown
        st.markdown("### 👥 Patient Case Selection")

        case_options = [f"Case {case['case_id']}: {case['description']}" for case in curated_cases]
        selected_case_idx = st.selectbox(
            "Choose a patient case for analysis:",
            range(len(case_options)),
            format_func=lambda x: case_options[x],
            key="case_selector"
        )

        selected_case = curated_cases[selected_case_idx]
        case_id = selected_case['case_id']

        # Initialize prediction state
        if 'show_prediction' not in st.session_state:
            st.session_state.show_prediction = False
        if 'current_case_id' not in st.session_state:
            st.session_state.current_case_id = None

        # Reset prediction state when switching cases
        if st.session_state.current_case_id != case_id:
            st.session_state.show_prediction = False
            st.session_state.current_case_id = case_id

        st.markdown(f"### 📋 Patient {case_id} - ECG Analysis")

        # Display patient demographics
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

        # ECG display controls
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<div class="ecg-viewer">', unsafe_allow_html=True)

            # View toggle
            view_type = st.radio(
                "ECG View:",
                ["Lead II (Single)", "12-Lead View"],
                horizontal=True,
                key="view_toggle"
            )

            # Determine display parameters
            display_view = "single" if view_type == "Lead II (Single)" else "12lead"
            overlay_type = "gradcam" if st.session_state.show_prediction else "clean"

            # Display ECG
            if not st.session_state.show_prediction:
                st.markdown("#### 📊 Pre-Colored ECG Trace")
                st.info("ECG color-coded by true diagnosis. Click 'Run AI Prediction' to see model analysis.")
            else:
                st.markdown("#### 🧠 AI Analysis with Grad-CAM Attribution")
                st.success("AI prediction complete with interpretability overlay.")

            display_ecg_image(case_id, display_view, overlay_type)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("#### 🎯 AI Analysis Controls")

            # Case information
            true_class = selected_case['true_class']
            color_class = get_diagnosis_color_class(true_class)

            st.markdown(f"**True Diagnosis:** <span class='{color_class}'>{true_class}</span>",
                        unsafe_allow_html=True)

            st.markdown("---")

            # Prediction button
            if st.button("🔮 **Run AI Prediction**", key="predict_btn", use_container_width=True):
                simulate_prediction_progress()
                st.success("✅ AI Prediction Complete!")
                st.session_state.show_prediction = True
                st.rerun()

            # Show prediction results
            if st.session_state.show_prediction:
                st.markdown("#### 📊 Prediction Results")
                display_prediction_results(selected_case)

        # Post-prediction analysis
        if st.session_state.show_prediction:
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🧠 SHAP Demographics Analysis")
                display_shap_analysis(case_id)

            with col2:
                st.markdown("#### 📝 Clinical Assessment")
                display_clinical_note(selected_case)

    # TAB 2: Performance Metrics
    with tab2:
        st.markdown("## 📈 Model Performance Analysis")
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
        st.markdown("## 🏥 Clinical Case Explorer")
        create_case_explorer_grid(curated_cases)

    # TAB 4: Robustness Testing
    with tab4:
        st.markdown("## 🔬 Model Robustness Testing")
        display_robustness_results(performance_data)

    # Footer
    st.markdown("---")
    display_footer()


if __name__ == "__main__":
    main()