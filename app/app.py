"""
🫀 ECG Classification AI - Streamlit Application
Medical AI interface for cardiac arrhythmia analysis.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import streamlit as st
from ecg_ui_helpers import (
    load_custom_css,
    load_evaluation_data,
    display_model_card,
    display_ecg_image,
    simulate_prediction_progress,
    get_diagnosis_color_class,
    display_prediction_results,
    display_shap_analysis,
    display_clinical_note,
    display_performance_plots,
    create_case_explorer_grid,
    get_case_summary,
    display_robustness_results,
    display_footer
)

# ================ SIDEBAR TOGGLE ================
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

st.set_page_config(
    page_title="ECG Classification AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state
)

# Sidebar toggle button
if st.button("🫀", help="Toggle sidebar"):
    st.session_state.sidebar_state = (
        'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
    )
    st.rerun()

st.markdown(
    '<div style="font-size:0.75rem; color:#6b7280; margin-top:-10px;">Menu</div>',
    unsafe_allow_html=True
)

# ================ MAIN APP ================
def main():
    """Main application entry point"""
    load_custom_css()

    st.markdown("""
    <div class="cardiac-header">
        <h1>🫀 ECG Classification AI</h1>
        <p>AI-powered cardiac arrhythmia analysis with clinical precision</p>
        <p><strong>By Ridwan Oladipo, MD | AI Specialist</strong></p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading AI model and evaluation data..."):
        curated_cases, model_card, performance_data = load_evaluation_data()

    if curated_cases is None:
        st.error("Failed to load evaluation data. Please ensure all data files are present.")
        return

    st.markdown("""
    <div class="success-indicator">
        ✅ AI Model Ready | 96.2% MI Sensitivity | 99.97% MI Specificity | Clinical Precision
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        display_model_card(model_card)

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

        if 'show_prediction' not in st.session_state:
            st.session_state.show_prediction = False
        if 'current_case_id' not in st.session_state:
            st.session_state.current_case_id = None

        if st.session_state.current_case_id != case_id:
            st.session_state.show_prediction = False
            st.session_state.current_case_id = case_id

        st.markdown(f"### 📋 Patient {case_id} - ECG Analysis")

        col_view, col_empty = st.columns([0.6, 0.4])
        with col_view:
            view_type = st.radio(
                "ECG View:",
                ["Lead II (Single)", "12-Lead View"],
                horizontal=True,
                key="view_toggle"
            )
        display_view = "single" if view_type == "Lead II (Single)" else "12lead"
        overlay_type = "gradcam" if st.session_state.show_prediction else "clean"

        st.markdown(f"#### 📊 ECG Display ({view_type})")
        display_ecg_image(case_id, view_type=display_view, overlay_type=overlay_type)

        true_class = selected_case['true_class']
        color_class = get_diagnosis_color_class(true_class)
        st.markdown(f"**True Diagnosis:** <span class='{color_class}'>{true_class}</span>", unsafe_allow_html=True)

        if not st.session_state.show_prediction:
            st.info("Click 'Run AI Prediction' to analyze this ECG with the AI model.")
        else:
            st.success("AI prediction complete with Grad-CAM overlay and explainability details below.")

        if st.button("🔮 Run AI Prediction", key="predict_btn"):
            simulate_prediction_progress()
            st.session_state.show_prediction = True
            st.success("✅ AI prediction complete for selected case.")
            st.rerun()

        if st.session_state.show_prediction:
            display_prediction_results(selected_case)
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
        st.markdown("Comprehensive evaluation results and clinical validation metrics.")
        display_performance_plots()

    # TAB 3: Clinical Case Explorer
    with tab3:
        st.markdown("## 🏥 Clinical Case Explorer")
        st.markdown("Explore curated ECG cases with clinical summaries below.")

        for case in curated_cases:
            get_case_summary(case)

        create_case_explorer_grid(curated_cases)

    # TAB 4: Robustness Testing
    with tab4:
        st.markdown("## 🔬 Model Robustness Testing")
        display_robustness_results(performance_data)

    # FOOTER
    st.markdown("---")
    display_footer()


# ================ EXECUTION ================
if __name__ == "__main__":
    main()
