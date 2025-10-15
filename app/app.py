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
    display_prediction_results
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

        st.markdown(f"### 📋 Patient {case_id} - ECG Visualization")
        display_ecg_image(case_id, view_type="single", overlay_type="clean")

        true_class = selected_case['true_class']
        color_class = get_diagnosis_color_class(true_class)
        st.markdown(f"**True Diagnosis:** <span class='{color_class}'>{true_class}</span>", unsafe_allow_html=True)

        if st.button("🔮 Run AI Prediction", key="predict_btn"):
            simulate_prediction_progress()
            st.success("✅ AI prediction complete for selected case.")
            display_prediction_results(selected_case)


# ================ EXECUTION ================
if __name__ == "__main__":
    main()
