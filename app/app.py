"""
🫀 ECG Classification AI - Streamlit Application
Medical AI interface for cardiac arrhythmia analysis.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import streamlit as st
from ecg_ui_helpers import load_custom_css

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
    # Load custom CSS for medical-grade UI
    load_custom_css()

    st.markdown("""
    <div class="cardiac-header">
        <h1>🫀 ECG Classification AI</h1>
        <p>AI-powered cardiac arrhythmia analysis with clinical precision</p>
        <p><strong>By Ridwan Oladipo, MD | AI Specialist</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Custom styling loaded successfully.")


# ================ EXECUTION ================
if __name__ == "__main__":
    main()
