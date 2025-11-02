"""
KIKA - Nuclear Data Viewer
Main application entry point

Powered by MCNPy
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import mcnpy
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure page
st.set_page_config(
    page_title="KIKA",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/monleon96/MCNPy',
        'Report a bug': "https://github.com/monleon96/MCNPy/issues",
        'About': """
        # KIKA - Nuclear Data Viewer
        
        A modern interface for visualizing and analyzing nuclear data.
        
        **Powered by MCNPy**
        
        Version: 0.1.0 (MVP)
        """
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">⚛️ KIKA</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Nuclear Data Visualization & Analysis Platform</p>', unsafe_allow_html=True)

# Welcome section
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 📊 ACE Data Viewer")
    st.markdown("""
    Upload and visualize ACE format nuclear data files:
    - **Cross sections** for various reactions
    - **Angular distributions** at different energies
    - **Multi-library comparisons** (JEFF, JENDL, ENDF/B)
    """)
    if st.button("Open ACE Viewer →", key="ace_btn"):
        st.switch_page("pages/1_📊_ACE_Viewer.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 📈 ENDF Data Viewer")
    st.markdown("""
    Explore ENDF-6 format evaluated nuclear data:
    - **Angular distributions** (MF4 Legendre expansions)
    - **Uncertainty bands** (MF34 covariance data)
    - **Library comparisons** with uncertainties
    """)
    if st.button("Open ENDF Viewer →", key="endf_btn"):
        st.switch_page("pages/2_📈_ENDF_Viewer.py")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Additional features row
col3, col4, col5 = st.columns(3)

with col3:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 🔧 NJOY Processing")
    st.markdown("""
    Generate ACE files from ENDF data:
    - **ENDF to ACE conversion**
    - **Temperature selection**
    - **Automatic versioning**
    """)
    if st.button("Open NJOY →", key="njoy_btn"):
        st.switch_page("pages/3_🔧_NJOY_Processing.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 📂 Covariance Data")
    st.markdown("""
    Analyze nuclear data uncertainties:
    - **Covariance matrices**
    - **Correlation plots**
    - **Uncertainty propagation**
    """)
    st.info("🚧 Coming soon!")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    st.markdown("""
    Configure your experience:
    - **NJOY configuration**
    - **Plot styling**
    - **Export options**
    """)
    if st.button("Open Settings →", key="settings_btn"):
        st.switch_page("pages/5_⚙️_Settings.py")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Powered by <strong>MCNPy</strong> | Built with Streamlit</p>
    <p style='font-size: 0.9rem;'>
        <a href='https://github.com/monleon96/MCNPy' target='_blank'>GitHub</a> • 
        <a href='https://mcnpy.readthedocs.io' target='_blank'>Documentation</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with global file management
from components.file_sidebar import render_file_upload_sidebar

render_file_upload_sidebar()

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 Quick Start")
    st.markdown("""
    1. **Upload files** using the uploader above
    2. Navigate to **ACE** or **ENDF Viewer**
    3. Select loaded files and configure plots
    4. Generate and export visualizations
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    with st.expander("Supported file formats"):
        st.markdown("""
        - **ACE**: `.ace`, `.02c`, `.20c`, etc.
        - **ENDF**: `.txt`, `.endf`, `.endf6`
        - **Auto-detection** tries both formats
        """)
    
    with st.expander("File management"):
        st.markdown("""
        - Files uploaded on any page are **globally accessible**
        - Use **Auto** mode for automatic type detection
        - Specify type manually for faster upload
        - Remove files individually or clear all at once
        """)
