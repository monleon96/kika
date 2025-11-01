"""
Settings Page

Configure application preferences and user settings
"""

import streamlit as st
import sys
from pathlib import Path
import json

# Page config
st.set_page_config(page_title="Settings - KIKA", page_icon="⚙️", layout="wide")

# Header
st.title("⚙️ Settings")
st.markdown("Configure your KIKA experience")
st.markdown("---")

# Settings categories
tab1, tab2, tab3, tab4 = st.tabs(["🎨 Appearance", "📊 Plot Defaults", "💾 Export", "👤 Profile"])

with tab1:
    st.header("Appearance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Theme")
        theme = st.selectbox(
            "Color theme",
            ["Light", "Dark", "Auto"],
            help="Choose your preferred color scheme"
        )
        
        st.info("ℹ️ Theme changes will be applied in future versions")
        
        st.subheader("Layout")
        default_layout = st.selectbox(
            "Default layout",
            ["Wide", "Centered"],
            help="Default page layout"
        )
        
        sidebar_state = st.selectbox(
            "Sidebar default state",
            ["Expanded", "Collapsed", "Auto"],
            help="How sidebar appears on page load"
        )
    
    with col2:
        st.subheader("Font & Size")
        font_size = st.slider("Base font size", 12, 20, 14)
        code_font = st.selectbox("Code font", ["Monospace", "Consolas", "Monaco"])
        
        st.subheader("Accessibility")
        high_contrast = st.checkbox("High contrast mode")
        reduce_animations = st.checkbox("Reduce animations")
        screen_reader = st.checkbox("Screen reader optimizations")

with tab2:
    st.header("Plot Default Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Figure Size")
        default_width = st.slider("Default width (inches)", 6, 16, 10)
        default_height = st.slider("Default height (inches)", 4, 12, 6)
        default_dpi = st.selectbox("Default DPI", [100, 150, 200, 300], index=1)
        
        st.subheader("Axes")
        default_grid = st.checkbox("Show grid by default", value=True)
        default_legend = st.checkbox("Show legend by default", value=True)
        default_legend_loc = st.selectbox(
            "Legend location",
            ["best", "upper right", "upper left", "lower right", "lower left"]
        )
    
    with col2:
        st.subheader("Line Styles")
        default_linewidth = st.slider("Line width", 0.5, 5.0, 2.0, 0.5)
        default_markersize = st.slider("Marker size", 2, 12, 6)
        
        st.subheader("Colors")
        color_palette = st.selectbox(
            "Default color palette",
            ["Default", "Colorblind-friendly", "Grayscale", "Vibrant", "Pastel"]
        )
        
        st.markdown("**Custom colors:**")
        cols = st.columns(5)
        colors = []
        for i, col in enumerate(cols):
            with col:
                color = st.color_picker(f"Color {i+1}", f"#{['667eea', 'ff6b6b', '4ecdc4', 'ffe66d', '95e1d3'][i]}")
                colors.append(color)

with tab3:
    st.header("Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Default Export Format")
        default_format = st.selectbox(
            "Image format",
            ["PNG", "PDF", "SVG", "JPG"],
            help="Default format for downloading plots"
        )
        
        export_dpi = st.selectbox(
            "Export DPI",
            [100, 150, 200, 300, 600],
            index=3,
            help="Higher DPI = better quality but larger file size"
        )
        
        transparent_bg = st.checkbox("Transparent background (PNG/SVG)", value=False)
    
    with col2:
        st.subheader("File Naming")
        filename_template = st.text_input(
            "Filename template",
            value="kika_{datatype}_{timestamp}",
            help="Available variables: {datatype}, {timestamp}, {mt}, {energy}"
        )
        
        st.markdown("**Example filename:**")
        st.code("kika_cross_section_2024-11-01_143052.png")
        
        st.subheader("Data Export")
        data_export_format = st.selectbox(
            "Data format",
            ["CSV", "JSON", "HDF5", "Parquet"],
            help="Format for exporting numerical data"
        )

with tab4:
    st.header("User Profile")
    
    st.info("🚧 User authentication and profiles coming soon!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Account Information")
        st.text_input("Username", value="Guest", disabled=True)
        st.text_input("Email", value="guest@kika.app", disabled=True)
        
        st.markdown("---")
        st.subheader("Subscription")
        st.markdown("**Plan:** Free (MVP)")
        st.markdown("**Features:**")
        st.markdown("""
        - ✓ Basic ACE visualization
        - ✓ Cross section plotting
        - ✓ Angular distribution plotting
        - ✗ Batch processing
        - ✗ Cloud storage
        - ✗ API access
        """)
    
    with col2:
        st.subheader("Usage Statistics")
        st.metric("Files processed", "0", help="Total files analyzed")
        st.metric("Plots generated", "0", help="Total plots created")
        st.metric("Storage used", "0 MB / ∞", help="Data storage")
        
        st.markdown("---")
        st.subheader("Preferences")
        auto_save = st.checkbox("Auto-save session", value=False, disabled=True)
        email_notifications = st.checkbox("Email notifications", value=False, disabled=True)
        share_analytics = st.checkbox("Share anonymous usage data", value=False, disabled=True)

# Action buttons
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        st.success("Settings saved successfully!")
        st.balloons()

with col2:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.warning("Settings reset to defaults")
        st.rerun()

with col3:
    if st.button("📥 Import Settings", use_container_width=True):
        st.info("Upload a settings JSON file (coming soon)")

with col4:
    if st.button("📤 Export Settings", use_container_width=True):
        settings_dict = {
            "theme": theme,
            "layout": default_layout,
            "plot_defaults": {
                "width": default_width,
                "height": default_height,
                "dpi": default_dpi,
            }
        }
        st.download_button(
            "Download settings.json",
            data=json.dumps(settings_dict, indent=2),
            file_name="kika_settings.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Settings • KIKA v0.1.0 (MVP)</p>
</div>
""", unsafe_allow_html=True)
