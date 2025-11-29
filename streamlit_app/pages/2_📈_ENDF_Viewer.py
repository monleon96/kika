"""
ENDF Data Viewer Page

Upload and visualize ENDF format nuclear data files with uncertainties
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import io
import traceback
from datetime import datetime
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kika.endf.read_endf import read_endf
from kika.plotting import PlotBuilder
import matplotlib.pyplot as plt
import numpy as np

# Import config history utilities
from components.saved_configs import render_saved_configs_sidebar
from components.clear_cache import render_clear_cache_button
from utils.config_history import save_configuration
from utils.backend_auth import handle_verification_query, require_user, render_account_sidebar

# Page config
st.set_page_config(page_title="ENDF Viewer - KIKA", page_icon="📈", layout="wide")

handle_verification_query()
current_user = require_user()
render_account_sidebar(current_user)

# Custom CSS
st.markdown("""
    <style>
    .upload-section {
        padding: 1rem;
        border: 2px dashed #667eea;
        border-radius: 0.5rem;
        background: #f8f9fa;
        margin-bottom: 1rem;
    }
    .plot-container {
        padding: 1rem;
        background: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        background: #e3f2fd;
        color: #1976d2;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("📈 ENDF Data Viewer")
st.markdown("Upload and visualize ENDF format nuclear data files with uncertainties")
st.markdown("---")

# Mark current page for navigation
st.session_state.current_page = "pages/2_📈_ENDF_Viewer.py"

# Sidebar - Global file management
from components.file_sidebar import render_file_upload_sidebar

render_file_upload_sidebar()

# Get globally loaded ENDF files
from utils.file_manager import get_endf_files

st.session_state.endf_objects = {name: data['object'] for name, data in get_endf_files().items()}

# Render saved configurations and the shared clear-cache button at the end of the sidebar
with st.sidebar:
    st.markdown("---")
    render_saved_configs_sidebar()
    # Render clear-cache button as the last sidebar item
    render_clear_cache_button(key_prefix="clear_cache_endf")

# Initialize session state for plot data selections
if 'endf_plot_selections' not in st.session_state:
    st.session_state.endf_plot_selections = []

# Initialize unique ID counter for selections
if 'endf_selection_id_counter' not in st.session_state:
    st.session_state.endf_selection_id_counter = 0

# Main content
# Create a small two-tab layout: Home (basic info) and Viewer (the existing interactive UI)
tab_home, tab_viewer = st.tabs(["Home", "Viewer"])

with tab_home:
    st.header("📖 About ENDF Viewer")
    
    # Quick stats
    n_files = len(st.session_state.endf_objects)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is ENDF?
        
        ENDF (Evaluated Nuclear Data File) is a standard format for storing 
        nuclear reaction data, cross sections, angular distributions, and 
        uncertainties used in nuclear physics simulations.
        
        ### Supported Data Types:
        - **MF4**: Angular distributions (Legendre coefficients)
        - **MF34**: Angular distribution uncertainties (covariance data)
        
        ### Features:
        - **Multi-file comparison** - overlay data from different sources
        - **Uncertainty visualization** - show confidence bands
        - **Interactive configuration** - customize every aspect of the plot
        - **High-quality export** - save publication-ready figures
        """)
    
    with col2:
        st.markdown("""
        ### Current Status
        """)
        st.metric("Loaded ENDF Files", n_files)
        
        if n_files > 0:
            st.markdown("**Loaded files:**")
            for name in st.session_state.endf_objects.keys():
                st.write(f"- `{name}`")
        else:
            st.info("👈 Upload ENDF files from the sidebar to get started")
        
        st.markdown("""
        ### Getting Started
        1. Upload ENDF files using the sidebar uploader
        2. Go to the **Viewer** tab
        3. Add data series and configure styling
        4. Customize labels, scales, and appearance
        5. Export your plot in various formats
        """)
    
    st.markdown("---")
    
    # Quick tips
    st.markdown("### 💡 Quick Tips")
    col_tip1, col_tip2, col_tip3 = st.columns(3)
    
    with col_tip1:
        st.info("""
        **📁 File Management**
        
        Files uploaded in the sidebar are shared across all pages.
        Upload once, use everywhere!
        """)
    
    with col_tip2:
        st.info("""
        **💾 Save Configurations**
        
        Save your plot configurations to quickly restore them later.
        """)
    
    with col_tip3:
        st.info("""
        **🎨 Customization**
        
        Every aspect of the plot is customizable - from colors to fonts to grid styles.
        """)

with tab_viewer:
    # Always show plotting interface, with informational message if no files
    if not st.session_state.endf_objects:
        st.info("👈 Upload ENDF files from the sidebar to get started. The plotting interface will be ready once files are loaded.")

    # Always show plotting interface (whether or not files are loaded)
    # Function to reset all plot settings to defaults
    def reset_all_plot_settings():
        """Reset all plot settings to default values"""
        # Clear series
        st.session_state.endf_plot_selections = []
        st.session_state.endf_selection_id_counter = 0
        
        # Reset all figure settings to defaults
        keys_to_delete = [
            'endf_plot_title', 'endf_x_label', 'endf_y_label',
            'endf_fig_width', 'endf_fig_height',
            'endf_title_fontsize', 'endf_label_fontsize', 'endf_legend_fontsize',
            'endf_show_legend', 'endf_legend_loc',
            'endf_log_x', 'endf_log_y',
            'endf_show_grid', 'endf_grid_alpha', 'endf_show_minor_grid', 'endf_minor_grid_alpha',
            'endf_tick_fontsize_x', 'endf_tick_fontsize_y',
            'endf_max_ticks_x', 'endf_max_ticks_y',
            'endf_rotate_x', 'endf_rotate_y',
            'endf_x_min', 'endf_x_max', 'endf_y_min', 'endf_y_max'
        ]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        
        # Also clear current config tracking
        if 'endf_current_config_id' in st.session_state:
            del st.session_state['endf_current_config_id']

    # Plotting interface header
    col_header, col_new = st.columns([4, 1])
    with col_header:
        st.header("🎨 Interactive Plot Configuration")
    with col_new:
        st.markdown("")  # Spacing
        if st.button("🆕 New Plot", key="new_plot_btn", help="Clear all series and start a new plot", width="stretch"):
            reset_all_plot_settings()
            st.success("✓ Cleared! Ready for a new plot.")
            st.rerun()

    # Callback function to reset all selections when data type changes
    def reset_on_data_type_change():
        """Reset plot selections and ID counter when data type changes"""
        st.session_state.endf_plot_selections = []
        st.session_state.endf_selection_id_counter = 0

    # Data type selection
    data_type = st.selectbox(
        "Select Data Type",
        ["Angular Distributions (MF4)", "Angular Distribution Uncertainties (MF34)"],
        help="Choose what type of data to visualize",
        key="endf_data_type_select",
        on_change=reset_on_data_type_change
    )
        
    st.markdown("---")

    # Create two-column layout: Settings (left) and Plot (right)
    col_settings, col_plot = st.columns([2, 3], gap="medium")

    # ========== LEFT COLUMN: Settings ==========
    with col_settings:
        st.markdown("### 📊 Plot Series Configuration")
        st.markdown("Each card represents one data series. Configure data source, uncertainties, and styling for each series individually.")
        
        # Series management buttons
        col_add, col_clear = st.columns([3, 1])
        
        with col_add:
            if st.button("➕ Add New Series", key="add_new_endf_series_btn", type="primary"):
                # Add a new empty series
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                # Get default file
                default_file = list(st.session_state.endf_objects.keys())[0] if st.session_state.endf_objects else None
                default_label = f"Series {len(st.session_state.endf_plot_selections) + 1}"
                default_mt = 2  # Elastic scattering as default
                default_order = 1  # First Legendre order as default
                
                # Determine if uncertainties are available
                default_has_unc = False
                if default_file:
                    endf_obj = st.session_state.endf_objects[default_file]
                    default_has_unc = 34 in endf_obj.mf.keys()
                
                new_series = {
                    'id': st.session_state.endf_selection_id_counter,
                    'file': default_file,
                    'mt': default_mt,
                    'order': default_order,
                    'label': default_label,
                    'type': 'angular' if data_type == "Angular Distributions (MF4)" else 'uncertainty',
                    'include_uncertainty': default_has_unc and data_type == "Angular Distributions (MF4)",
                    'uncertainty_sigma': 1.0,
                    'color': colors[len(st.session_state.endf_plot_selections) % len(colors)],
                    'linewidth': 2.0,
                    'linestyle': '-',
                    'marker': None,
                    'markersize': 6
                }
                st.session_state.endf_plot_selections.append(new_series)
                st.session_state.endf_selection_id_counter += 1
                st.rerun()
        
        with col_clear:
            if st.button("🗑️ Clear All", key="clear_all_endf_selections", width="stretch"):
                st.session_state.endf_plot_selections = []
                st.rerun()
        
        # Display series as cards
        if len(st.session_state.endf_plot_selections) == 0:
            st.info("👆 Click 'Add New Series' to start building your plot")
        else:
            # Show as expandable cards
            for idx, selection in enumerate(st.session_state.endf_plot_selections):
                sel_id = selection['id']
                
                # Create title for expander with more info
                mt_str = f"MT={selection['mt']}" if selection['mt'] is not None else "MT=?"
                label_str = selection.get('label', 'Untitled')
                
                if selection['type'] == 'angular':
                    order_str = f"L={selection.get('order', 1)}"
                    unc_str = " + σ" if selection.get('include_uncertainty', False) else ""
                    title = f"📈 {label_str} | {selection.get('file', 'No file')} ({mt_str}, {order_str}{unc_str})"
                else:
                    order_str = f"L={selection.get('order', 1)}"
                    title = f"📊 {label_str} | {selection.get('file', 'No file')} ({mt_str}, {order_str} uncertainty)"
                
                with st.expander(title, expanded=True):
                    # Data Source Configuration with inline Remove button
                    col_header, col_del = st.columns([5, 1])
                    with col_header:
                        st.markdown("**📂 Data Source**")
                    with col_del:
                        if st.button("🗑️", key=f"remove_endf_plot_series_{sel_id}", help="Remove this series"):
                            st.session_state.endf_plot_selections.pop(idx)
                            st.rerun()
                    
                    col_data1, col_data2, col_data3 = st.columns(3)
                    
                    with col_data1:
                        # File selection
                        file_list = list(st.session_state.endf_objects.keys())
                        current_file = selection.get('file')
                        if current_file and current_file in file_list:
                            file_index = file_list.index(current_file)
                        else:
                            file_index = 0
                        
                        new_file = st.selectbox(
                            "ENDF File",
                            file_list,
                            index=file_index,
                            key=f"endf_file_select_{sel_id}"
                        )
                        st.session_state.endf_plot_selections[idx]['file'] = new_file
                        
                        # Get available MTs based on selected file
                        available_mts = []
                        has_mf34 = False
                        if new_file:
                            endf_obj = st.session_state.endf_objects[new_file]
                            
                            if selection['type'] == 'angular' and 4 in endf_obj.mf:
                                available_mts = list(endf_obj.mf[4].mt.keys())
                            elif selection['type'] == 'uncertainty':
                                if 34 not in endf_obj.mf:
                                    st.warning("⚠️ This file does not contain MF34 (uncertainty) data. Please upload a file with covariance data or switch to 'Angular Distributions (MF4)'.")
                                else:
                                    available_mts = list(endf_obj.mf[34].mt.keys())
                            
                            has_mf34 = 34 in endf_obj.mf.keys()
                
                    with col_data2:
                        # MT selection
                        if new_file and available_mts:
                            current_mt = selection.get('mt')
                            if current_mt and current_mt in available_mts:
                                mt_index = available_mts.index(current_mt)
                            else:
                                mt_index = 0
                            
                            new_mt = st.selectbox(
                                "MT Number",
                                available_mts,
                                index=mt_index,
                                key=f"endf_mt_select_{sel_id}",
                                format_func=lambda x: f"MT={x}"
                            )
                            st.session_state.endf_plot_selections[idx]['mt'] = new_mt
                        else:
                            st.warning("No MT numbers available")
                            st.session_state.endf_plot_selections[idx]['mt'] = None
                
                    with col_data3:
                        # Order selection (Legendre polynomial order)
                        current_order = selection.get('order', 1)
                        new_order = st.number_input(
                            "Legendre Order (L)",
                            min_value=0,
                            max_value=20,
                            value=current_order,
                            step=1,
                            key=f"endf_order_select_{sel_id}",
                            help="Legendre polynomial order (0 for isotropic, 1 for first moment, etc.)"
                        )
                        st.session_state.endf_plot_selections[idx]['order'] = new_order
                
                    # Uncertainty options (only for Angular Distributions)
                    if selection['type'] == 'angular':
                        st.markdown("**📊 Uncertainty Options**")
                        
                        if not has_mf34:
                            st.info("ℹ️ No MF34 uncertainty data available for this file")
                        else:
                            col_unc1, col_unc2 = st.columns([2, 1])
                            
                            with col_unc1:
                                include_unc = st.checkbox(
                                    "Include uncertainty band",
                                    value=selection.get('include_uncertainty', False),
                                    key=f"endf_include_unc_{sel_id}",
                                    help="Show uncertainty band from MF34 covariance data"
                                )
                                st.session_state.endf_plot_selections[idx]['include_uncertainty'] = include_unc
                            
                            with col_unc2:
                                if include_unc:
                                    sigma_level = st.number_input(
                                        "Sigma level (σ)",
                                        min_value=0.5,
                                        max_value=3.0,
                                        value=selection.get('uncertainty_sigma', 1.0),
                                        step=0.5,
                                        key=f"endf_sigma_{sel_id}",
                                        help="Number of standard deviations (1σ = 68%, 2σ = 95%, 3σ = 99.7%)"
                                    )
                                    st.session_state.endf_plot_selections[idx]['uncertainty_sigma'] = sigma_level
                
                    # Label with Automatic/Custom options
                    st.markdown("**🏷️ Series Label**")
                    label_mode = st.radio(
                        "Label mode",
                        ["Automatic", "Custom"],
                        index=0 if selection.get('label_mode', 'auto') == 'auto' else 1,
                        key=f"endf_label_mode_{sel_id}",
                        horizontal=True,
                        help="Automatic updates label based on file/MT/order, Custom allows manual entry"
                    )
                    
                    if label_mode == "Automatic":
                        st.session_state.endf_plot_selections[idx]['label_mode'] = 'auto'
                        # Generate automatic label
                        try:
                            endf_obj = st.session_state.endf_objects[new_file]
                            isotope = endf_obj.isotope or "Unknown"
                            mt = selection['mt']
                            order = selection['order']
                            
                            if selection['type'] == 'angular':
                                auto_label = f"{isotope} MT={mt} L={order}"
                                # Add sigma information if uncertainty band is included
                                if selection.get('include_uncertainty', False):
                                    sigma_level = selection.get('uncertainty_sigma', 1.0)
                                    if sigma_level == 1.0:
                                        auto_label += " (±1σ)"
                                    else:
                                        auto_label += f" (±{sigma_level}σ)"
                            else:  # uncertainty only
                                auto_label = f"{isotope} MT={mt} L={order}"
                        except:
                            auto_label = f"Series {idx+1}"
                        
                        st.session_state.endf_plot_selections[idx]['label'] = auto_label
                        st.info(f"Label: {auto_label}")
                    else:
                        st.session_state.endf_plot_selections[idx]['label_mode'] = 'custom'
                        custom_label = st.text_input(
                            "Custom label",
                            value=selection.get('label', f"Series {idx+1}"),
                            key=f"endf_label_custom_{sel_id}"
                        )
                        st.session_state.endf_plot_selections[idx]['label'] = custom_label
                    
                    st.markdown("---")
                    
                    # Styling Options in expander
                    with st.expander("🎨 Styling Options", expanded=False):
                        # Color and line styling
                        col_left, col_right = st.columns([1, 1])
                        
                        with col_left:
                            # --- SINGLE-SET COLOR PALETTE (colored boxes + custom) ---
                            st.markdown("**Color**")

                            # Palette: (hex color, label)
                            palette = [
                                ("#1f77b4", "Blue"),
                                ("#ff7f0e", "Orange"),
                                ("#2ca02c", "Green"),
                                ("#d62728", "Red"),
                                ("#9467bd", "Purple"),
                                ("#8c564b", "Brown"),
                                ("#e377c2", "Magenta"),
                                ("#7f7f7f", "Gray"),
                                ("#bcbd22", "Yellow"),
                                ("#17becf", "Cyan"),
                            ]

                            current_color = selection.get('color', '#1f77b4')
                            use_custom = selection.get('use_custom_color', False)

                            # Create columns for color buttons - 6 colors per row
                            color_cols = st.columns(6)
                            selected_color = None
                            
                            for i, (hex_color, label) in enumerate(palette):
                                col_idx = i % 6
                                with color_cols[col_idx]:
                                    # Display colored box with HTML
                                    border_color = "#333" if hex_color == current_color and not use_custom else "#ddd"
                                    st.markdown(
                                        f"<div style=\"width: 100%; height: 20px; background: {hex_color}; border: 2px solid {border_color}; border-radius: 4px; margin-bottom: 4px;\"></div>",
                                        unsafe_allow_html=True
                                    )
                                    # Button below the colored box
                                    if st.button("●", key=f"endf_color_btn_{sel_id}_{i}", help=label, width="stretch"):
                                        selected_color = hex_color
                            
                            # Custom color option in new row
                            st.markdown("")
                            custom_col1, custom_col2 = st.columns([1, 5])
                            with custom_col1:
                                if st.button("🎨", key=f"endf_color_custom_btn_{sel_id}", help="Custom color", width="stretch"):
                                    st.session_state.endf_plot_selections[idx]['use_custom_color'] = True
                                    st.rerun()
                            
                            with custom_col2:
                                if use_custom or st.session_state.endf_plot_selections[idx].get('use_custom_color', False):
                                    new_color = st.color_picker(
                                        "Custom",
                                        value=current_color,
                                        key=f"endf_color_picker_{sel_id}",
                                        label_visibility="collapsed"
                                    )
                                    st.session_state.endf_plot_selections[idx]['color'] = new_color
                                    st.session_state.endf_plot_selections[idx]['use_custom_color'] = True
                            
                            # Update color if a preset was clicked
                            if selected_color:
                                st.session_state.endf_plot_selections[idx]['color'] = selected_color
                                st.session_state.endf_plot_selections[idx]['use_custom_color'] = False
                                st.rerun()

                            # Live preview - short and thick bar
                            preview_hex = st.session_state.endf_plot_selections[idx]['color']
                            st.markdown(
                                f"<div style='height: 12px; border-radius: 4px; "
                                f"background: {preview_hex}; margin-top: 8px;'></div>",
                                unsafe_allow_html=True
                            )
                            # --- end SINGLE-SET COLOR PALETTE ---
                        
                        with col_right:
                            # Line properties
                            st.markdown("**Line Style**")
                            
                            linestyle_options = {
                                '-': 'Solid',
                                '--': 'Dashed',
                                ':': 'Dotted',
                                '-.': 'Dash-dot'
                            }
                            current_linestyle = selection.get('linestyle', '-')
                            linestyle_index = list(linestyle_options.keys()).index(current_linestyle)
                            
                            new_linestyle = st.selectbox(
                                "Style",
                                list(linestyle_options.keys()),
                                index=linestyle_index,
                                format_func=lambda x: linestyle_options[x],
                                key=f"endf_linestyle_{sel_id}"
                            )
                            st.session_state.endf_plot_selections[idx]['linestyle'] = new_linestyle
                            
                            new_linewidth = st.slider(
                                "Width",
                                0.5, 5.0,
                                selection.get('linewidth', 2.0),
                                0.5,
                                key=f"endf_linewidth_{sel_id}"
                            )
                            st.session_state.endf_plot_selections[idx]['linewidth'] = new_linewidth
                            
                            st.markdown("**Markers**")
                            marker_options = {
                                None: 'None',
                                'o': 'Circle',
                                's': 'Square',
                                '^': 'Triangle',
                                'D': 'Diamond',
                                'x': 'X',
                                '+': 'Plus'
                            }
                            current_marker = selection.get('marker', None)
                            marker_index = list(marker_options.keys()).index(current_marker)
                            
                            new_marker = st.selectbox(
                                "Marker",
                                list(marker_options.keys()),
                                index=marker_index,
                                format_func=lambda x: marker_options[x],
                                key=f"endf_marker_{sel_id}"
                            )
                            st.session_state.endf_plot_selections[idx]['marker'] = new_marker
                            
                            if new_marker:
                                new_markersize = st.slider(
                                    "Size",
                                    2, 12,
                                    selection.get('markersize', 6),
                                    1,
                                    key=f"endf_markersize_{sel_id}"
                                )
                                st.session_state.endf_plot_selections[idx]['markersize'] = new_markersize
        
        st.markdown("---")
        
        # Define defaults based on data type
        st.markdown("### 🎨 Figure Configuration")
        
        default_fig_width = 10
        default_fig_height = 6
        default_title_fontsize = 14
        default_label_fontsize = 12
        default_tick_fontsize = 10
        default_legend_fontsize = 10
        
        if data_type == "Angular Distributions (MF4)":
            default_title = "Angular Distribution Comparison"
            default_x_label = "Energy (eV)"
            default_y_label = "Legendre Coefficient"
            default_log_x = True
            default_log_y = False
        else:  # Angular Distribution Uncertainties
            default_title = "Angular Distribution Uncertainty Comparison"
            default_x_label = "Energy (eV)"
            default_y_label = "Relative Uncertainty (%)"
            default_log_x = True
            default_log_y = False
        
        # Initialize figure size in session state
        if 'endf_fig_width' not in st.session_state:
            st.session_state.endf_fig_width = default_fig_width
        if 'endf_fig_height' not in st.session_state:
            st.session_state.endf_fig_height = default_fig_height
        fig_width = st.session_state.endf_fig_width
        fig_height = st.session_state.endf_fig_height
        
        # Labels, Title & Legend
        with st.expander("🏷️ Labels, Title & Legend", expanded=True):
            # Title
            st.markdown("**Title**")
            col_title, col_title_font = st.columns([3, 1])
            with col_title:
                plot_title = st.text_input(
                    "Label",
                    value=default_title,
                    key="endf_plot_title"
                )
            with col_title_font:
                title_fontsize = st.number_input("Size", min_value=8, max_value=32,
                                                value=default_title_fontsize, step=1, key="endf_title_fontsize")
            
            st.markdown("")
            
            # Axis labels
            st.markdown("**Axis Labels**")
            col_labels1, col_labels2 = st.columns(2)
            with col_labels1:
                col_xlabel, col_xlabel_font = st.columns([3, 1])
                with col_xlabel:
                    x_label = st.text_input("X-axis", value=default_x_label, key="endf_x_label")
                with col_xlabel_font:
                    label_fontsize = st.number_input("Size", min_value=8, max_value=24,
                                                    value=default_label_fontsize, step=1, key="endf_label_fontsize")
            
            with col_labels2:
                y_label = st.text_input("Y-axis", value=default_y_label, key="endf_y_label")
            
            st.markdown("")
            
            # Legend
            st.markdown("**Legend**")
            col_leg1, col_leg2, col_leg3 = st.columns([1, 2, 1])
            
            with col_leg1:
                show_legend = st.checkbox("Show legend", value=True, key="endf_show_legend")
            
            with col_leg2:
                if show_legend:
                    legend_options = ["best", "upper right", "upper left", "lower right", "lower left", "center", "center right"]
                    legend_location = st.selectbox(
                        "Position",
                        legend_options,
                        index=0,
                        key="endf_legend_loc"
                    )
            
            with col_leg3:
                if show_legend:
                    legend_fontsize = st.number_input("Size", min_value=6, max_value=20,
                                                     value=default_legend_fontsize, step=1, key="endf_legend_fontsize")
        
        # Zoom, Scales & Grid
        with st.expander("🔍 Zoom, Scales & Grid", expanded=True):
            # Zoom / Axis Limits (FIRST)
            st.markdown("**🔍 Zoom / Axis Limits**")
            st.info("💡 Leave empty to show full range automatically")
            
            col_zoom1, col_zoom2 = st.columns(2)
            
            with col_zoom1:
                st.markdown("**X-axis Range**")
                col_xmin, col_xmax = st.columns(2)
                with col_xmin:
                    x_min = st.text_input("Min", value="", key="endf_x_min", 
                                         placeholder="auto", help="Minimum X value (leave empty for auto)")
                with col_xmax:
                    x_max = st.text_input("Max", value="", key="endf_x_max", 
                                         placeholder="auto", help="Maximum X value (leave empty for auto)")
            
            with col_zoom2:
                st.markdown("**Y-axis Range**")
                col_ymin, col_ymax = st.columns(2)
                with col_ymin:
                    y_min = st.text_input("Min", value="", key="endf_y_min", 
                                         placeholder="auto", help="Minimum Y value (leave empty for auto)")
                with col_ymax:
                    y_max = st.text_input("Max", value="", key="endf_y_max", 
                                         placeholder="auto", help="Maximum Y value (leave empty for auto)")
            
            st.markdown("---")
            
            # Scales and Grid (AFTER Zoom)
            col_scale1, col_scale2 = st.columns(2)
            
            with col_scale1:
                st.markdown("**Scales**")
                if 'endf_log_x' not in st.session_state:
                    st.session_state.endf_log_x = default_log_x
                if 'endf_log_y' not in st.session_state:
                    st.session_state.endf_log_y = default_log_y
                log_x = st.checkbox("Logarithmic X-axis", value=st.session_state.endf_log_x, key="endf_log_x")
                log_y = st.checkbox("Logarithmic Y-axis", value=st.session_state.endf_log_y, key="endf_log_y")
            
            with col_scale2:
                st.markdown("**Grid**")
                if 'endf_show_grid' not in st.session_state:
                    st.session_state.endf_show_grid = True
                show_grid = st.checkbox("Show grid", value=st.session_state.endf_show_grid, key="endf_show_grid")
                if show_grid:
                    if 'endf_grid_alpha' not in st.session_state:
                        st.session_state.endf_grid_alpha = 0.3
                    grid_alpha = st.slider("Grid transparency", 0.0, 1.0,
                                          st.session_state.endf_grid_alpha, 0.1, key="endf_grid_alpha")
                    if 'endf_show_minor_grid' not in st.session_state:
                        st.session_state.endf_show_minor_grid = False
                    show_minor_grid = st.checkbox("Show minor grid", value=st.session_state.endf_show_minor_grid, key="endf_show_minor_grid")
        
        # Advanced: Tick parameters
        with st.expander("⚙️ Advanced: Tick Parameters"):
            col_tick1, col_tick2 = st.columns(2)
            
            with col_tick1:
                st.markdown("**X-axis Ticks**")
                if 'endf_tick_fontsize_x' not in st.session_state:
                    st.session_state.endf_tick_fontsize_x = default_tick_fontsize
                tick_fontsize_x = st.number_input("Tick font size", min_value=6, max_value=20,
                                                 value=st.session_state.endf_tick_fontsize_x, step=1, key="endf_tick_fontsize_x")
                if 'endf_max_ticks_x' not in st.session_state:
                    st.session_state.endf_max_ticks_x = 10
                max_ticks_x = st.number_input("Max ticks", min_value=3, max_value=20,
                                             value=st.session_state.endf_max_ticks_x, step=1, key="endf_max_ticks_x")
                if 'endf_rotate_x' not in st.session_state:
                    st.session_state.endf_rotate_x = 0
                rotate_x = st.number_input("Rotate labels (degrees)", min_value=0, max_value=90,
                                          value=st.session_state.endf_rotate_x, step=15, key="endf_rotate_x")
            
            with col_tick2:
                st.markdown("**Y-axis Ticks**")
                if 'endf_tick_fontsize_y' not in st.session_state:
                    st.session_state.endf_tick_fontsize_y = default_tick_fontsize
                tick_fontsize_y = st.number_input("Tick font size", min_value=6, max_value=20,
                                                 value=st.session_state.endf_tick_fontsize_y, step=1, key="endf_tick_fontsize_y")
                if 'endf_max_ticks_y' not in st.session_state:
                    st.session_state.endf_max_ticks_y = 10
                max_ticks_y = st.number_input("Max ticks", min_value=3, max_value=20,
                                             value=st.session_state.endf_max_ticks_y, step=1, key="endf_max_ticks_y")
                if 'endf_rotate_y' not in st.session_state:
                    st.session_state.endf_rotate_y = 0
                rotate_y = st.number_input("Rotate labels (degrees)", min_value=0, max_value=90,
                                          value=st.session_state.endf_rotate_y, step=15, key="endf_rotate_y")

    # ========== RIGHT COLUMN: Live Plot ==========
    with col_plot:
        st.markdown("### 📊 Plot Preview")
        
        # Default DPI for plot display
        default_display_dpi = 100
        
        # Check if we should generate plot
        if not st.session_state.endf_plot_selections:
            st.info("👈 Add data series from the left panel to see the plot here")
        else:
            try:
                # Create PlotBuilder with figure size
                builder = PlotBuilder(figsize=(fig_width, fig_height))
                
                # Add data for each selection
                errors = []
                for idx, selection in enumerate(st.session_state.endf_plot_selections):
                    try:
                        file_name = selection['file']
                        endf_obj = st.session_state.endf_objects[file_name]
                        mt = selection['mt']
                        order = selection['order']
                        
                        if selection['type'] == 'angular':
                            # Angular distributions from MF4
                            include_unc = selection.get('include_uncertainty', False)
                            sigma_level = selection.get('uncertainty_sigma', 1.0)
                            
                            plot_data = endf_obj.to_plot_data(
                                mf=4,
                                mt=mt,
                                order=order,
                                uncertainty=include_unc,
                                sigma=sigma_level
                            )
                            
                            # Apply styling
                            if isinstance(plot_data, tuple):
                                data_obj, unc_band = plot_data
                                data_obj = data_obj.apply_styling(
                                    label=selection['label'],
                                    color=selection['color'],
                                    linestyle=selection['linestyle'],
                                    linewidth=selection['linewidth'],
                                    marker=selection['marker'],
                                    markersize=selection.get('markersize', 6)
                                )
                                if unc_band and unc_band.color is None:
                                    unc_band.color = selection['color']
                                builder.add_data(data_obj, uncertainty=unc_band)
                            else:
                                plot_data = plot_data.apply_styling(
                                    label=selection['label'],
                                    color=selection['color'],
                                    linestyle=selection['linestyle'],
                                    linewidth=selection['linewidth'],
                                    marker=selection['marker'],
                                    markersize=selection.get('markersize', 6)
                                )
                                builder.add_data(plot_data)
                            
                        else:  # uncertainty only
                            plot_data = endf_obj.to_plot_data(
                                mf=34,
                                mt=mt,
                                order=order
                            )
                            
                            plot_data = plot_data.apply_styling(
                                label=selection['label'],
                                color=selection['color'],
                                linestyle=selection['linestyle'],
                                linewidth=selection['linewidth'],
                                marker=selection['marker'],
                                markersize=selection.get('markersize', 6)
                            )
                            
                            builder.add_data(plot_data)
                    
                    except Exception as e:
                        errors.append(f"Series {idx+1} ({selection.get('label', 'Untitled')}): {str(e)}")
                        st.warning(f"⚠️ Skipping series {idx+1}: {str(e)}")
                        continue
                
                # Check if we have any data to plot
                if len(builder._data_list) == 0:
                    st.error("No valid data to plot. All series encountered errors.")
                    if errors:
                        with st.expander("Error details"):
                            for error in errors:
                                st.write(f"• {error}")
                else:
                    # Configure plot
                    builder.set_labels(title=plot_title, x_label=x_label, y_label=y_label)
                    builder.set_scales(log_x=log_x, log_y=log_y)
                    
                    # Set axis limits (zoom) if specified
                    x_lim = None
                    y_lim = None
                    
                    if x_min or x_max:
                        try:
                            x_min_val = float(x_min) if x_min else None
                            x_max_val = float(x_max) if x_max else None
                            if x_min_val is not None or x_max_val is not None:
                                x_lim = (x_min_val, x_max_val)
                        except ValueError:
                            st.warning("⚠️ Invalid X-axis limits. Using auto range.")
                    
                    if y_min or y_max:
                        try:
                            y_min_val = float(y_min) if y_min else None
                            y_max_val = float(y_max) if y_max else None
                            if y_min_val is not None or y_max_val is not None:
                                y_lim = (y_min_val, y_max_val)
                        except ValueError:
                            st.warning("⚠️ Invalid Y-axis limits. Using auto range.")
                    
                    if x_lim is not None or y_lim is not None:
                        builder.set_limits(x_lim=x_lim, y_lim=y_lim)
                    
                    # Ensure legend_fontsize has a safe default if legend is hidden
                    legend_fontsize_safe = st.session_state.get('endf_legend_fontsize', default_legend_fontsize)
                    builder.set_font_sizes(
                        title=title_fontsize,
                        labels=label_fontsize,
                        legend=legend_fontsize_safe
                    )
                    
                    if show_grid:
                        builder.set_grid(show_grid)
                    
                    if show_legend:
                        builder.set_legend(loc=legend_location)
                    
                    # Build plot
                    fig = builder.build()
                    
                    # Apply grid settings after building
                    ax = fig.axes[0]
                    if show_grid:
                        ax.grid(True, which='major', alpha=grid_alpha)
                        if show_minor_grid:
                            ax.grid(True, which='minor', alpha=grid_alpha*0.5)
                    else:
                        ax.grid(False)
                    
                    # Apply tick parameters
                    if not log_x:
                        from matplotlib.ticker import MaxNLocator
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks_x))
                    
                    if not log_y:
                        from matplotlib.ticker import MaxNLocator
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks_y))
                    
                    ax.tick_params(axis='x', labelsize=tick_fontsize_x)
                    ax.tick_params(axis='y', labelsize=tick_fontsize_y)
                    
                    if rotate_x > 0:
                        plt.setp(ax.get_xticklabels(), rotation=rotate_x, ha='right')
                    if rotate_y > 0:
                        plt.setp(ax.get_yticklabels(), rotation=rotate_y, ha='right')
                    
                    fig.tight_layout()
                    
                    # Display plot
                    st.pyplot(fig, dpi=default_display_dpi)
                    
                    if errors:
                        with st.expander(f"⚠️ {len(errors)} series had errors"):
                            for error in errors:
                                st.write(f"• {error}")
                    
                    # Export Settings Section
                    st.markdown("---")
                    
                    with st.expander("💾 Export Settings", expanded=False):
                        # Figure Size
                        st.markdown("**📐 Figure Size**")
                        col_size1, col_size2 = st.columns(2)
                        with col_size1:
                            new_fig_width = st.number_input("Width (inches)", min_value=6, max_value=20,
                                                       value=st.session_state.endf_fig_width, step=1, key="endf_fig_width_right")
                            if new_fig_width != st.session_state.endf_fig_width:
                                st.session_state.endf_fig_width = new_fig_width
                                st.rerun()
                        with col_size2:
                            new_fig_height = st.number_input("Height (inches)", min_value=4, max_value=16,
                                                        value=st.session_state.endf_fig_height, step=1, key="endf_fig_height_right")
                            if new_fig_height != st.session_state.endf_fig_height:
                                st.session_state.endf_fig_height = new_fig_height
                                st.rerun()
                        
                        st.markdown("")
                        
                        # Export Resolution and Format
                        col_exp1, col_exp2 = st.columns(2)
                        
                        with col_exp1:
                            st.markdown("**📸 Resolution**")
                            export_dpi = st.selectbox(
                                "DPI",
                                [72, 100, 150, 200, 300, 600],
                                index=4,
                                help="Higher DPI = better quality but larger file size",
                                key="endf_export_dpi"
                            )
                            st.caption(f"Output: {int(new_fig_width * export_dpi)} × {int(new_fig_height * export_dpi)} px")
                        
                        with col_exp2:
                            st.markdown("**📄 Format**")
                            export_format = st.selectbox(
                                "File format",
                                ["png", "pdf", "svg", "eps"],
                                index=0,
                                help="PNG: raster. PDF/SVG/EPS: vector, scalable",
                                key="endf_export_format"
                            )
                            if export_format in ["pdf", "svg", "eps"]:
                                st.caption("✓ Vector - scalable")
                            else:
                                st.caption("✓ Raster - web ready")
                    
                    # Download button (outside expander)
                    st.markdown("")
                    buf = io.BytesIO()
                    fig.savefig(buf, format=export_format, dpi=export_dpi, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label=f"💾 Download Plot ({export_format.upper()}, {export_dpi} DPI)",
                        data=buf,
                        file_name=f"kika_endf_plot.{export_format}",
                        mime=f"image/{export_format}",
                        width="stretch",
                        type="primary"
                    )
                        
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
                with st.expander("Show error details"):
                    st.exception(e)

    # Generate plot and save plot buttons (INSIDE tab_viewer)
    st.markdown("---")

    # Initialize save dialog state
    if 'endf_show_save_dialog' not in st.session_state:
        st.session_state.endf_show_save_dialog = False
    if 'endf_show_overwrite_dialog' not in st.session_state:
        st.session_state.endf_show_overwrite_dialog = False
    if 'endf_pending_config_name' not in st.session_state:
        st.session_state.endf_pending_config_name = None

    # Check if we're working on an existing saved plot
    current_config_id = st.session_state.get('endf_current_config_id', None)
    is_updating = current_config_id is not None

    # Show overwrite confirmation dialog if active
    if st.session_state.endf_show_overwrite_dialog:
        st.warning(f"⚠️ A plot named '{st.session_state.endf_pending_config_name}' already exists.")
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            if st.button("✓ Overwrite", key="confirm_overwrite_endf", width="stretch", type="primary"):
                from utils.config_history import get_config_by_name
                existing_config = get_config_by_name(st.session_state.endf_pending_config_name)
                if existing_config:
                    save_configuration('endf', st.session_state.endf_pending_config_name, config_id=existing_config['id'])
                    st.session_state.endf_current_config_id = existing_config['id']
                st.session_state.endf_show_overwrite_dialog = False
                st.session_state.endf_show_save_dialog = False
                st.session_state.endf_pending_config_name = None
                st.success(f"✓ Plot '{st.session_state.endf_pending_config_name}' updated!")
                st.rerun()
        with col2:
            if st.button("✏️ Rename", key="rename_endf", width="stretch"):
                st.session_state.endf_show_overwrite_dialog = False
                # Keep save dialog open
                st.rerun()
        with col3:
            if st.button("Cancel", key="cancel_overwrite_endf", width="stretch"):
                st.session_state.endf_show_overwrite_dialog = False
                st.session_state.endf_show_save_dialog = False
                st.session_state.endf_pending_config_name = None
                st.rerun()
        st.markdown("---")

    # Show save dialog if active
    elif st.session_state.endf_show_save_dialog:
        with st.form(key="save_endf_config_form"):
            st.markdown("#### 💾 Save Plot")
            
            # Get default name (auto-generated)
            from utils.config_history import save_configuration
            file_names = list(st.session_state.get('endf_objects', {}).keys())
            series_count = len(st.session_state.endf_plot_selections)
            
            # Create preview of auto-generated name
            if file_names:
                first_file = file_names[0]
                if len(first_file) > 20:
                    first_file = first_file[:17] + "..."
                auto_name = f"ENDF - {first_file} ({series_count} series)"
            else:
                timestamp_str = datetime.now().strftime("%b%d %H:%M")
                auto_name = f"ENDF Config - {timestamp_str}"
            
            config_name = st.text_input(
                "Plot Name",
                value="",
                placeholder=auto_name,
                help="Leave empty for auto-generated name"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("💾 Save", width="stretch", type="primary")
            with col2:
                cancelled = st.form_submit_button("Cancel", width="stretch")
            
            if submitted:
                from utils.config_history import get_config_by_name
                final_name = config_name.strip() if config_name.strip() else auto_name
                
                # Check if name already exists
                existing_config = get_config_by_name(final_name)
                if existing_config:
                    # Show overwrite dialog
                    st.session_state.endf_pending_config_name = final_name
                    st.session_state.endf_show_overwrite_dialog = True
                    st.rerun()
                else:
                    # Save new plot
                    saved_config = save_configuration('endf', final_name)
                    st.session_state.endf_current_config_id = saved_config['id']
                    st.session_state.endf_show_save_dialog = False
                    st.success(f"✓ Plot '{final_name}' saved!")
                    st.rerun()
            
            if cancelled:
                st.session_state.endf_show_save_dialog = False
                st.rerun()
        st.markdown("---")

    # Action buttons
    if is_updating:
        # Show Generate, Update, and Save As buttons when working on existing saved plot
        col_gen, col_update, col_save = st.columns([3, 2, 2])
        with col_gen:
            generate_clicked = st.button("🎨 Generate Plot", key="generate_endf_plot_btn", type="primary", width="stretch")
        with col_update:
            update_clicked = st.button("🔄 Update Plot", key="update_endf_config_btn", width="stretch", 
                                    help="Update the current saved plot")
        with col_save:
            save_clicked = st.button("💾 Save As New", key="save_endf_config_btn", width="stretch", 
                                    help="Save as a new plot")
        
        # Handle update button click
        if update_clicked:
            if not st.session_state.endf_plot_selections:
                st.warning("⚠️ Please add at least one data series before updating")
            else:
                from utils.config_history import get_saved_configurations
                # Find current saved plot to get its name
                configs = get_saved_configurations()
                current_config = next((c for c in configs if c['id'] == current_config_id), None)
                if current_config:
                    save_configuration('endf', current_config['name'], config_id=current_config_id)
                    st.success(f"✓ Plot '{current_config['name']}' updated!")
                    st.rerun()
    else:
        # Show Generate and Save buttons for new plots
        col_gen, col_save = st.columns([3, 2])
        with col_gen:
            generate_clicked = st.button("🎨 Generate Plot", key="generate_endf_plot_btn", type="primary", width="stretch")
        with col_save:
            save_clicked = st.button("💾 Save Plot", key="save_endf_config_btn", width="stretch", 
                                    help="Save current plot for later restoration")

    # Handle save button click (both Save and Save As)
    if 'save_clicked' in locals() and save_clicked:
        if not st.session_state.endf_plot_selections:
            st.warning("⚠️ Please add at least one data series before saving")
        else:
            st.session_state.endf_show_save_dialog = True
            st.rerun()

    if generate_clicked:
        if not st.session_state.endf_plot_selections:
            st.error("Please add at least one data series to plot!")
        else:
            st.success("✓ Plot ready! Scroll up to view and download your plot.")

# Footer
st.markdown("---")
try:
    import kika
    kika_version = getattr(kika, "__version__", "unknown")
except Exception:
    kika_version = 'unknown'

# Footer (always render)
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
<p>ENDF Viewer • Powered by KIKA v{kika_version} & PlotBuilder</p>
<p style='font-size: 0.8em;'>💡 Tip: If data doesn't load correctly, click the 🔄 button in the sidebar to clear cache</p>
</div>
""", unsafe_allow_html=True)
