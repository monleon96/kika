"""
ACE Data Viewer Page

Upload and visualize ACE format nuclear data files
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import io
from datetime import datetime
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import kika
from kika.plotting import PlotBuilder
import matplotlib.pyplot as plt
import numpy as np

# Import config history utilities
from components.saved_configs import render_saved_configs_sidebar
from components.clear_cache import render_clear_cache_button
from utils.config_history import save_configuration
from utils.backend_auth import handle_verification_query, require_user, render_account_sidebar

# Page config
st.set_page_config(page_title="ACE Viewer - KIKA", page_icon="📊", layout="wide")

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
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("📊 ACE Data Viewer")
st.markdown("Upload and visualize ACE format nuclear data files")
st.markdown("---")

# Mark current page for navigation
st.session_state.current_page = "pages/1_📊_ACE_Viewer.py"

# Sidebar - Global file management
from components.file_sidebar import render_file_upload_sidebar

render_file_upload_sidebar()

# Get globally loaded ACE files
from utils.file_manager import get_ace_files

st.session_state.ace_objects = {name: data['object'] for name, data in get_ace_files().items()}

# Render saved configurations and the shared clear-cache button at the end of the sidebar
with st.sidebar:
    st.markdown("---")
    render_saved_configs_sidebar()
    # Render clear-cache button as the last sidebar item
    render_clear_cache_button(key_prefix="clear_cache_ace")

# Initialize session state for plot data selections
if 'plot_selections' not in st.session_state:
    st.session_state.plot_selections = []

# Initialize unique ID counter for selections
if 'selection_id_counter' not in st.session_state:
    st.session_state.selection_id_counter = 0

# Main content with tabs
tab_home, tab_viewer = st.tabs(["Home", "Viewer"])

# ============================================================================
# TAB 1: Home
# ============================================================================
with tab_home:
    st.header("📖 About ACE Viewer")
    
    # Quick stats
    n_files = len(st.session_state.ace_objects)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is ACE?
        
        ACE (A Compact ENDF) is a binary format used by Monte Carlo transport 
        codes like MCNP to store nuclear data. It contains cross sections, 
        angular distributions, and other reaction data in a compact, 
        fast-to-access format.
        
        ### Supported Data Types:
        - **Cross Sections**: Reaction probabilities vs. energy (barns)
        - **Angular Distributions**: Scattering angle probabilities
        
        ### Features:
        - **Multi-file comparison** - overlay data from different libraries
        - **Interactive configuration** - customize every aspect of the plot
        - **High-quality export** - save publication-ready figures
        - **Energy interpolation** - evaluate angular distributions at any energy
        """)
    
    with col2:
        st.markdown("""
        ### Current Status
        """)
        st.metric("Loaded ACE Files", n_files)
        
        if n_files > 0:
            st.markdown("**Loaded files:**")
            for name in st.session_state.ace_objects.keys():
                st.write(f"- `{name}`")
        else:
            st.info("👈 Upload ACE files from the sidebar to get started")
        
        st.markdown("""
        ### Getting Started
        1. Upload ACE files using the sidebar uploader
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

# ============================================================================
# TAB 2: Viewer
# ============================================================================
with tab_viewer:
    # Always show plotting interface, with informational message if no files
    if not st.session_state.ace_objects:
        st.info("👈 Upload ACE files from the sidebar to get started. The plotting interface will be ready once files are loaded.")

    # Always show plotting interface (whether or not files are loaded)
    # Function to reset figure settings to defaults based on current data type
    def reset_figure_settings_to_defaults():
        """Reset figure settings to default values based on current data type"""
        # Get current data type from session state
        current_data_type = st.session_state.get('data_type_select', 'Cross Section')
        
        # Define defaults based on data type
        default_fig_width = 10
        default_fig_height = 6
        default_title_fontsize = 14
        default_label_fontsize = 12
        default_tick_fontsize = 10
        default_legend_fontsize = 10
        
        if current_data_type == "Cross Section":
            default_title = "Cross Section Comparison"
            default_x_label = "Energy (MeV)"
            default_y_label = "Cross Section (barns)"
            default_log_x = True
            default_log_y = True
        else:  # Angular Distribution
            default_title = "Angular Distribution Comparison"
            default_x_label = "cos(θ)"
            default_y_label = "Probability Density"
            default_log_x = False
            default_log_y = False
        
        # Set all figure settings to defaults in session state
        st.session_state.plot_title = default_title
        st.session_state.x_label = default_x_label
        st.session_state.y_label = default_y_label
        st.session_state.fig_width = default_fig_width
        st.session_state.fig_height = default_fig_height
        st.session_state.title_fontsize = default_title_fontsize
        st.session_state.label_fontsize = default_label_fontsize
        st.session_state.legend_fontsize = default_legend_fontsize
        st.session_state.show_legend = True
        st.session_state.legend_loc = "best"
        st.session_state.log_x = default_log_x
        st.session_state.log_y = default_log_y
        st.session_state.show_grid = True
        st.session_state.grid_alpha = 0.3
        st.session_state.show_minor_grid = False
        st.session_state.minor_grid_alpha = 0.15
        st.session_state.tick_fontsize_x = default_tick_fontsize
        st.session_state.tick_fontsize_y = default_tick_fontsize
        st.session_state.max_ticks_x = 10
        st.session_state.max_ticks_y = 10
        st.session_state.rotate_x = 0
        st.session_state.rotate_y = 0
        st.session_state.x_min = ""
        st.session_state.x_max = ""
        st.session_state.y_min = ""
        st.session_state.y_max = ""
    
    # Function to reset all plot settings to defaults
    def reset_all_plot_settings():
        """Reset all plot settings to default values"""
        # Clear series
        st.session_state.plot_selections = []
        st.session_state.selection_id_counter = 0
        
        # Reset figure settings
        reset_figure_settings_to_defaults()
        
        # Also clear current config tracking
        if 'ace_current_config_id' in st.session_state:
            del st.session_state['ace_current_config_id']
    
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
        """Reset plot selections, ID counter, and figure settings when data type changes"""
        st.session_state.plot_selections = []
        st.session_state.selection_id_counter = 0
        # Reset figure settings to match new data type
        reset_figure_settings_to_defaults()
    
    # Data type selection
    data_type = st.selectbox(
        "Select Data Type",
        ["Cross Section", "Angular Distribution"],
        help="Choose what type of data to visualize",
        key="data_type_select",
        on_change=reset_on_data_type_change
    )
    
    st.markdown("---")
    
    # Create two-column layout: Settings (left) and Plot (right)
    col_settings, col_plot = st.columns([2, 3], gap="medium")
    
    # ========== LEFT COLUMN: Settings ==========
    with col_settings:
        st.markdown("### 📊 Plot Series Settings")
        st.markdown("Each card represents one data series. Configure data source and styling for each series individually.")
        
        # Series management buttons
        col_add, col_clear = st.columns([3, 1])
        
        with col_add:
            if st.button("➕ Add New Series", key="add_new_series_btn", type="primary"):
                # Add a new empty series
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                # Get default file and generate proper label from API
                default_file = list(st.session_state.ace_objects.keys())[0] if st.session_state.ace_objects else None
                default_label = f"Series {len(st.session_state.plot_selections) + 1}"
                default_mt = None
                default_energy = 1.0
                
                # Try to get a better default label from the ACE object
                if default_file:
                    ace_obj = st.session_state.ace_objects[default_file]
                    if data_type == "Cross Section":
                        available_mts = ace_obj.mt_numbers if hasattr(ace_obj, 'mt_numbers') else []
                        if available_mts:
                            default_mt = 2 if 2 in available_mts else available_mts[0]
                            try:
                                plot_data = ace_obj.to_plot_data('xs', mt=default_mt)
                                default_label = plot_data.label if plot_data.label else default_label
                            except:
                                pass
                    else:  # Angular Distribution
                        if hasattr(ace_obj, 'angular_distributions') and ace_obj.angular_distributions:
                            available_mts = ace_obj.angular_distributions.get_neutron_reaction_mt_numbers()
                            if ace_obj.angular_distributions.has_elastic_data and 2 not in available_mts:
                                available_mts = [2] + available_mts
                            if available_mts:
                                default_mt = 2 if 2 in available_mts else available_mts[0]
                                try:
                                    plot_data = ace_obj.to_plot_data('angular', mt=default_mt, energy=default_energy)
                                    default_label = plot_data.label if plot_data.label else default_label
                                except:
                                    pass
                
                new_series = {
                    'id': st.session_state.selection_id_counter,
                    'file': default_file,
                    'mt': default_mt,
                    'label': default_label,
                    'type': 'xs' if data_type == "Cross Section" else 'angular',
                    'energy': default_energy if data_type == "Angular Distribution" else None,
                    'color': colors[len(st.session_state.plot_selections) % len(colors)],
                    'linewidth': 2.0,
                    'linestyle': '-',
                    'marker': None,
                    'markersize': 6
                }
                st.session_state.plot_selections.append(new_series)
                st.session_state.selection_id_counter += 1
                st.rerun()
        
        with col_clear:
            if st.button("🗑️ Clear All", key="clear_all_selections", width="stretch"):
                st.session_state.plot_selections = []
                st.rerun()
        
        
        # Display series as cards
        if len(st.session_state.plot_selections) == 0:
            st.info("👆 Click 'Add New Series' to start building your plot")
        else:
            # Show as expandable cards
            for idx, selection in enumerate(st.session_state.plot_selections):
                sel_id = selection['id']
                
                # Create title for expander with more info
                mt_str = f"MT={selection['mt']}" if selection['mt'] is not None else "MT=?"
                label_str = selection.get('label', 'Untitled')
                if selection['type'] == 'xs':
                    title = f"📊 {label_str} | {selection.get('file', 'No file')} ({mt_str})"
                else:
                    # Format energy with max 3 decimal places, removing trailing zeros
                    energy_val = selection.get('energy', 1.0)
                    energy_str = f"E={energy_val:.3f}".rstrip('0').rstrip('.') + " MeV"
                    title = f"📊 {label_str} | {selection.get('file', 'No file')} ({mt_str}, {energy_str})"
                
                with st.expander(title, expanded=True):
                    # Data Source Settings with inline Remove button
                    col_header, col_del = st.columns([5, 1])
                    with col_header:
                        st.markdown("**📂 Data Source**")
                    with col_del:
                        if st.button("🗑️", key=f"remove_plot_series_{sel_id}", help="Remove this series"):
                            st.session_state.plot_selections.pop(idx)
                            st.rerun()
                    
                    col_data1, col_data2 = st.columns(2)
                    
                    with col_data1:
                        # File selection
                        file_list = list(st.session_state.ace_objects.keys())
                        current_file = selection.get('file')
                        if current_file and current_file in file_list:
                            file_index = file_list.index(current_file)
                        else:
                            file_index = 0
                        
                        new_file = st.selectbox(
                            "ACE File",
                            file_list,
                            index=file_index,
                            key=f"file_select_{sel_id}"
                        )
                        st.session_state.plot_selections[idx]['file'] = new_file
                        
                        # Get available MTs based on selected file and data type
                        if new_file:
                            ace_obj = st.session_state.ace_objects[new_file]
                            
                            if data_type == "Cross Section":
                                available_mts = ace_obj.mt_numbers if hasattr(ace_obj, 'mt_numbers') else []
                            else:  # Angular Distribution
                                if hasattr(ace_obj, 'angular_distributions') and ace_obj.angular_distributions:
                                    available_mts = ace_obj.angular_distributions.get_neutron_reaction_mt_numbers()
                                    if ace_obj.angular_distributions.has_elastic_data and 2 not in available_mts:
                                        available_mts = [2] + available_mts
                                else:
                                    available_mts = []
                            
                            if available_mts:
                                st.caption(f"📋 Available MTs: {', '.join(map(str, available_mts[:20]))}" + 
                                         ("..." if len(available_mts) > 20 else ""))
                    
                    with col_data2:
                        # MT selection
                        if new_file and available_mts:
                            current_mt = selection.get('mt')
                            if current_mt in available_mts:
                                mt_index = available_mts.index(current_mt)
                            else:
                                mt_index = 0
                            
                            new_mt = st.selectbox(
                                "MT Number",
                                available_mts,
                                index=mt_index,
                                key=f"mt_select_{sel_id}"
                            )
                            
                            # Update MT and auto-update label if it changed
                            if new_mt != current_mt:
                                st.session_state.plot_selections[idx]['mt'] = new_mt
                                # Try to get label from API
                                try:
                                    if data_type == "Cross Section":
                                        plot_data = ace_obj.to_plot_data('xs', mt=new_mt)
                                    else:
                                        energy = selection.get('energy', 1.0)
                                        plot_data = ace_obj.to_plot_data('angular', mt=new_mt, energy=energy)
                                    if plot_data.label:
                                        st.session_state.plot_selections[idx]['label'] = plot_data.label
                                except:
                                    pass
                            else:
                                st.session_state.plot_selections[idx]['mt'] = new_mt
                        else:
                            st.warning("No MT numbers available")
                            st.session_state.plot_selections[idx]['mt'] = None
                        
                        # Energy for Angular Distribution
                        if data_type == "Angular Distribution":
                            current_energy = selection.get('energy', 1.0)
                            # Round to 3 decimal places max, removing trailing zeros
                            rounded_energy = round(float(current_energy), 3)
                            new_energy = st.number_input(
                                "Energy (MeV)",
                                min_value=0.0,
                                max_value=20.0,
                                value=rounded_energy,
                                step=0.1,
                                format="%.3f",
                                key=f"energy_select_{sel_id}",
                                help="Energy at which to evaluate the angular distribution. "
                                     "Values between tabulated energies are interpolated linearly. "
                                     "Values outside the data range return isotropic distribution."
                            )
                            # Store rounded value
                            st.session_state.plot_selections[idx]['energy'] = round(new_energy, 3)
                    
                    # Label with Automatic/Custom options
                    st.markdown("**Series Label**")
                    label_mode = st.radio(
                        "Label mode",
                        ["Automatic", "Custom"],
                        index=0 if selection.get('label_mode', 'auto') == 'auto' else 1,
                        key=f"label_mode_{sel_id}",
                        horizontal=True,
                        help="Automatic updates label based on file/MT/energy, Custom allows manual entry"
                    )
                    
                    if label_mode == "Automatic":
                        st.session_state.plot_selections[idx]['label_mode'] = 'auto'
                        # Generate automatic label
                        try:
                            if new_file and new_mt is not None:
                                if data_type == "Cross Section":
                                    plot_data = ace_obj.to_plot_data('xs', mt=new_mt)
                                else:
                                    energy = selection.get('energy', 1.0)
                                    plot_data = ace_obj.to_plot_data('angular', mt=new_mt, energy=energy)
                                auto_label = plot_data.label if plot_data.label else f"Series {idx+1}"
                            else:
                                auto_label = f"Series {idx+1}"
                        except:
                            auto_label = f"Series {idx+1}"
                        
                        st.session_state.plot_selections[idx]['label'] = auto_label
                        st.info(f"Label: {auto_label}")
                    else:
                        st.session_state.plot_selections[idx]['label_mode'] = 'custom'
                        custom_label = st.text_input(
                            "Custom label",
                            value=selection.get('label', f"Series {idx+1}"),
                            key=f"label_custom_{sel_id}"
                        )
                        st.session_state.plot_selections[idx]['label'] = custom_label
                    
                    st.markdown("---")
                    
                    with st.expander("🎨 Styling Options", expanded=False):
                        # Color and line styling - reorganized layout (left: colors, right: line properties)
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
                                    st.markdown(
                                        f'<div style="width: 100%; height: 20px; background: {hex_color}; '
                                        f'border: 2px solid {"#333" if hex_color == current_color and not use_custom else "#ddd"}; '
                                        f'border-radius: 4px; margin-bottom: 4px;"></div>',
                                        unsafe_allow_html=True
                                    )
                                    # Button below the colored box
                                    if st.button("●", key=f"color_btn_{sel_id}_{i}", help=label, width="stretch"):
                                        selected_color = hex_color
                            
                            # Custom color option in new row
                            st.markdown("")
                            custom_col1, custom_col2 = st.columns([1, 5])
                            with custom_col1:
                                if st.button("🎨", key=f"color_custom_btn_{sel_id}", help="Custom color", width="stretch"):
                                    st.session_state.plot_selections[idx]['use_custom_color'] = True
                                    st.rerun()
                            
                            with custom_col2:
                                if use_custom or st.session_state.plot_selections[idx].get('use_custom_color', False):
                                    new_color = st.color_picker(
                                        "Custom",
                                        value=current_color,
                                        key=f"color_picker_{sel_id}",
                                        label_visibility="collapsed"
                                    )
                                    st.session_state.plot_selections[idx]['color'] = new_color
                                    st.session_state.plot_selections[idx]['use_custom_color'] = True
                            
                            # Update color if a preset was clicked
                            if selected_color:
                                st.session_state.plot_selections[idx]['color'] = selected_color
                                st.session_state.plot_selections[idx]['use_custom_color'] = False
                                st.rerun()

                            # Live preview - short and thick bar
                            preview_hex = st.session_state.plot_selections[idx]['color']
                            st.markdown(
                                f"<div style='height: 12px; border-radius: 4px; "
                                f"background: {preview_hex}; margin-top: 8px;'></div>",
                                unsafe_allow_html=True
                            )
                            # --- end SINGLE-SET COLOR PALETTE ---
                        
                        with col_right:
                            # Line properties
                            new_linewidth = st.number_input(
                                "Line Width",
                                min_value=0.5,
                                max_value=5.0,
                                value=selection.get('linewidth', 2.0),
                                step=0.5,
                                key=f"linewidth_edit_{sel_id}"
                            )
                            st.session_state.plot_selections[idx]['linewidth'] = new_linewidth
                            
                            new_linestyle = st.selectbox(
                                "Line Style",
                                ["-", "--", "-.", ":"],
                                index=["-", "--", "-.", ":"].index(selection.get('linestyle', '-')),
                                format_func=lambda x: {"-": "Solid", "--": "Dashed", "-.": "Dash-dot", ":": "Dotted"}[x],
                                key=f"linestyle_edit_{sel_id}"
                            )
                            st.session_state.plot_selections[idx]['linestyle'] = new_linestyle
                            
                            # Marker options
                            has_marker = selection.get('marker') is not None
                            new_use_markers = st.checkbox(
                                "Show markers",
                                value=has_marker,
                                key=f"use_markers_edit_{sel_id}"
                            )
                            
                            if new_use_markers:
                                marker_options = ["o", "s", "^", "v", "D", "*", "x", "+"]
                                current_marker = selection.get('marker', 'o')
                                if current_marker not in marker_options:
                                    current_marker = 'o'
                                
                                col_marker1, col_marker2 = st.columns(2)
                                
                                with col_marker1:
                                    new_marker = st.selectbox(
                                        "Marker Style",
                                        marker_options,
                                        index=marker_options.index(current_marker),
                                        format_func=lambda x: {"o": "Circle", "s": "Square", "^": "Triangle up", 
                                                              "v": "Triangle down", "D": "Diamond", "*": "Star",
                                                              "x": "X", "+": "Plus"}[x],
                                        key=f"marker_edit_{sel_id}"
                                    )
                                    st.session_state.plot_selections[idx]['marker'] = new_marker
                                
                                with col_marker2:
                                    new_markersize = st.number_input(
                                        "Marker Size",
                                        min_value=2,
                                        max_value=15,
                                        value=selection.get('markersize', 6),
                                        step=1,
                                        key=f"markersize_edit_{sel_id}"
                                    )
                                    st.session_state.plot_selections[idx]['markersize'] = new_markersize
                            else:
                                st.session_state.plot_selections[idx]['marker'] = None
        
        st.markdown("---")
        
        # Figure Configuration section
        st.markdown("### 🎨 Figure Configuration")
        
        # Define defaults based on data type and PlotBuilder defaults
        default_fig_width = 10
        default_fig_height = 6
        default_title_fontsize = 14
        default_label_fontsize = 12
        default_tick_fontsize = 10
        default_legend_fontsize = 10
        
        if data_type == "Cross Section":
            default_title = "Cross Section Comparison"
            default_x_label = "Energy (MeV)"
            default_y_label = "Cross Section (barns)"
            default_log_x = True
            default_log_y = True
        else:  # Angular Distribution
            default_title = "Angular Distribution Comparison"
            default_x_label = "cos(θ)"
            default_y_label = "Probability Density"
            default_log_x = False
            default_log_y = False
        
        # Initialize figure size in session state
        if 'fig_width' not in st.session_state:
            st.session_state.fig_width = default_fig_width
        if 'fig_height' not in st.session_state:
            st.session_state.fig_height = default_fig_height
        fig_width = st.session_state.fig_width
        fig_height = st.session_state.fig_height
        
        # Labels, Title & Legend (expandable)
        with st.expander("🏷️ Labels, Title & Legend", expanded=True):
            # Title
            st.markdown("**Title**")
            col_title, col_title_font = st.columns([3, 1])
            with col_title:
                plot_title = st.text_input(
                    "Label",
                    value=default_title,
                    key="plot_title"
                )
            with col_title_font:
                title_fontsize = st.number_input("Size", min_value=8, max_value=32, 
                                                value=default_title_fontsize, step=1, key="title_fontsize")
            
            st.markdown("")  # Spacing
            
            # Axis labels
            st.markdown("**Axis Labels**")
            col_labels1, col_labels2 = st.columns(2)
            with col_labels1:
                col_xlabel, col_xlabel_font = st.columns([3, 1])
                with col_xlabel:
                    x_label = st.text_input("X-axis", value=default_x_label, key="x_label")
                with col_xlabel_font:
                    label_fontsize = st.number_input("Size", min_value=8, max_value=24, 
                                                    value=default_label_fontsize, step=1, key="label_fontsize")
            
            with col_labels2:
                y_label = st.text_input("Y-axis", value=default_y_label, key="y_label")
            
            st.markdown("")  # Spacing
            
            # Legend
            st.markdown("**Legend**")
            col_leg1, col_leg2, col_leg3 = st.columns([1, 2, 1])
            
            with col_leg1:
                show_legend = st.checkbox("Show legend", value=True, key="show_legend")
            
            with col_leg2:
                if show_legend:
                    legend_options = ["best", "upper right", "upper left", "lower right", "lower left", "center", "center right"]
                    legend_location = st.selectbox(
                        "Position",
                        legend_options,
                        index=0,
                        key="legend_loc"
                    )
            
            with col_leg3:
                if show_legend:
                    legend_fontsize = st.number_input("Size", min_value=6, max_value=20, 
                                                     value=default_legend_fontsize, step=1, key="legend_fontsize")
        
        # Zoom, Scales & Grid (expandable)
        with st.expander("🔍 Zoom, Scales & Grid", expanded=True):
            # Zoom / Axis Limits (FIRST)
            st.markdown("**🔍 Zoom / Axis Limits**")
            st.info("💡 Leave empty to show full range automatically")
            
            col_zoom1, col_zoom2 = st.columns(2)
            
            with col_zoom1:
                st.markdown("**X-axis Range**")
                col_xmin, col_xmax = st.columns(2)
                with col_xmin:
                    x_min = st.text_input("Min", value="", key="x_min", 
                                         placeholder="auto", help="Minimum X value (leave empty for auto)")
                with col_xmax:
                    x_max = st.text_input("Max", value="", key="x_max", 
                                         placeholder="auto", help="Maximum X value (leave empty for auto)")
            
            with col_zoom2:
                st.markdown("**Y-axis Range**")
                col_ymin, col_ymax = st.columns(2)
                with col_ymin:
                    y_min = st.text_input("Min", value="", key="y_min", 
                                         placeholder="auto", help="Minimum Y value (leave empty for auto)")
                with col_ymax:
                    y_max = st.text_input("Max", value="", key="y_max", 
                                         placeholder="auto", help="Maximum Y value (leave empty for auto)")
            
            st.markdown("---")
            
            # Scales and Grid (AFTER Zoom)
            col_scale1, col_scale2 = st.columns(2)
            
            with col_scale1:
                st.markdown("**Scales**")
                log_x = st.checkbox("Logarithmic X-axis", value=default_log_x, key="log_x")
                log_y = st.checkbox("Logarithmic Y-axis", value=default_log_y, key="log_y")
            
            with col_scale2:
                st.markdown("**Grid**")
                show_grid = st.checkbox("Show grid", value=True, key="show_grid")
                if show_grid:
                    grid_alpha = st.slider("Grid transparency", 0.0, 1.0, 
                                          0.3, 0.1, key="grid_alpha")
                    show_minor_grid = st.checkbox("Show minor grid", value=False, key="show_minor_grid")
                    if show_minor_grid:
                        minor_grid_alpha = st.slider("Minor grid transparency", 0.0, 1.0, 
                                                    0.15, 0.05, key="minor_grid_alpha")
        
        # Advanced: Tick parameters (expandable, collapsed by default)
        with st.expander("⚙️ Advanced: Tick Parameters"):
            col_tick1, col_tick2 = st.columns(2)
            
            with col_tick1:
                st.markdown("**X-axis Ticks**")
                tick_fontsize_x = st.number_input("Tick font size", min_value=6, max_value=20, 
                                                 value=default_tick_fontsize, step=1, key="tick_fontsize_x")
                max_ticks_x = st.number_input("Max ticks", min_value=3, max_value=20, 
                                             value=10, step=1, key="max_ticks_x")
                rotate_x = st.number_input("Rotate labels (degrees)", min_value=0, max_value=90, 
                                          value=0, step=15, key="rotate_x")
            
            with col_tick2:
                st.markdown("**Y-axis Ticks**")
                tick_fontsize_y = st.number_input("Tick font size", min_value=6, max_value=20, 
                                                 value=default_tick_fontsize, step=1, key="tick_fontsize_y")
                max_ticks_y = st.number_input("Max ticks", min_value=3, max_value=20, 
                                             value=10, step=1, key="max_ticks_y")
                rotate_y = st.number_input("Rotate labels (degrees)", min_value=0, max_value=90, 
                                          value=0, step=15, key="rotate_y")
    
    # ========== RIGHT COLUMN: Plot Preview ==========
    with col_plot:
        st.markdown("### 📊 Plot Preview")
        
        # Default DPI for plot display
        default_display_dpi = 100
        
        # Check if we should generate plot
        if not st.session_state.plot_selections:
            st.info("👈 Add data series from the left panel to see the plot here")
        else:
            try:
                # Create PlotBuilder with figure size
                builder = PlotBuilder(figsize=(fig_width, fig_height))
                
                # Add data for each selection
                errors = []
                for idx, selection in enumerate(st.session_state.plot_selections):
                    file_name = selection['file']
                    mt = selection['mt']
                    label = selection['label']
                    plot_type = selection['type']
                    
                    ace_obj = st.session_state.ace_objects[file_name]
                    
                    try:
                        if plot_type == 'xs':
                            plot_data = ace_obj.to_plot_data('xs', mt=mt, label=label)
                        else:  # Angular Distribution
                            energy = selection['energy']
                            plot_data = ace_obj.to_plot_data('angular', mt=mt, energy=energy, label=label)
                        
                        # Apply per-series styling from selection
                        plot_data.color = selection.get('color', '#1f77b4')
                        plot_data.linewidth = selection.get('linewidth', 2.0)
                        plot_data.linestyle = selection.get('linestyle', '-')
                        
                        # Apply marker if specified
                        if selection.get('marker') is not None:
                            plot_data.marker = selection.get('marker')
                            plot_data.markersize = selection.get('markersize', 6)
                        else:
                            plot_data.marker = None
                        
                        builder.add_data(plot_data)
                        
                    except Exception as e:
                        error_msg = f"{label} (MT={mt}): {str(e)}"
                        errors.append(error_msg)
                        st.warning(f"⚠️ Skipping {label}: {str(e)}")
                
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
                    
                    # Parse X limits
                    if x_min or x_max:
                        try:
                            x_min_val = float(x_min) if x_min else None
                            x_max_val = float(x_max) if x_max else None
                            if x_min_val is not None or x_max_val is not None:
                                x_lim = (x_min_val, x_max_val)
                        except ValueError:
                            st.warning("⚠️ Invalid X-axis limits. Using auto range.")
                    
                    # Parse Y limits
                    if y_min or y_max:
                        try:
                            y_min_val = float(y_min) if y_min else None
                            y_max_val = float(y_max) if y_max else None
                            if y_min_val is not None or y_max_val is not None:
                                y_lim = (y_min_val, y_max_val)
                        except ValueError:
                            st.warning("⚠️ Invalid Y-axis limits. Using auto range.")
                    
                    # Apply limits if any are set
                    if x_lim is not None or y_lim is not None:
                        builder.set_limits(x_lim=x_lim, y_lim=y_lim)
                    
                    builder.set_font_sizes(
                        title=title_fontsize,
                        labels=label_fontsize,
                        legend=legend_fontsize
                    )
                    
                    if show_grid:
                        builder.set_grid(grid=True)
                    
                    if show_legend:
                        builder.set_legend(loc=legend_location)
                    
                    # Build plot
                    fig = builder.build()
                    
                    # Apply grid settings after building
                    ax = fig.axes[0]
                    if show_grid:
                        ax.grid(True, alpha=grid_alpha, which='major')
                        if show_minor_grid:
                            ax.minorticks_on()
                            ax.grid(True, alpha=minor_grid_alpha, which='minor', linestyle=':', linewidth=0.5)
                    else:
                        ax.grid(False)
                    
                    # Apply tick parameters after building
                    ax = fig.axes[0]
                    
                    # Set max number of ticks for each axis
                    if not log_x:
                        try:
                            ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks_x))
                        except Exception as e:
                            st.warning(f"Could not set X-axis tick parameters: {e}")
                    
                    if not log_y:
                        try:
                            ax.yaxis.set_major_locator(plt.MaxNLocator(max_ticks_y))
                        except Exception as e:
                            st.warning(f"Could not set Y-axis tick parameters: {e}")
                    
                    # Apply tick font sizes independently for each axis
                    ax.tick_params(axis='x', labelsize=tick_fontsize_x)
                    ax.tick_params(axis='y', labelsize=tick_fontsize_y)
                    
                    # Apply rotation
                    if rotate_x > 0:
                        ax.tick_params(axis='x', rotation=rotate_x)
                    if rotate_y > 0:
                        ax.tick_params(axis='y', rotation=rotate_y)
                    
                    # Ensure tight layout
                    fig.tight_layout()
                    
                    # Display plot
                    st.pyplot(fig, dpi=default_display_dpi)
                    
                    if errors:
                        with st.expander(f"⚠️ {len(errors)} series had errors"):
                            for error in errors:
                                st.write(f"• {error}")
                    
                    # Export settings below the plot
                    st.markdown("---")
                    
                    with st.expander("💾 Export Settings", expanded=False):
                        # Figure Size
                        st.markdown("**📐 Figure Size**")
                        col_size1, col_size2 = st.columns(2)
                        with col_size1:
                            new_fig_width = st.number_input("Width (inches)", min_value=6, max_value=20,
                                                       value=st.session_state.fig_width, step=1, key="fig_width_right")
                            if new_fig_width != st.session_state.fig_width:
                                st.session_state.fig_width = new_fig_width
                                st.rerun()
                        with col_size2:
                            new_fig_height = st.number_input("Height (inches)", min_value=4, max_value=16,
                                                        value=st.session_state.fig_height, step=1, key="fig_height_right")
                            if new_fig_height != st.session_state.fig_height:
                                st.session_state.fig_height = new_fig_height
                                st.rerun()
                        
                        st.markdown("")
                        
                        # Export Resolution and Format
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📸 Resolution**")
                            dpi = st.selectbox(
                                "DPI (dots per inch)",
                                [72, 100, 150, 200, 300, 600],
                                index=4,
                                help="Higher DPI = better quality but larger file size",
                                key="export_dpi"
                            )
                            
                            st.markdown(f"**Output resolution:** {int(fig_width * dpi)} × {int(fig_height * dpi)} pixels")
                        
                        with col2:
                            st.markdown("**📄 Format**")
                            export_format = st.selectbox(
                                "Export format",
                                ["png", "pdf", "svg", "eps"],
                                index=0,
                                help="PNG: raster, good for web. PDF/SVG/EPS: vector, scalable",
                                key="export_format"
                            )
                            
                            if export_format in ["pdf", "svg", "eps"]:
                                st.info("Vector formats are scalable and ideal for publications")
                            else:
                                st.info("PNG is ideal for web use and presentations")
                    
                    # Export button (outside expander)
                    st.markdown("")
                    buf = io.BytesIO()
                    fig.savefig(buf, format=export_format, dpi=dpi, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label=f"💾 Download as {export_format.upper()}",
                        data=buf,
                        file_name=f"kika_plot_{data_type.lower().replace(' ', '_')}.{export_format}",
                        mime=f"image/{export_format}",
                        width="stretch",
                        type="primary"
                    )
                        
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
                with st.expander("Show error details"):
                    st.exception(e)
    
    # Generate plot and save config buttons
    st.markdown("---")
    
    # Initialize save dialog state
    if 'show_save_dialog' not in st.session_state:
        st.session_state.show_save_dialog = False
    if 'show_overwrite_dialog' not in st.session_state:
        st.session_state.show_overwrite_dialog = False
    if 'pending_config_name' not in st.session_state:
        st.session_state.pending_config_name = None
    
    # Check if we're working on an existing config
    current_config_id = st.session_state.get('ace_current_config_id', None)
    is_updating = current_config_id is not None
    
    # Show overwrite confirmation dialog if active
    if st.session_state.show_overwrite_dialog:
        st.warning(f"⚠️ A plot named '{st.session_state.pending_config_name}' already exists.")
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            if st.button("✓ Overwrite", key="confirm_overwrite_ace", width="stretch", type="primary"):
                from utils.config_history import get_config_by_name
                existing_config = get_config_by_name(st.session_state.pending_config_name)
                if existing_config:
                    save_configuration('ace', st.session_state.pending_config_name, config_id=existing_config['id'])
                    st.session_state.ace_current_config_id = existing_config['id']
                st.session_state.show_overwrite_dialog = False
                st.session_state.show_save_dialog = False
                st.session_state.pending_config_name = None
                st.success(f"✓ Plot '{st.session_state.pending_config_name}' updated!")
                st.rerun()
        with col2:
            if st.button("✏️ Rename", key="rename_ace", width="stretch"):
                st.session_state.show_overwrite_dialog = False
                # Keep save dialog open
                st.rerun()
        with col3:
            if st.button("Cancel", key="cancel_overwrite_ace", width="stretch"):
                st.session_state.show_overwrite_dialog = False
                st.session_state.show_save_dialog = False
                st.session_state.pending_config_name = None
                st.rerun()
        st.markdown("---")
    
    # Show save dialog if active
    elif st.session_state.show_save_dialog:
        with st.form(key="save_config_form"):
            st.markdown("#### 💾 Save Plot")
            
            # Get default name (auto-generated)
            from utils.config_history import save_configuration
            file_names = list(st.session_state.get('ace_objects', {}).keys())
            series_count = len(st.session_state.plot_selections)
            
            # Create preview of auto-generated name
            if file_names:
                first_file = file_names[0]
                if len(first_file) > 20:
                    first_file = first_file[:17] + "..."
                auto_name = f"ACE - {first_file} ({series_count} series)"
            else:
                timestamp_str = datetime.now().strftime("%b%d %H:%M")
                auto_name = f"ACE Plot - {timestamp_str}"
            
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
                    st.session_state.pending_config_name = final_name
                    st.session_state.show_overwrite_dialog = True
                    st.rerun()
                else:
                    # Save new config
                    saved_config = save_configuration('ace', final_name)
                    st.session_state.ace_current_config_id = saved_config['id']
                    st.session_state.show_save_dialog = False
                    st.success(f"✓ Plot '{final_name}' saved!")
                    st.rerun()
            
            if cancelled:
                st.session_state.show_save_dialog = False
                st.rerun()
        st.markdown("---")
    
    # Action buttons (Save/Update)
    if is_updating:
        # Show Update and Save As buttons when working on existing plot
        col_update, col_save = st.columns([1, 1])
        with col_update:
            update_clicked = st.button("🔄 Update Saved Plot", key="update_config_btn", width="stretch", 
                                    help="Update the current saved plot", type="primary")
        with col_save:
            save_clicked = st.button("💾 Save As New", key="save_config_btn", width="stretch", 
                                    help="Save as a new plot")
        
        # Handle update button click
        if update_clicked:
            if not st.session_state.plot_selections:
                st.warning("⚠️ Please add at least one data series before updating")
            else:
                from utils.config_history import get_saved_configurations
                # Find current config to get its name
                configs = get_saved_configurations()
                current_config = next((c for c in configs if c['id'] == current_config_id), None)
                if current_config:
                    save_configuration('ace', current_config['name'], config_id=current_config_id)
                    st.success(f"✓ Plot '{current_config['name']}' updated!")
                    st.rerun()
    else:
        # Show Save button for new plots
        if st.button("💾 Save Plot", key="save_config_btn", width="stretch", 
                    help="Save current plot for later restoration", type="primary"):
            save_clicked = True
        else:
            save_clicked = False
    
    # Handle save button click (both Save and Save As)
    if 'save_clicked' in locals() and save_clicked:
        if not st.session_state.plot_selections:
            st.warning("⚠️ Please add at least one data series before saving")
        else:
            st.session_state.show_save_dialog = True
            st.rerun()

# Footer
st.markdown("---")
try:
    kika_version = kika.__version__ if hasattr(kika, '__version__') else 'unknown'
except:
    kika_version = 'unknown'

st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>ACE Viewer • Powered by KIKA v{kika_version} & PlotBuilder</p>
    <p style='font-size: 0.8em;'>💡 Tip: If data doesn't load correctly, click the 🔄 button in the sidebar to clear cache</p>
</div>
""", unsafe_allow_html=True)
