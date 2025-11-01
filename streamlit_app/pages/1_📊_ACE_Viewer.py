"""
ACE Data Viewer Page

Upload and visualize ACE format nuclear data files
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mcnpy
from mcnpy.plotting import PlotBuilder
import matplotlib.pyplot as plt
import numpy as np

# Page config
st.set_page_config(page_title="ACE Viewer - KIKA", page_icon="📊", layout="wide")

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

# Initialize session state
if 'ace_files' not in st.session_state:
    st.session_state.ace_files = {}
if 'ace_objects' not in st.session_state:
    st.session_state.ace_objects = {}

# Sidebar - File Upload
with st.sidebar:
    st.header("📁 File Upload")
    
    uploaded_files = st.file_uploader(
        "Upload ACE files",
        type=['ace', '02c', '20c', '03c', '30c', '81c', '40c'],
        accept_multiple_files=True,
        help="Upload one or more ACE format files"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            
            if file_name not in st.session_state.ace_files:
                # Save file temporarily and load
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ace') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    # Load ACE file
                    ace_obj = mcnpy.read_ace(tmp_path)
                    
                    # Validate that we have the minimum required data
                    if not ace_obj.header:
                        raise ValueError("ACE file loaded but header is missing")
                    if not ace_obj.cross_section:
                        st.warning(f"⚠️ {file_name}: No cross section data found")
                    
                    st.session_state.ace_files[file_name] = tmp_path
                    st.session_state.ace_objects[file_name] = ace_obj
                    st.success(f"✓ Loaded: {file_name}")
                except Exception as e:
                    st.error(f"✗ Error loading {file_name}: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Display loaded files
    st.markdown("### 📚 Loaded Files")
    if st.session_state.ace_objects:
        for i, (name, ace_obj) in enumerate(st.session_state.ace_objects.items()):
            with st.expander(f"📄 {name}", expanded=False):
                # Access header properties correctly
                if ace_obj.header:
                    st.write(f"**ZAID:** {ace_obj.header.zaid}")
                    if ace_obj.header.temperature is not None:
                        # Temperature is in MeV, convert to Kelvin (1 MeV = 1.160451812e10 K)
                        temp_k = ace_obj.header.temperature * 1.160451812e10
                        st.write(f"**Temperature:** {temp_k:.2f} K ({ace_obj.header.temperature:.6e} MeV)")
                    if ace_obj.header.atomic_weight_ratio is not None:
                        st.write(f"**AWR:** {ace_obj.header.atomic_weight_ratio:.4f}")
                    if ace_obj.header.date:
                        st.write(f"**Date:** {ace_obj.header.date}")
                    if ace_obj.header.comment:
                        st.write(f"**Comment:** {ace_obj.header.comment[:50]}..." if len(ace_obj.header.comment) > 50 else f"**Comment:** {ace_obj.header.comment}")
                
                # Show basic data info
                if ace_obj.header and ace_obj.header.num_energies:
                    st.write(f"**Energy Points:** {ace_obj.header.num_energies}")
                if ace_obj.header and ace_obj.header.num_reactions:
                    st.write(f"**Reactions:** {ace_obj.header.num_reactions}")
                
                if st.button(f"Remove", key=f"remove_{i}"):
                    del st.session_state.ace_files[name]
                    del st.session_state.ace_objects[name]
                    st.rerun()
    else:
        st.info("No files loaded yet")

# Main content
if not st.session_state.ace_objects:
    st.info("👈 Upload ACE files from the sidebar to get started")
    
    # Show example
    st.markdown("### 📖 Example Usage")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Cross Section Plotting:**
        1. Upload one or more ACE files
        2. Select "Cross Section" data type
        3. Choose MT number (e.g., 2 for elastic)
        4. Configure plot settings
        5. Generate plot
        """)
    
    with col2:
        st.markdown("""
        **Angular Distribution Plotting:**
        1. Upload ACE files
        2. Select "Angular Distribution"
        3. Choose MT number and energy
        4. Compare multiple libraries
        5. Export high-quality plots
        """)

else:
    # Plotting interface
    st.header("🎨 Plot Configuration")
    
    # Data type selection
    data_type = st.selectbox(
        "Select Data Type",
        ["Cross Section", "Angular Distribution"],
        help="Choose what type of data to visualize"
    )
    
    # Create tabs for different configurations
    tab1, tab2, tab3 = st.tabs(["📊 Plot Setup", "🎨 Styling", "💾 Export"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # File selection
            st.markdown("### 📁 Select Files")
            selected_files = st.multiselect(
                "Choose files to plot",
                list(st.session_state.ace_objects.keys()),
                default=list(st.session_state.ace_objects.keys())[:3],
                help="Select up to 5 files for comparison"
            )
            
            # MT number
            mt_number = st.number_input(
                "MT Number",
                min_value=1,
                max_value=999,
                value=2,
                help="Reaction type (e.g., 2=elastic, 16=(n,2n), 102=(n,γ))"
            )
        
        with col2:
            # Additional parameters based on data type
            if data_type == "Angular Distribution":
                energy = st.number_input(
                    "Energy (MeV)",
                    min_value=0.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.1,
                    help="Incident neutron energy in MeV"
                )
            
            # Labels
            st.markdown("### 🏷️ Labels")
            use_custom_labels = st.checkbox("Use custom labels", value=False)
            
            if use_custom_labels:
                labels = []
                for file in selected_files:
                    label = st.text_input(
                        f"Label for {file}",
                        value=file.split('.')[0],
                        key=f"label_{file}"
                    )
                    labels.append(label)
            else:
                labels = [f.split('.')[0] for f in selected_files]
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot title
            plot_title = st.text_input(
                "Plot Title",
                value=f"{data_type} Comparison" + (f" at {energy} MeV" if data_type == "Angular Distribution" else "")
            )
            
            # Axis labels
            if data_type == "Cross Section":
                x_label = st.text_input("X-axis label", value="Energy (MeV)")
                y_label = st.text_input("Y-axis label", value="Cross Section (barns)")
            else:
                x_label = st.text_input("X-axis label", value="cos(θ)")
                y_label = st.text_input("Y-axis label", value="Probability Density")
        
        with col2:
            # Scale options
            if data_type == "Cross Section":
                log_x = st.checkbox("Logarithmic X-axis", value=True)
                log_y = st.checkbox("Logarithmic Y-axis", value=True)
            else:
                log_x = False
                log_y = False
            
            # Grid
            show_grid = st.checkbox("Show grid", value=True)
            
            # Legend
            show_legend = st.checkbox("Show legend", value=True)
            legend_location = st.selectbox(
                "Legend location",
                ["best", "upper right", "upper left", "lower right", "lower left"],
                index=0
            )
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_width = st.slider("Figure width (inches)", 6, 16, 10)
            fig_height = st.slider("Figure height (inches)", 4, 12, 6)
        
        with col2:
            dpi = st.selectbox("DPI (resolution)", [100, 150, 200, 300], index=2)
            export_format = st.selectbox("Export format", ["png", "pdf", "svg"], index=0)
    
    # Generate plot button
    st.markdown("---")
    if st.button("🎨 Generate Plot", type="primary", use_container_width=True):
        if not selected_files:
            st.error("Please select at least one file!")
        else:
            try:
                with st.spinner("Generating plot..."):
                    # Create PlotBuilder
                    builder = PlotBuilder()
                    
                    # Add data for each selected file
                    errors = []
                    for file_name, label in zip(selected_files, labels):
                        ace_obj = st.session_state.ace_objects[file_name]
                        
                        try:
                            if data_type == "Cross Section":
                                plot_data = ace_obj.to_plot_data('xs', mt=mt_number, label=label)
                            else:  # Angular Distribution
                                plot_data = ace_obj.to_plot_data('angular', mt=mt_number, energy=energy, label=label)
                            
                            builder.add_data(plot_data)
                        except Exception as e:
                            error_msg = f"{file_name}: {str(e)}"
                            errors.append(error_msg)
                            st.warning(f"⚠️ Skipping {file_name}: {str(e)}")
                    
                    # Check if we have any data to plot
                    if len(builder._data_list) == 0:
                        st.error("No valid data to plot. All files encountered errors.")
                        if errors:
                            with st.expander("Error details"):
                                for error in errors:
                                    st.write(f"• {error}")
                        raise ValueError("No valid plot data")
                    
                    # Configure plot
                    builder.set_labels(title=plot_title, x_label=x_label, y_label=y_label)
                    
                    if data_type == "Cross Section":
                        builder.set_scales(log_x=log_x, log_y=log_y)
                    
                    if show_grid:
                        builder.set_grid(show_grid=True)
                    
                    if show_legend:
                        builder.set_legend_params(show_legend=True, location=legend_location)
                    
                    # Build plot
                    fig = builder.build()
                    
                    # Adjust figure size
                    fig.set_size_inches(fig_width, fig_height)
                    
                    # Display plot
                    st.markdown("### 📊 Plot")
                    st.pyplot(fig, dpi=dpi)
                    
                    # Export button
                    buf = io.BytesIO()
                    fig.savefig(buf, format=export_format, dpi=dpi, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label=f"💾 Download as {export_format.upper()}",
                        data=buf,
                        file_name=f"kika_plot_{data_type.lower().replace(' ', '_')}.{export_format}",
                        mime=f"image/{export_format}",
                        use_container_width=True
                    )
                    
                    st.success("✓ Plot generated successfully!")
                    
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>ACE Viewer • Powered by MCNPy & PlotBuilder</p>
</div>
""", unsafe_allow_html=True)
