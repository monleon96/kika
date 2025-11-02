"""
NJOY ACE File Generation Page

Process ENDF files to generate ACE files using NJOY
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import os
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mcnpy
from mcnpy.njoy.run_njoy import run_njoy
from mcnpy.endf.read_endf import read_endf
from mcnpy._constants import NDLIBRARY_TO_SUFFIX

# Page config
st.set_page_config(page_title="NJOY Processing - KIKA", page_icon="🔧", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .upload-section {
        padding: 1.5rem;
        border: 2px dashed #667eea;
        border-radius: 0.5rem;
        background: #f8f9fa;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🔧 NJOY ACE File Generation")
st.markdown("Process ENDF files to generate ACE format files for MCNP")
st.markdown("---")

# Initialize session state
if 'njoy_uploaded_files' not in st.session_state:
    st.session_state.njoy_uploaded_files = {}
if 'njoy_endf_objects' not in st.session_state:
    st.session_state.njoy_endf_objects = {}
if 'njoy_processing_results' not in st.session_state:
    st.session_state.njoy_processing_results = []

# Load NJOY settings from session state (set in Settings page)
if 'njoy_exe_path' not in st.session_state:
    st.session_state.njoy_exe_path = "/usr/local/bin/njoy"
if 'njoy_version' not in st.session_state:
    st.session_state.njoy_version = "NJOY 2016.78"
if 'njoy_output_dir' not in st.session_state:
    st.session_state.njoy_output_dir = "./njoy_output"
if 'njoy_create_xsdir' not in st.session_state:
    st.session_state.njoy_create_xsdir = True

# Check NJOY configuration
njoy_exe = st.session_state.njoy_exe_path
njoy_configured = os.path.exists(njoy_exe) if njoy_exe else False

if not njoy_configured:
    st.error("⚠️ NJOY executable not configured or not found!")
    st.markdown(f"**Current path:** `{njoy_exe}`")
    st.markdown("Please configure NJOY in the Settings page:")
    if st.button("Go to Settings →"):
        st.switch_page("pages/5_⚙️_Settings.py")
    st.stop()

# Sidebar - Global file management and Configuration
from components.file_sidebar import render_file_upload_sidebar
from components.clear_cache import render_clear_cache_button

render_file_upload_sidebar()

with st.sidebar:
    st.header("⚙️ NJOY Configuration")
    
    st.markdown(f"**Executable:** `{os.path.basename(njoy_exe)}`")
    st.markdown(f"**Version:** {st.session_state.njoy_version}")
    
    if st.button("⚙️ Change Settings", width="stretch"):
        st.switch_page("pages/5_⚙️_Settings.py")
    
    st.markdown("---")
    st.markdown("💡 **Tip:** Upload ENDF files using the sidebar uploader above. Files are shared across all pages!")
    
    # Render clear-cache button as the last sidebar item
    render_clear_cache_button(key_prefix="clear_cache_njoy")

# Get globally loaded ENDF files
from utils.file_manager import get_endf_files

global_endf_files = get_endf_files()


# Main content with tabs
tab_about, tab_processing, tab_results = st.tabs(["📖 About NJOY", "⚙️ Processing Configuration", "📊 Results"])

# ============================================================================
# TAB 1: About NJOY
# ============================================================================
with tab_about:
    st.header("📖 About NJOY Processing")
    
    # Show info about NJOY
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is NJOY?
        
        NJOY is a nuclear data processing system that converts ENDF-6 format 
        evaluated nuclear data into ACE format files suitable for Monte Carlo 
        transport codes like MCNP.
        
        ### Processing Steps:
        - **RECONR**: Reconstruct cross sections from resonance parameters
        - **BROADR**: Doppler broaden cross sections to target temperature
        - **THERMR**: Add thermal scattering data (for thermal neutrons)
        - **HEATR**: Generate heating (KERMA) data  
        - **GASPR**: Add gas production cross sections
        - **ACER**: Generate final ACE format file for MCNP
        
        ### Temperature Processing
        Cross sections are Doppler-broadened to the specified temperature(s).
        Common temperatures:
        - **293.6 K** (20.45°C) - Room temperature
        - **600 K** - Hot conditions
        - **900 K** - High temperature applications
        """)
    
    with col2:
        st.markdown("""
        ### Key Features
        
        - **Automatic versioning** to prevent overwrites
        - **Multiple temperatures** in one run
        - **Library naming** for organization  
        - **XSDIR generation** for MCNP integration
        - **Organized output** with logs and metadata
        
        ### Output Structure
        ```
        output_dir/
        ├── ace/
        │   └── [library]/
        │       └── [temperature]K/
        │           └── [isotope].ace
        └── njoy_files/
            └── [library]/
                └── [temperature]K/
                    ├── [isotope].input
                    ├── [isotope].output
                    └── [isotope].xsdir
        ```
        
        ### Getting Started
        1. Upload ENDF files using the sidebar uploader
        2. Go to the **Processing Configuration** tab
        3. Configure temperatures and library settings
        4. Run NJOY processing
        5. View results in the **Results** tab
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
        **⚙️ Settings**
        
        Configure NJOY executable path and version in the Settings page.
        """)
    
    with col_tip3:
        st.info("""
        **🔄 Auto-versioning**
        
        Running NJOY multiple times won't overwrite files - versions are created automatically.
        """)

# ============================================================================
# TAB 2: Processing Configuration  
# ============================================================================
with tab_processing:
    st.header("⚙️ Processing Configuration")
    
    # Show informational message if no files are loaded
    if not global_endf_files:
        st.warning("⚠️ No ENDF files loaded. Please upload ENDF files using the sidebar to continue.")
        st.markdown("---")
        st.markdown("""
        ### How to Upload Files
        
        1. Look for the **📁 File Upload** section in the sidebar (left side)
        2. Click the file uploader or drag & drop your ENDF files
        3. Files will be automatically detected and loaded
        4. Once loaded, return here to configure processing
        """)
        st.stop()
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Temperature")
            
        # Temperature input mode
        temp_mode = st.radio(
            "Input mode",
            ["Single temperature", "Multiple temperatures"],
            help="Process at one or multiple temperatures"
        )
        
        if temp_mode == "Single temperature":
            temperature = st.number_input(
            "Temperature (K)",
            min_value=0.1,
            max_value=10000.0,
            value=293.6,
            format="%.1f",
            help="Temperature in Kelvin"
        )
            temperatures = [temperature]
        else:
            temp_input = st.text_area(
                "Temperatures (K)",
                value="293.6\n600\n900",
                help="Enter one temperature per line"
            )
            try:
                temperatures = [float(t.strip()) for t in temp_input.split('\n') if t.strip()]
            except ValueError:
                st.error("Invalid temperature values")
                temperatures = []
        
        st.info(f"Will process at {len(temperatures)} temperature(s)")
    
    with col2:
        st.subheader("📚 Library")
        
        # Library selection
        library_mode = st.radio(
            "Library type",
            ["Standard library", "Custom name"],
            help="Select a standard library or provide custom name",
            key="library_mode_radio"
        )

        # Initialize suffix variables to safe defaults
        custom_suffix = None
        library_key = None
        current_suffix = None
        
        if library_mode == "Standard library":
            standard_libraries = list(NDLIBRARY_TO_SUFFIX.keys())
            library_name = st.selectbox(
                "Select library",
                standard_libraries,
                index=standard_libraries.index('jeff40') if 'jeff40' in standard_libraries else 0,
                help="Standard nuclear data library"
            )
        else:
            library_name = st.text_input(
                "Custom library name",
                value="custom",
                help="Custom identifier (e.g., 'talys', 'modified')",
                key="custom_library_name"
            )
            
            # Check if custom name needs suffix mapping
            library_key = library_name.lower().replace('-', '').replace('/', '').replace('.', '')
            
            # Get current suffix if it exists, otherwise use default
            current_suffix = NDLIBRARY_TO_SUFFIX.get(library_key, "99")
            
            # Always show the suffix input field
            custom_suffix = st.text_input(
                "Library suffix (2-3 chars)",
                value=current_suffix,
                max_chars=3,
                help="Suffix for ACE filenames (e.g., '99', 'TLY')",
                key=f"custom_library_suffix_{library_key}"
            )
            
        if custom_suffix:
            # Update mapping with current value
            NDLIBRARY_TO_SUFFIX[library_key] = custom_suffix
            if custom_suffix != current_suffix:
                st.success(f"✓ Suffix updated to: {custom_suffix}")
            else:
                st.info(f"✓ Using suffix: {custom_suffix}")

    with col3:
        st.subheader("🎯 Options")
        
        # Output directory
        import platform
        system = platform.system()
        
        if system == 'Linux':
            default_path = st.session_state.njoy_output_dir if not st.session_state.njoy_output_dir.startswith(('C:', 'D:', 'E:')) else "/tmp/njoy_output"
            help_text = "Absolute Linux path (e.g., /home/user/njoy_output or ~/njoy_output)"
        else:
            default_path = st.session_state.njoy_output_dir
            help_text = "Absolute path for output files (e.g., C:/Users/user/njoy_output)"
        
        output_dir = st.text_input(
            "Output directory",
            value=default_path,
            help=help_text
        )
        
        # Convert to absolute path and normalize
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        
        # Warn if using Windows path on Linux
        if system == 'Linux' and (output_dir.startswith(('C:', 'D:', 'E:')) or '\\' in output_dir):
            st.error("⚠️ Cannot use Windows paths on Linux! Please use a Linux path (e.g., /home/user/njoy_output)")
        else:
            st.caption(f"📂 Resolved path: `{output_dir}`")
        
        # Additional suffix
        use_suffix = st.checkbox(
            "Use custom suffix",
            value=False,
            help="Add custom suffix to filenames"
        )
        
        if use_suffix:
            additional_suffix = st.text_input(
                "Custom suffix",
                value="",
                help="Custom identifier (e.g., 'test', 'v1')"
            )
        else:
            additional_suffix = None
            st.info("Auto-versioning enabled")
        
        # Create XSDIR
        create_xsdir = st.checkbox(
            "Create XSDIR files",
            value=st.session_state.njoy_create_xsdir,
        help="Generate XSDIR files for MCNP"
    )

    st.markdown("---")

    # File preview
    st.header("📋 Files to Process")

    preview_data = []
    for name, file_data in global_endf_files.items():
        endf_obj = file_data['object']
        preview_data.append({
            "File": name,
            "Isotope": endf_obj.isotope if hasattr(endf_obj, 'isotope') else "Unknown",
            "ZAID": endf_obj.zaid if hasattr(endf_obj, 'zaid') else "Unknown",
            "MAT": endf_obj.mat if hasattr(endf_obj, 'mat') else "Unknown",
            "Temperatures": len(temperatures),
        })

    st.dataframe(preview_data, width="stretch")

    total_runs = len(global_endf_files) * len(temperatures)
    st.info(f"📊 Total ACE files to generate: **{total_runs}** ({len(global_endf_files)} files × {len(temperatures)} temperatures)")

    st.markdown("---")

    # Run NJOY button — full width and centered in the page
    run_button = st.button(
        "Run NJOY",
        type="primary",
        width="stretch",
        disabled=len(temperatures) == 0 or not library_name
    )

    # Note: Clear Results buttons moved to the Results tab for better UX.

    # Run NJOY processing
    if run_button:
        st.markdown("---")
        st.header("🔄 Processing Status")
        
        results = []
        successful = 0
        failed = 0
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_runs = len(global_endf_files) * len(temperatures)
        current_run = 0
        
        for file_name, file_data in global_endf_files.items():
            endf_path = file_data['path']
            endf_obj = file_data['object']
            
            st.markdown(f"### Processing: {file_name}")
            
            for temperature in temperatures:
                current_run += 1
                progress = current_run / total_runs
                progress_bar.progress(progress)
                status_text.text(f"Processing {current_run}/{total_runs}: {file_name} at {temperature} K")
                
                with st.status(f"Temperature: {temperature} K", expanded=True) as status:
                    try:
                        st.write("Running NJOY...")
                        
                        # Run NJOY
                        result = run_njoy(
                            njoy_exe=njoy_exe,
                            endf_path=endf_path,
                            temperature=temperature,
                            library_name=library_name,
                            output_dir=output_dir,
                            njoy_version=st.session_state.njoy_version,
                            additional_suffix=additional_suffix,
                        )
                        
                        if result["returncode"] == 0:
                            st.write("✓ NJOY completed successfully")
                            st.write(f"ACE file: `{result['ace_file']}`")
                            successful += 1
                            status.update(label=f"✓ {temperature} K - Success", state="complete")
                            
                            results.append({
                                'file': file_name,
                                'isotope': endf_obj.isotope,
                                'temperature': temperature,
                                'status': 'success',
                                'ace_file': result.get('ace_file'),
                                'njoy_output': result.get('njoy_output'),
                                'xsdir_file': result.get('xsdir_file'),
                            })
                        else:
                            st.write(f"✗ NJOY failed (return code: {result['returncode']})")
                            failed += 1
                            status.update(label=f"✗ {temperature} K - Failed", state="error")
                            
                            results.append({
                                'file': file_name,
                                'isotope': endf_obj.isotope,
                                'temperature': temperature,
                                'status': 'failed',
                                'error': f"Return code {result['returncode']}",
                                'njoy_output': result.get('njoy_output'),
                            })
                    
                    except Exception as e:
                        st.write(f"✗ Error: {str(e)}")
                        failed += 1
                        status.update(label=f"✗ {temperature} K - Error", state="error")
                        
                        results.append({
                            'file': file_name,
                            'isotope': endf_obj.isotope,
                            'temperature': temperature,
                            'status': 'error',
                            'error': str(e),
                        })
        
        # Store results in session state
        st.session_state.njoy_processing_results = results
        
        # Summary
        progress_bar.empty()
        status_text.empty()
        
        st.markdown("---")
        st.header("📊 Processing Summary")
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        with col_sum1:
            st.metric("Total Runs", total_runs)
        with col_sum2:
            st.metric("Successful", successful, delta=None if failed == 0 else "✓")
        with col_sum3:
            st.metric("Failed", failed, delta=None if failed == 0 else "✗")
        
        if successful > 0:
            st.success(f"✓ Successfully generated {successful} ACE file(s)")
        if failed > 0:
            st.error(f"✗ {failed} run(s) failed")

# ============================================================================
# TAB 3: Results
# ============================================================================
with tab_results:
    st.header("📊 Processing Results")
    
    # Initialize result groups to avoid NameError when there are no results
    successful_results = []
    failed_results = []
    if not st.session_state.njoy_processing_results:
        st.info("No processing results yet. Run NJOY from the **Processing Configuration** tab to see results here.")
    else:
        st.markdown("---")
        st.header("📁 Generated Files")
        
        # Group results by status
        successful_results = [r for r in st.session_state.njoy_processing_results if r['status'] == 'success']
        failed_results = [r for r in st.session_state.njoy_processing_results if r['status'] != 'success']
        
        if successful_results:
            st.subheader("✅ Successful")
            
            # Overall summary
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### 📂 Output Directory Structure")
            st.markdown("""
            Your files have been saved in the following locations:
            
            **ACE Files:** `output_dir/ace/[library]/[temperature]K/`
            - Contains the nuclear data in ACE format for MCNP
            
            **NJOY Files:** `output_dir/njoy_files/[library]/[temperature]K/`
            - `.input` - NJOY input deck used for processing
            - `.output` - Complete NJOY execution log
            - `.xsdir` - XSDIR entry for this ACE file (for MCNP)
            - `.ps` - PostScript visualization (if generated)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            for i, result in enumerate(successful_results):
                with st.expander(
                    f"📄 {result['isotope']} at {result['temperature']} K ({result['file']})",
                    expanded=False
                ):
                    st.markdown("**File Locations:**")
                    
                    # ACE file
                    if result.get('ace_file'):
                        ace_dir = os.path.dirname(result['ace_file'])
                        ace_name = os.path.basename(result['ace_file'])
                        st.markdown(f"📊 **ACE File:**")
                        st.code(result['ace_file'], language=None)
                        st.caption(f"Directory: `{ace_dir}`")
                    
                    st.markdown("")
                    
                    # NJOY processing files
                    if result.get('njoy_output'):
                        njoy_dir = os.path.dirname(result['njoy_output'])
                        st.markdown(f"🔧 **NJOY Processing Files:**")
                        st.code(njoy_dir, language=None)
                        
                        # List all related files
                        st.markdown("Files in this directory:")
                        if result.get('njoy_output'):
                            st.markdown(f"- `{os.path.basename(result['njoy_output'])}` - NJOY execution log")
                        
                        # Check for input file
                        input_file = result['njoy_output'].replace('.output', '.input')
                        if os.path.exists(input_file):
                            st.markdown(f"- `{os.path.basename(input_file)}` - NJOY input deck")
                        
                        # Check for xsdir
                        if result.get('xsdir_file') and os.path.exists(result['xsdir_file']):
                            st.markdown(f"- `{os.path.basename(result['xsdir_file'])}` - XSDIR entry")
                        
                        # Check for viewr output
                        ps_file = result['njoy_output'].replace('.output', '.ps')
                        if os.path.exists(ps_file):
                            st.markdown(f"- `{os.path.basename(ps_file)}` - PostScript visualization")
                    
                    st.markdown("")
                    st.info("💡 All files are saved directly to your output directory. You can find them in the paths shown above.")
    
    if failed_results:
            st.subheader("❌ Failed")
            for i, result in enumerate(failed_results):
                with st.expander(
                    f"📄 {result['isotope']} at {result['temperature']} K ({result['file']})",
                    expanded=True
                ):
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
                    
                    if result.get('njoy_output') and os.path.exists(result['njoy_output']):
                        st.markdown("**NJOY Output Log:**")
                        with open(result['njoy_output'], 'r') as f:
                            log_content = f.read()
                            # Show last 50 lines
                            lines = log_content.split('\n')
                            if len(lines) > 50:
                                st.text_area(
                                    "Last 50 lines of log:",
                                    value='\n'.join(lines[-50:]),
                                    height=300,
                                    key=f"log_failed_{i}"
                                )
                            else:
                                st.text_area(
                                    "Full log:",
                                    value=log_content,
                                    height=300,
                                    key=f"log_failed_{i}"
                                )

    # Button to clear stored results placed below the results listing
    st.markdown("---")
    if st.session_state.get('njoy_processing_results'):
        if st.button("🗑️ Clear Results", key="njoy_clear_results", help="Clear previous processing results", width="stretch"):
            st.session_state.njoy_processing_results = []
            st.rerun()

# Footer
st.markdown("---")
try:
    mcnpy_version = mcnpy.__version__ if hasattr(mcnpy, '__version__') else 'unknown'
except:
    mcnpy_version = 'unknown'

st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>NJOY Processing • Powered by MCNPy v{mcnpy_version} & NJOY</p>
    <p style='font-size: 0.8em;'>💡 Tip: Configure NJOY settings in the Settings page before processing</p>
</div>
""", unsafe_allow_html=True)
