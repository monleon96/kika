"""
Sidebar component for global file management
Shows uploaded ENDF and ACE files across all pages
"""

import streamlit as st
from utils.file_manager import (
    initialize_file_storage,
    add_uploaded_file,
    remove_file,
    get_endf_files,
    get_ace_files,
    format_file_size,
)


def render_file_upload_sidebar():
    """
    Render the file upload and management section in the sidebar
    Should be called on every page that wants to use the global file system
    """
    # Initialize storage
    initialize_file_storage()
    
    with st.sidebar:
        st.markdown("### 📁 File Upload")
        
        # Option to override auto-detection
        with st.expander("⚙️ Advanced Options", expanded=False):
            force_type = st.radio(
                "File type",
                ["Auto-detect", "Force ENDF", "Force ACE"],
                index=0,
                help="Auto-detect works for most files. Use 'Force' if auto-detection fails."
            )
        
        # File uploader with auto-detection
        uploaded_files = st.file_uploader(
            "Upload nuclear data files",
            type=['txt', 'endf', 'endf6', 'ace', '02c', '03c', '20c', '21c', '22c', '23c', '24c'],
            accept_multiple_files=True,
            help="Upload ENDF or ACE files - type will be automatically detected",
            key="global_file_uploader"
        )
        
        # Process uploaded files with auto-detection or forced type
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if already uploaded
                endf_files = get_endf_files()
                ace_files = get_ace_files()
                
                if uploaded_file.name in endf_files or uploaded_file.name in ace_files:
                    continue  # Already uploaded
                
                # Determine file type based on user selection
                if force_type == "Force ENDF":
                    file_type = 'endf'
                elif force_type == "Force ACE":
                    file_type = 'ace'
                else:
                    file_type = None  # Auto-detect
                
                # Add file with specified or auto-detected type
                success, message = add_uploaded_file(uploaded_file, file_type=file_type)
                
                if success:
                    st.success(message, icon="✅")
                else:
                    st.error(message, icon="❌")
        
        # Show uploaded ENDF files
        st.markdown("---")
        st.markdown("### 📄 Loaded ENDF Files")
        
        endf_files = get_endf_files()
        if endf_files:
            for filename, file_data in endf_files.items():
                with st.expander(f"📄 {filename}", expanded=False):
                    endf_obj = file_data['object']
                    
                    # Display file info
                    st.markdown(f"**Isotope:** {endf_obj.isotope if hasattr(endf_obj, 'isotope') else 'Unknown'}")
                    st.markdown(f"**ZAID:** {endf_obj.zaid if hasattr(endf_obj, 'zaid') else 'Unknown'}")
                    st.markdown(f"**MAT:** {endf_obj.mat if hasattr(endf_obj, 'mat') else 'Unknown'}")
                    st.markdown(f"**Size:** {format_file_size(file_data['size'])}")
                    
                    # Remove button
                    if st.button("🗑️ Remove", key=f"remove_endf_{filename}"):
                        success, message = remove_file(filename, 'endf')
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        else:
            st.info("No ENDF files loaded", icon="ℹ️")
        
        # Show uploaded ACE files
        st.markdown("---")
        st.markdown("### 📊 Loaded ACE Files")
        
        ace_files = get_ace_files()
        if ace_files:
            for filename, file_data in ace_files.items():
                with st.expander(f"📊 {filename}", expanded=False):
                    ace_obj = file_data['object']
                    
                    # Display file info
                    st.markdown(f"**ZAID:** {ace_obj.zaid if hasattr(ace_obj, 'zaid') else 'Unknown'}")
                    st.markdown(f"**Temperature:** {ace_obj.temperature if hasattr(ace_obj, 'temperature') else 'Unknown'} K")
                    st.markdown(f"**Library:** {ace_obj.library if hasattr(ace_obj, 'library') else 'Unknown'}")
                    st.markdown(f"**Size:** {format_file_size(file_data['size'])}")
                    
                    # Remove button
                    if st.button("🗑️ Remove", key=f"remove_ace_{filename}"):
                        success, message = remove_file(filename, 'ace')
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        else:
            st.info("No ACE files loaded", icon="ℹ️")
        
        # Clear all button
        if endf_files or ace_files:
            st.markdown("---")
            if st.button("🗑️ Clear All Files", width="stretch"):
                from utils.file_manager import clear_all_files
                clear_all_files()
                st.success("All files cleared!")
                st.rerun()
