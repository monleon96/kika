"""
Centralized file management for KIKA
Handles ENDF and ACE file uploads with auto-detection
"""

import os
import tempfile
import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def initialize_file_storage():
    """Initialize session state for file storage"""
    if 'uploaded_endf_files' not in st.session_state:
        st.session_state.uploaded_endf_files = {}  # {filename: {'path': str, 'object': endf_obj}}
    
    if 'uploaded_ace_files' not in st.session_state:
        st.session_state.uploaded_ace_files = {}  # {filename: {'path': str, 'object': ace_obj}}
    
    if 'temp_file_paths' not in st.session_state:
        st.session_state.temp_file_paths = []  # Track temp files for cleanup


def _is_valid_zaid(zaid) -> bool:
    """
    Check if ZAID is valid (not None, not 0, and reasonable value)
    ZAID format: ZZZAAA (Z = atomic number, A = mass number)
    """
    if zaid is None:
        return False
    
    try:
        zaid_int = int(zaid)
        # ZAID should be positive and reasonable (1000 to 999999)
        # Z can be 1-118, A can be 0-300 roughly
        if zaid_int <= 0 or zaid_int > 999999:
            return False
        
        # Extract Z (atomic number) and A (mass number)
        z = zaid_int // 1000
        a = zaid_int % 1000
        
        # Basic validation: Z should be 1-118 (known elements)
        if z < 1 or z > 118:
            return False
        
        # A should be reasonable (0-400, including metastable states)
        if a > 400:
            return False
        
        return True
    except (ValueError, TypeError):
        return False


def detect_file_type(file_path: str) -> Tuple[Optional[str], Optional[object]]:
    """
    Auto-detect file type by trying to parse it
    
    Strategy:
    1. Try reading as ENDF first
    2. Check if ZAID is valid - if yes, it's ENDF
    3. If not valid, try reading as ACE
    4. Check if ZAID is valid - if yes, it's ACE
    5. If neither works, return None
    
    Returns
    -------
    Tuple[Optional[str], Optional[object]]
        ('endf', endf_object) or ('ace', ace_object) or (None, None)
    """
    from kika.endf.read_endf import read_endf
    from kika.ace.parsers import read_ace
    
    # Try ENDF first
    try:
        endf_obj = read_endf(file_path)
        if endf_obj:
            # Check for valid ZAID
            zaid = getattr(endf_obj, 'zaid', None)
            if _is_valid_zaid(zaid):
                # Additional validation: should have MAT and MF
                if hasattr(endf_obj, 'mat') and hasattr(endf_obj, 'mf'):
                    return 'endf', endf_obj
    except Exception:
        pass
    
    # Try ACE if ENDF failed or had invalid ZAID
    try:
        ace_obj = read_ace(file_path)
        if ace_obj:
            # Check for valid ZAID
            zaid = getattr(ace_obj, 'zaid', None)
            if _is_valid_zaid(zaid):
                return 'ace', ace_obj
    except Exception:
        pass
    
    # Neither format worked
    return None, None


def add_uploaded_file(uploaded_file, file_type: Optional[str] = None) -> Tuple[bool, str]:
    """
    Add an uploaded file to the global storage
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
    file_type : Optional[str]
        'endf', 'ace', or None for auto-detection
    
    Returns
    -------
    Tuple[bool, str]
        (success, message)
    """
    try:
        # Create temp file
        suffix = Path(uploaded_file.name).suffix
        temp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            suffix=suffix,
            delete=False
        )
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        
        temp_path = temp_file.name
        st.session_state.temp_file_paths.append(temp_path)
        
        # Auto-detect if type not specified
        if file_type is None:
            detected_type, file_obj = detect_file_type(temp_path)
            if detected_type is None:
                os.unlink(temp_path)
                st.session_state.temp_file_paths.remove(temp_path)
                return False, (
                    f"Could not detect file type for '{uploaded_file.name}'. "
                    "File could not be parsed as ENDF or ACE, or ZAID validation failed. "
                    "Try using 'Force ENDF' or 'Force ACE' in Advanced Options."
                )
            file_type = detected_type
        else:
            # Parse with specified type
            if file_type == 'endf':
                from kika.endf.read_endf import read_endf
                file_obj = read_endf(temp_path)
            elif file_type == 'ace':
                from kika.ace.parsers import read_ace
                file_obj = read_ace(temp_path)
            else:
                os.unlink(temp_path)
                st.session_state.temp_file_paths.remove(temp_path)
                return False, f"Unknown file type: {file_type}"
        
        # Store in appropriate collection
        file_data = {
            'path': temp_path,
            'object': file_obj,
            'original_name': uploaded_file.name,
            'size': len(uploaded_file.getvalue()),
        }
        
        if file_type == 'endf':
            st.session_state.uploaded_endf_files[uploaded_file.name] = file_data
            isotope = file_obj.isotope if hasattr(file_obj, 'isotope') else 'Unknown'
            return True, f"Added ENDF file: {uploaded_file.name} ({isotope})"
        
        elif file_type == 'ace':
            st.session_state.uploaded_ace_files[uploaded_file.name] = file_data
            zaid = file_obj.zaid if hasattr(file_obj, 'zaid') else 'Unknown'
            return True, f"Added ACE file: {uploaded_file.name} (ZAID: {zaid})"
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            if temp_path in st.session_state.temp_file_paths:
                st.session_state.temp_file_paths.remove(temp_path)
        return False, f"Error processing file: {str(e)}"


def remove_file(filename: str, file_type: str) -> Tuple[bool, str]:
    """
    Remove a file from storage
    
    Parameters
    ----------
    filename : str
        Name of the file to remove
    file_type : str
        'endf' or 'ace'
    
    Returns
    -------
    Tuple[bool, str]
        (success, message)
    """
    try:
        if file_type == 'endf':
            file_data = st.session_state.uploaded_endf_files.get(filename)
            if file_data:
                temp_path = file_data['path']
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if temp_path in st.session_state.temp_file_paths:
                    st.session_state.temp_file_paths.remove(temp_path)
                del st.session_state.uploaded_endf_files[filename]
                return True, f"✅ Removed ENDF file: {filename}"
        
        elif file_type == 'ace':
            file_data = st.session_state.uploaded_ace_files.get(filename)
            if file_data:
                temp_path = file_data['path']
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if temp_path in st.session_state.temp_file_paths:
                    st.session_state.temp_file_paths.remove(temp_path)
                del st.session_state.uploaded_ace_files[filename]
                return True, f"✅ Removed ACE file: {filename}"
        
        return False, f"❌ File not found: {filename}"
    
    except Exception as e:
        return False, f"❌ Error removing file: {str(e)}"


def clear_all_files():
    """Clear all uploaded files and cleanup temp files"""
    # Clean up all temp files
    for temp_path in st.session_state.temp_file_paths:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
    
    st.session_state.temp_file_paths = []
    st.session_state.uploaded_endf_files = {}
    st.session_state.uploaded_ace_files = {}


def get_endf_files() -> Dict[str, dict]:
    """Get all uploaded ENDF files"""
    return st.session_state.get('uploaded_endf_files', {})


def get_ace_files() -> Dict[str, dict]:
    """Get all uploaded ACE files"""
    return st.session_state.get('uploaded_ace_files', {})


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
