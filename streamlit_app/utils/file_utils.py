"""
File handling utilities

Helpers for uploading, processing, and managing files
"""

import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, BinaryIO
import streamlit as st


def save_uploaded_file(uploaded_file: BinaryIO, suffix: Optional[str] = None) -> str:
    """
    Save an uploaded Streamlit file to a temporary location
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        suffix: File suffix/extension (e.g., '.ace')
        
    Returns:
        Path to saved temporary file
    """
    if suffix is None:
        suffix = Path(uploaded_file.name).suffix
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name


def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    Clean up temporary files
    
    Args:
        file_paths: List of file paths to delete
    """
    for path in file_paths:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


def get_file_info(file_path: str) -> dict:
    """
    Get information about a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    path = Path(file_path)
    
    if not path.exists():
        return {}
    
    stat = path.stat()
    
    return {
        'name': path.name,
        'size': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': stat.st_mtime,
        'suffix': path.suffix,
    }


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def validate_ace_file(file_path: str) -> tuple[bool, str]:
    """
    Validate if a file is a valid ACE file
    
    Args:
        file_path: Path to file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to read first few bytes to check format
        with open(file_path, 'rb') as f:
            header = f.read(100)
        
        # Basic validation (ACE files are typically ASCII or binary)
        if len(header) < 10:
            return False, "File too short to be a valid ACE file"
        
        return True, ""
    
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


class FileManager:
    """
    Manage uploaded files in session state
    """
    
    def __init__(self, session_key: str = 'file_manager'):
        """
        Initialize file manager
        
        Args:
            session_key: Key for storing files in session state
        """
        self.session_key = session_key
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {}
    
    def add_file(self, name: str, path: str, metadata: Optional[dict] = None) -> None:
        """Add a file to the manager"""
        st.session_state[self.session_key][name] = {
            'path': path,
            'metadata': metadata or {},
            'info': get_file_info(path)
        }
    
    def get_file(self, name: str) -> Optional[dict]:
        """Get file information"""
        return st.session_state[self.session_key].get(name)
    
    def remove_file(self, name: str) -> None:
        """Remove a file"""
        if name in st.session_state[self.session_key]:
            file_info = st.session_state[self.session_key][name]
            cleanup_temp_files([file_info['path']])
            del st.session_state[self.session_key][name]
    
    def get_all_files(self) -> dict:
        """Get all managed files"""
        return st.session_state[self.session_key]
    
    def clear_all(self) -> None:
        """Clear all files"""
        for name in list(self.get_all_files().keys()):
            self.remove_file(name)
