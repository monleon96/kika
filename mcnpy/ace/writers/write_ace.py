import os
import io
import numpy as np
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.writers.write_header import write_header

def write_ace(ace: Ace, filepath: str = None, overwrite: bool = False) -> str:
    """
    Write an Ace object to a file in ACE format.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to write
    filepath : str, optional
        The file path to write to. If None, will use the original filename
        from which the ACE data was read, or create a name based on ZAID and temperature.
    overwrite : bool, optional
        Whether to overwrite an existing file, defaults to False
        
    Returns
    -------
    str
        A success message including the path to the written file
        
    Raises
    ------
    ValueError
        If the Ace object has invalid data
    FileExistsError
        If the file already exists and overwrite is False
    IOError
        If there is an error writing to the file
    """
    # Check that the Ace object has the necessary data
    if ace.header is None:
        raise ValueError("Ace object must have a header")
    
    if ace.xss_data is None or len(ace.xss_data) == 0:
        raise ValueError("Ace object must have XSS data")
    
    # If filepath is not provided, use the original filename if available
    if filepath is None:
        if ace.filename is not None:
            # Split the filename to add "_recon" before the extension
            base_path, extension = os.path.splitext(ace.filename)
            filepath = f"{base_path}_recon{extension}"
            print(f"Using modified original filename: {filepath}")
        else:
            # Extract ZAID and temperature from header if available
            zaid = ace.header.zaid if hasattr(ace.header, 'zaid') and ace.header.zaid is not None else "unknown"
            temp = ace.header.temperature if hasattr(ace.header, 'temperature') and ace.header.temperature is not None else "0K"
            
            # Create a filename in the format zaid_temp.ace
            filepath = f"{zaid}_{int(temp)}K.ace"
            
            print(f"No filepath provided. Writing to {filepath} in the current directory.")
    
    # Check if the file already exists
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"File {filepath} already exists and overwrite is False")
    
    # Validate the XSS data indices
    xss_length = len(ace.xss_data)
    used_indices = set()
    
    # Check if xss_data contains raw values rather than XssEntry objects
    # and convert if necessary
    is_raw_values = False
    if xss_length > 0 and not hasattr(ace.xss_data[0], 'index'):
        is_raw_values = True
        # Convert raw values to XssEntry objects
        converted_xss = []
        for i, value in enumerate(ace.xss_data):
            converted_xss.append(XssEntry(index=i, value=value))
        ace.xss_data = converted_xss
    
    for entry in ace.xss_data:
        # Check if each entry has a valid index
        if entry.index is None:
            raise ValueError("Found XSS entry with no index")
        
        # Check if index is within valid range (allowing for 0th element)
        if entry.index < 0 or entry.index >= xss_length:
            raise ValueError(f"Invalid XSS index {entry.index} (valid range: 0 to {xss_length-1})")
        
        # Check for duplicate indices
        if entry.index in used_indices:
            raise ValueError(f"Duplicate XSS index found: {entry.index}")
        
        used_indices.add(entry.index)
    
    # Check for missing indices
    if len(used_indices) != xss_length:
        missing_indices = set(range(xss_length)) - used_indices
        raise ValueError(f"Missing XSS indices: {sorted(missing_indices)}")
    
    try:
        # Use a memory buffer for better performance
        buffer = io.StringIO()
        
        # Write the header
        header_str = write_header(ace.header)
        buffer.write(header_str)
        buffer.write('\n')  # Add a newline after the header
        
        # Sort the XSS data by index
        sorted_xss = sorted(ace.xss_data, key=lambda entry: entry.index)
        
        # Skip the 0th element (if it exists) when writing to file
        # Start from index 1 or index 0 if there's only one element
        start_idx = 1 if len(sorted_xss) > 1 else 0
        
        # Pre-allocate line buffers for better performance
        line_values = []
        lines = []
        
        # Convert all values first (more efficient than inside the loop)
        values = []
        for entry in sorted_xss[start_idx:]:
            # Get the actual numeric value, handling nested XssEntry objects
            value = entry.value
            while isinstance(value, XssEntry):
                value = value.value
            values.append(value)
        
        # Process in chunks of 4 values
        for i, value in enumerate(values):
            # Format value as string
            if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
                value_str = f"{int(value):20d}"
            else:
                value_str = f"{value:20.11E}"
            
            line_values.append(value_str)
            
            # If we have 4 values or this is the last entry, prepare the line
            if (i + 1) % 4 == 0 or i == len(values) - 1:
                line = ''.join(line_values).ljust(80)
                lines.append(line)
                line_values = []
        
        # Write all lines at once
        buffer.write('\n'.join(lines))
        buffer.write('\n')  # Final newline
        
        # Write the entire buffer to file in one operation
        with open(filepath, 'w', buffering=1024*1024) as f:  # Use a large buffer
            f.write(buffer.getvalue())
        
        # Return a success message with the file path
        return f"Success! ACE file written to: {filepath}"
        
    except Exception as e:
        raise IOError(f"Error writing ACE file: {e}")

def write_ace_ascii(ace: Ace, filepath: str = None, overwrite: bool = False) -> str:
    """
    Write an Ace object to a formatted ASCII file.
    This is a convenience wrapper around write_ace.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to write
    filepath : str, optional
        The file path to write to. If None, will use the original filename 
        from which the ACE data was read, or create a name based on ZAID and temperature.
    overwrite : bool, optional
        Whether to overwrite an existing file, defaults to False
        
    Returns
    -------
    str
        A success message including the path to the written file
    """
    return write_ace(ace, filepath, overwrite)

def write_ace_binary(ace: Ace, filepath: str = None, overwrite: bool = False) -> str:
    """
    Write an Ace object to a binary file.
    Note: This is a placeholder for a future implementation.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to write
    filepath : str, optional
        The file path to write to. If None, will use the original filename
        from which the ACE data was read, or create a name based on ZAID and temperature.
    overwrite : bool, optional
        Whether to overwrite an existing file, defaults to False
        
    Returns
    -------
    str
        A success message including the path to the written file
        
    Raises
    ------
    NotImplementedError
        This function is not yet implemented
    """
    raise NotImplementedError("Writing ACE files in binary format is not yet implemented")


