import os
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
            filepath = ace.filename
            print(f"Using original filename: {filepath}")
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
    
    for entry in ace.xss_data:
        # Check if each entry has a valid index
        if entry.index is None:
            raise ValueError("Found XSS entry with no index")
        
        # Check if index is within valid range
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
        with open(filepath, 'w') as f:
            # Write the header
            header_str = write_header(ace.header)
            f.write(header_str)
            f.write('\n')  # Add a newline after the header
            
            # Sort the XSS data by index
            sorted_xss = sorted(ace.xss_data, key=lambda entry: entry.index)
            
            # Write the XSS data in chunks of 4 values per line
            # Each value gets 20 characters (80/4)
            line = ""
            for i, entry in enumerate(sorted_xss):
                # Check if the value is an integer or can be represented exactly as an integer
                if isinstance(entry.value, int) or entry.value.is_integer():
                    # Format integers with right alignment
                    value_str = f"{int(entry.value):20d}"
                else:
                    # Format floats with scientific notation
                    value_str = f"{entry.value:20.11E}"
                
                line += value_str
                
                # If we have 4 values or this is the last entry, write the line
                if (i + 1) % 4 == 0 or i == len(sorted_xss) - 1:
                    # Pad the line to 80 characters if needed
                    line = line.ljust(80)
                    f.write(line)
                    f.write('\n')
                    line = ""
        
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


