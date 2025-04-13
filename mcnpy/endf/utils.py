"""
Utility functions for ENDF file parsing and writing.

Contains helper functions for handling the specific formatting requirements of ENDF files.
"""
import re
from typing import Dict, List, Union, Optional, Tuple, Any, Sequence


def format_endf_number(value: Union[int, float, None], width: int = 11) -> str:
    """
    Format a number according to ENDF specifications.
    
    Args:
        value: The numeric value to format
        width: Field width (default is 11 characters)
        
    Returns:
        Formatted string representation of the number
    """
    if value is None:
        return " " * width
    
    if isinstance(value, int):
        # Format integers right-justified in the field
        return f"{value:{width}d}"
    else:
        # Format floats with correct precision
        # ENDF typically uses fixed-point notation when possible
        if abs(value) < 1.0e10 and abs(value) >= 1.0e-10 or value == 0.0:
            # Use fixed-point notation with 6 decimal places
            return f"{value:{width}.6f}"
        else:
            # Use scientific notation for very large/small numbers
            # ENDF uses a special format where E is omitted: 1.234+5 instead of 1.234E+5
            mantissa = f"{value:.6E}".split('E')[0]
            exponent = int(f"{value:.6E}".split('E')[1])
            return f"{mantissa}{exponent:+d}".rjust(width)


def format_endf_data_line(values: Sequence[Union[int, float, None]], 
                          mat: int, mf: int, mt: int, line_num: int = 0) -> str:
    """
    Format a complete ENDF line with both data and identification parts.
    
    Args:
        values: Sequence of up to 6 numeric values for the data part
        mat: Material number
        mf: File number
        mt: Section number
        line_num: Line sequence number (optional)
        
    Returns:
        Formatted 80-character ENDF line
    """
    # Format the data part (columns 1-66)
    data_part = ""
    for i in range(min(6, len(values))):
        data_part += format_endf_number(values[i])
    
    # Pad to 66 characters if needed
    data_part = data_part.ljust(66)
    
    # Format the identification part (columns 67-80)
    id_part = f"{mat:4d}{mf:2d}{mt:3d}{line_num:5d}"
    
    return data_part + id_part


def parse_number(text: str) -> Union[float, int, None]:
    """
    Parse an ENDF-formatted number.
    
    ENDF uses a special format where numbers can be written in forms like:
    "1.234+5" meaning 1.234×10^5
    
    Args:
        text: The text representation of the number
        
    Returns:
        Parsed number as float or int, or None if parsing fails
    """
    text = text.strip()
    if not text:
        return None
    
    try:
        # Try standard float parsing first
        value = float(text)
        # Return as int if it's a whole number
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        # Handle ENDF-specific format where "+" or "-" might be used instead of "E"
        # For example, "1.234+5" instead of "1.234E+5"
        match = re.search(r'([-+]?\d*\.\d*)([+-]\d+)', text)
        if match:
            try:
                mantissa = float(match.group(1))
                exponent = int(match.group(2))
                value = mantissa * (10 ** exponent)
                if value.is_integer():
                    return int(value)
                return value
            except (ValueError, IndexError):
                pass
                
        # If all parsing fails
        return None


def parse_line(line: str) -> Dict[str, Any]:
    """
    Parse a standard ENDF record line into its components.
    
    Args:
        line: An 80-character ENDF line
        
    Returns:
        Dictionary with parsed components
    """
    result = {}
    
    # Parse data fields (columns 1-66)
    if len(line) >= 66:
        data_part = line[:66]
        # ENDF format typically has 6 fields of 11 characters each
        for i in range(6):
            field_name = f"C{i+1}"
            start = i * 11
            end = start + 11
            if end <= len(data_part):
                field_value = data_part[start:end].strip()
                result[field_name] = parse_number(field_value)
    
    # Parse identification fields (columns 67-80)
    if len(line) >= 75:
        result["MAT"] = int(line[66:70]) if line[66:70].strip() else None
        result["MF"] = int(line[70:72]) if line[70:72].strip() else None
        result["MT"] = int(line[72:75]) if line[72:75].strip() else None
        
    if len(line) >= 80:
        result["SEQ"] = int(line[75:80]) if line[75:80].strip() else None
    
    return result


def parse_endf_id(line: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Parse the identification fields from an ENDF line.
    
    ENDF format specifies:
    - Columns 67-70 (0-indexed: 66-69): MAT number
    - Columns 71-72 (0-indexed: 70-71): MF number
    - Columns 73-75 (0-indexed: 72-74): MT number
    
    Args:
        line: A line from an ENDF file
        
    Returns:
        Tuple of (MAT, MF, MT) numbers
    """
    if len(line) < 75:
        return None, None, None
    
    try:
        # ENDF format has specific columns for MAT, MF, MT
        mat_str = line[66:70].strip()
        mf_str = line[70:72].strip()
        mt_str = line[72:75].strip()
        
        # Convert to integers, handling empty strings
        mat = int(mat_str) if mat_str else None
        mf = int(mf_str) if mf_str else None
        mt = int(mt_str) if mt_str else None
        
        return mat, mf, mt
    except ValueError as e:
        # This might happen if the fields contain non-numeric data
        return None, None, None
