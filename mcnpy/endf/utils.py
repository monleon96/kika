"""
Utility functions for ENDF file parsing and writing.

Contains helper functions for handling the specific formatting requirements of ENDF files.
"""
import re
from typing import Dict,Union, List, Optional, Tuple, Sequence, Any
from .classes.mt import MT
from .classes.mf1.mf1mt import MT451
from .classes.mf import MF
import numpy as np

from math import floor, log10
from typing import Union

def format_endf_number(value: Union[int, float, None], width: int = 11) -> str:
    """
    Format a number according to ENDF specifications.

    The output is an 11-character field made up as follows:
      - The first character is '-' if the number is negative or a blank if positive.
      - The number is written in scientific notation without an 'E'.
      - When the exponent (after normalization) has only one digit (|exponent| < 10),
        the mantissa is printed with 6 decimal digits and the exponent with one digit.
      - When the exponent has two digits (|exponent| >= 10), the mantissa is printed with 5 decimal digits and the exponent with two digits.
      
    For example:
      - A number like -3.14159e-1 will be formatted as "-3.141590-1".
      - A number like 1.234567e+5 will be formatted as " 1.234567+5".
      - A number like 1.0e10 will be formatted as " 1.00000+10".

    Args:
        value: The number to be formatted. If None, returns a blank field.
        width: The total field width (default is 11 characters).

    Returns:
        A string representing the formatted number in ENDF style.
    """
    if value is None:
        return " " * width

    # Special handling for zero: use exponent 0 (one-digit) and 6 decimal places.
    if value == 0:
        return " 0.000000+0"

    sign_char = "-" if value < 0 else " "
    abs_val = abs(value)
    exponent = int(floor(log10(abs_val)))
    mantissa = abs_val / (10 ** exponent)

    # Select the number of decimals based on the exponent.
    # Use 6 decimals if |exponent| < 10, else use 5 decimals.
    # Adjust the mantissa if rounding would push it to 10 or more.
    while True:
        prec = 6 if abs(exponent) < 10 else 5
        mantissa_str = f"{mantissa:1.{prec}f}"
        if float(mantissa_str) < 10:
            break
        mantissa /= 10
        exponent += 1

    # Format the exponent: one digit if |exponent| < 10, two digits otherwise.
    if abs(exponent) < 10:
        exp_str = f"{abs(exponent):d}"
    else:
        exp_str = f"{abs(exponent):02d}"
    exp_sign = '+' if exponent >= 0 else '-'

    formatted = f"{sign_char}{mantissa_str}{exp_sign}{exp_str}"
    return formatted.rjust(width)


# Format constants for ENDF data types
ENDF_FORMAT_FLOAT = 'float'       # Scientific notation (e.g., " 1.234567+5")
ENDF_FORMAT_INT = 'int'           # Integer format (e.g., "         11")
ENDF_FORMAT_INT_ZERO = 'int_zero' # Integer with zero rendered as 0 (not blank)
ENDF_FORMAT_BLANK = 'blank'       # Blank field
ENDF_FORMAT_PRESERVE = 'preserve' # Use value's own type to determine format

def format_endf_data_line(values: Sequence[Union[int, float, None]], 
                         mat: int, mf: int, mt: int, line_num: int = 0,
                         formats: Optional[List[str]] = None) -> str:
    """
    Format a complete ENDF line with both data and identification parts.
    
    Args:
        values: Sequence of up to 6 numeric values for the data part
        mat: Material number
        mf: File number
        mt: Section number
        line_num: Line sequence number (optional)
        formats: Optional list of format types for each value (ENDF_FORMAT_*)
        
    Returns:
        Formatted 80-character ENDF line
    """
    # Format the data part (columns 1-66)
    data_part = ""
    
    # Apply formats if provided, otherwise use default formatting
    if formats:
        # Make sure formats list matches values length
        format_list = formats + [ENDF_FORMAT_PRESERVE] * (len(values) - len(formats))
        format_list = format_list[:len(values)]
        
        for i, (value, fmt) in enumerate(zip(values, format_list)):
            if fmt == ENDF_FORMAT_INT and value is not None:
                # Format as integer, with zero as blank
                if value == 0:
                    data_part += " " * 11  # Zero as blank for regular integers
                else:
                    data_part += f"{int(value):11d}"
            elif fmt == ENDF_FORMAT_INT_ZERO and value is not None:
                # Format as integer, with zero as actual zero
                data_part += f"{int(value):11d}"
            elif fmt == ENDF_FORMAT_BLANK or value is None:
                # Blank field
                data_part += " " * 11
            else:
                # Default to float format
                data_part += format_endf_number(value)
    else:
        # Use default formatting based on value type
        for i, value in enumerate(values[:6]):
            data_part += format_endf_number(value)
    
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


def group_lines_by_mt_with_positions(lines: List[str]) -> Tuple[Dict[int, List[str]], Dict[int, int]]:
    """
    Group lines by MT numbers and track their line counts.
    
    Args:
        lines: List of string lines
        
    Returns:
        Tuple of:
            - Dictionary mapping MT numbers to lists of lines
            - Dictionary mapping MT numbers to line counts
    """
    result: Dict[int, List[str]] = {}
    line_counts: Dict[int, int] = {}
    current_mt = None
    current_lines: List[str] = []
    
    for i, line in enumerate(lines):
        # Parse MT number from the line
        try:
            _, _, mt = parse_endf_id(line)
            
            # Skip MT=0 as a data section (it's a marker)
            if mt == 0:
                # If we were collecting a section, finalize it before the MT=0 marker
                if current_mt is not None and current_lines:
                    result[current_mt] = current_lines
                    line_counts[current_mt] = len(current_lines)
                    current_mt = None
                    current_lines = []
                continue
            
            # Handle section changes
            if current_mt is None:
                # Start a new section
                current_mt = mt
                current_lines = [line]
            elif mt != current_mt:
                # Complete the previous section
                result[current_mt] = current_lines
                line_counts[current_mt] = len(current_lines)
                
                # Start a new section
                current_mt = mt
                current_lines = [line]
            else:
                # Continue current section
                current_lines.append(line)
        except Exception:
            # If we can't parse the line, just add it to the current section if we have one
            if current_mt is not None:
                current_lines.append(line)
    
    # Add the last section if needed
    if current_mt is not None and current_lines:
        result[current_mt] = current_lines
        line_counts[current_mt] = len(current_lines)
    
    return result, line_counts


# Unused - Waiting for further implementation (Sampling)
def replace_section(filepath: str, section: Union['MT', 'MT451', 'MF'], output_filepath: Optional[str] = None) -> bool:
    """
    Replace a section in an ENDF file with a modified version.
    
    Args:
        filepath: Path to the original ENDF file
        section: The modified section object (MT, MT451, or MF)
        output_filepath: Path for the output file (if None, overwrites the input file)
        
    Returns:
        True if the replacement succeeded, False otherwise
    """
    start_pos, end_pos = section.position
    
    # If position information is missing, can't do targeted replacement
    if start_pos is None or end_pos is None:
        return False
    
    try:
        # Read original file
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Get modified section as string
        modified_lines = str(section).split('\n')
        
        # Replace the lines in the original file
        new_lines = lines[:start_pos] + modified_lines + lines[end_pos+1:]
        
        # Write the result
        out_path = output_filepath if output_filepath else filepath
        with open(out_path, 'w') as f:
            f.writelines([line + '\n' for line in new_lines])
        
        return True
    except Exception as e:
        print(f"Error replacing section: {e}")
        return False
