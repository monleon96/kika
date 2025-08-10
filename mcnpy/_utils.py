from mcnpy._constants import ATOMIC_NUMBER_TO_SYMBOL, BOLTZMANN_CONSTANT, K_TO_SUFFIX

def zaid_to_symbol(zaid: int) -> str:
    """
    Convert a ZAID to element-mass symbol (e.g., 26056 -> Fe56)
    
    Parameters
    ----------
    zaid : int
        ZAID identifier (ZZAAA format)
    
    Returns
    -------
    str
        Element symbol with mass number (e.g., "Fe56")
    """
    z = zaid // 1000
    a = str(zaid % 1000)
        
    if z in ATOMIC_NUMBER_TO_SYMBOL:
        return f"{ATOMIC_NUMBER_TO_SYMBOL[z]}{a}"
    return f"{zaid}"  # Fallback if conversion fails

def kelvin_to_MeV(temp: float) -> float:
    """
    Convert temperature in Kelvin to MeV.
    
    Parameters
    ----------
    temp : float
        Temperature in Kelvin
    
    Returns
    -------
    float
        Temperature in MeV
    """
    return temp * BOLTZMANN_CONSTANT

def MeV_to_kelvin(temp: float) -> float:
    """
    Convert temperature in MeV to Kelvin.
    
    Parameters
    ----------
    temp : float
        Temperature in MeV
    
    Returns
    -------
    float
        Temperature in Kelvin
    """
    return temp / BOLTZMANN_CONSTANT 


def temperature_to_suffix(temp_K: float) -> str:
    """
    Convert temperature in Kelvin to MCNP suffix based on K_TO_SUFFIX mapping.
    
    Parameters
    ----------
    temp_K : float
        Temperature in Kelvin
    
    Returns
    -------
    str
        MCNP suffix (e.g., ".02", ".03", etc.)
        
    Examples
    --------
    >>> temperature_to_suffix(293.6)
    '.02'
    >>> temperature_to_suffix(300)
    '.03'
    """
    # Find the closest temperature in K_TO_SUFFIX
    closest_temp = min(K_TO_SUFFIX.keys(), key=lambda k: abs(k - temp_K))
    return K_TO_SUFFIX[closest_temp]


def add_repr_method(method_name, description, buffer=None, method_col_width=30, desc_col_width=45, 
                  align_method="<", align_desc="<"):
    """
    Format a method name and description as a row in a two-column table with text wrapping.
    
    This utility function is designed to be used in __repr__ methods to create consistent
    method documentation sections across different classes.
    
    Parameters
    ----------
    method_name : str
        The name of the method to document
    description : str
        The description of what the method does
    buffer : str, optional
        An existing string buffer to append to. If None, a new string is returned.
    method_col_width : int, optional
        Width of the method name column, default 30
    desc_col_width : int, optional
        Width of the description column, default 45
    align_method : str, optional
        Alignment for method column ("<" for left, "^" for center, ">" for right), default "<"
    align_desc : str, optional
        Alignment for description column, default "<"
        
    Returns
    -------
    str
        The formatted string with the method and description properly formatted and wrapped
        
    Examples
    --------
    >>> buffer = ""
    >>> buffer = add_repr_method(".my_method()", "This is what the method does", buffer)
    >>> buffer = add_repr_method(".another_method(param)", "This has a very long description that will be automatically wrapped to the next line", buffer)
    >>> print(buffer)
    .my_method()                    This is what the method does
    .another_method(param)          This has a very long description that will 
                                    be automatically wrapped to the next line
    """
    result = ""
    
    # Format the first line with method name and first part of description
    if len(description) <= desc_col_width:
        result += "{:{align}{width1}} {:{align2}{width2}}\n".format(
            method_name, description, 
            align=align_method, width1=method_col_width, 
            align2=align_desc, width2=desc_col_width)
    else:
        # Find a good break point for the first line (at a space)
        break_point = desc_col_width
        if ' ' in description[:desc_col_width]:
            # Find the last space within the width limit
            last_space = description[:desc_col_width].rstrip().rfind(' ')
            if last_space > 0:
                break_point = last_space + 1
        
        first_part = description[:break_point].rstrip()
        result += "{:{align}{width1}} {:{align2}{width2}}\n".format(
            method_name, first_part, 
            align=align_method, width1=method_col_width, 
            align2=align_desc, width2=desc_col_width)
        
        # Process remaining text with improved word-aware wrapping
        remaining = description[break_point:].lstrip()
        while remaining:
            # Find a good break point for this chunk
            if len(remaining) <= desc_col_width:
                chunk = remaining
                remaining = ""
            else:
                chunk_end = desc_col_width
                if ' ' in remaining[:desc_col_width]:
                    # Find the last space within the width limit
                    last_space = remaining[:desc_col_width].rstrip().rfind(' ')
                    if last_space > 0:
                        chunk_end = last_space + 1
                
                chunk = remaining[:chunk_end].rstrip()
                remaining = remaining[chunk_end:].lstrip()
            
            # Add the continuation line with empty method column
            result += "{:{align}{width1}} {:{align2}{width2}}\n".format(
                "", chunk, 
                align=align_method, width1=method_col_width, 
                align2=align_desc, width2=desc_col_width)
    
    # Either append to the buffer or return the result
    if buffer is not None:
        return buffer + result
    else:
        return result


def create_repr_section(title, methods_dict, total_width=85, method_col_width=30, desc_col_width=None):
    """
    Create a complete representation section for methods with proper formatting.
    
    This creates a titled section with a header, method table, and bottom border.
    
    Parameters
    ----------
    title : str
        The title for the section, e.g. "Available Methods:"
    methods_dict : dict
        Dictionary mapping method names to their descriptions
    total_width : int, optional
        Total width of the table, default 85
    method_col_width : int, optional
        Width of the method name column, default 30
    desc_col_width : int, optional
        Width of the description column, calculated from total_width if None
        
    Returns
    -------
    str
        A complete formatted section with all methods
        
    Examples
    --------
    >>> methods = {
    ...     ".method1()": "Description of method 1",
    ...     ".method2(param)": "Description of method 2 with parameters"
    ... }
    >>> print(create_repr_section("Available Methods:", methods))
    Available Methods:
    -----------------------------
    Method                         Description
    -----------------------------
    .method1()                     Description of method 1
    .method2(param)                Description of method 2 with parameters
    -----------------------------
    """
    if desc_col_width is None:
        # Calculate description column width based on total width
        desc_col_width = total_width - method_col_width - 3  # -3 for spacing and formatting
    
    # Create the section header
    section = f"{title}\n"
    section += "-" * total_width + "\n"
    
    # Add table header
    section += "{:<{width1}} {:<{width2}}\n".format(
        "Method", "Description", width1=method_col_width, width2=desc_col_width)
    section += "-" * total_width + "\n"
    
    # Add each method
    for method, description in methods_dict.items():
        section = add_repr_method(method, description, section, method_col_width, desc_col_width)
    
    # Add bottom border
    section += "-" * total_width + "\n"
    
    return section
