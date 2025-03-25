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
