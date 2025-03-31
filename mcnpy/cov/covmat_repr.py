from mcnpy._utils import create_repr_section

def covmat_repr(self) -> str:
    """
    Get a detailed string representation of the CovMat object.
    
    Parameters
    ----------
    self : CovMat
        The covariance matrix object
        
    Returns
    -------
    str
        String representation with content summary
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Covariance Matrix Information':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of covariance matrix data
    description = (
        "This object contains covariance matrix data from SCALE format.\n"
        "Each matrix represents the covariance between cross sections for specific\n"
        "isotope-reaction pairs across energy groups.\n\n"
    )
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Covariance Data Summary:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # Add summary information
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Energy Groups", self.num_groups, 
        width1=property_col_width, width2=value_col_width)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Covariance Matrices", self.num_matrices, 
        width1=property_col_width, width2=value_col_width)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Unique Isotopes", len(self.unique_isotopes), 
        width1=property_col_width, width2=value_col_width)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Unique Reactions", len(self.unique_reactions), 
        width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for data access using create_repr_section
    data_access = {
        ".unique_isotopes": "Get set of unique isotope IDs",
        ".unique_reactions": "Get set of unique reaction MT numbers",
        ".num_matrices": "Get total number of covariance matrices",
        ".num_groups": "Get number of energy groups"
    }
    
    data_access_section = create_repr_section(
        "How to Access Covariance Data:", 
        data_access, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Add a blank line after the section
    data_access_section += "\n"
    
    # Create a section for available methods using create_repr_section
    methods = {
        ".get_matrix(...)": "Get specific covariance matrix",
        ".get_isotope_reactions()": "Get mapping of isotopes to their reactions",
        ".get_reactions_summary()": "Get DataFrame of isotopes with their reactions",
        ".get_isotope_covariance_matrix(...)": "Build combined covariance matrix for an isotope",
        ".to_dataframe()": "Convert all covariance data to DataFrame",
        ".save_excel()": "Save covariance data to Excel file"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + data_access_section + methods_section
