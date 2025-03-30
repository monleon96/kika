from mcnpy._utils import create_repr_section

def reaction_xs_repr(self) -> str:
    """Returns a formatted string representation of the ReactionCrossSection object.
    
    This representation provides an overview of a single reaction's cross section data.
    
    :returns: Formatted string representation of the ReactionCrossSection
    :rtype: str
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Reaction Cross Section Details':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the reaction cross section
    description = f"This object contains cross section data for reaction MT={int(self.mt.value) if self.mt else 'Unknown'}.\n\n"
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Data Information:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "MT Number", f"{int(self.mt.value) if self.mt else 'Not specified'}", 
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Energy Grid Start Index", self.energy_idx,
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Energy Points", self.num_energies,
        width1=property_col_width, width2=value_col_width)
    
    num_xs_values = len(self.xs_values)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Cross Section Values", num_xs_values,
        width1=property_col_width, width2=value_col_width)
    
    # Add XS value range if available
    if num_xs_values > 0:
        min_xs = min(xs.value for xs in self.xs_values)
        max_xs = max(xs.value for xs in self.xs_values)
        xs_range = f"{min_xs:.6g} - {max_xs:.6g} barns"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Cross Section Range", xs_range,
            width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for available methods
    methods = {
        ".get_energies(energy_grid)": "Get the energy points for this reaction"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + methods_section


def xs_data_repr(self) -> str:
    """Returns a formatted string representation of the CrossSectionData object.
    
    This representation provides an overview of all reaction cross sections and how to access them.
    
    :returns: Formatted string representation of the CrossSectionData
    :rtype: str
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Cross Section Data Information':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of cross section data
    description = (
        "This object contains cross section data for multiple reactions.\n"
        "Each reaction is identified by its MT number according to ENDF/B standards.\n\n"
    )
    
    # Create a summary table of available data
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Cross Section Data Summary:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # Add summary information
    num_reactions = len(self.reactions)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Reactions", num_reactions, 
        width1=property_col_width, width2=value_col_width)
    
    # List some common reaction MT numbers if available
    common_mts = {
        1: "Total",
        2: "Elastic",
        16: "(n,2n)",
        17: "(n,3n)",
        18: "Fission",
        102: "(n,γ)"
    }
    
    if num_reactions > 0:
        mt_list = self.mt_numbers
        common_available = [f"MT={mt} ({common_mts[mt]})" for mt in mt_list if mt in common_mts]
        
        if common_available:
            common_str = ", ".join(common_available[:5])
            if len(common_available) > 5:
                common_str += ", ..."
                
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Common Reactions Available", common_str,
                width1=property_col_width, width2=value_col_width)
        
        # List some MT numbers if we haven't already shown them all
        if len(common_available) < 5 and len(mt_list) > 0:
            mt_str = ", ".join(f"MT={mt}" for mt in mt_list[:5])
            if len(mt_list) > 5:
                mt_str += ", ..."
                
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Available MT Numbers", mt_str,
                width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for data access and methods
    properties = {
        ".has_data": "Check if any reaction cross section data is available",
        ".mt_numbers": "Get a list of available MT numbers"
    }
    
    properties_section = create_repr_section(
        "Properties:", 
        properties, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Add extra space for readability
    properties_section += "\n"
    
    # Methods section
    methods = {
        ".get_reaction_xs(mt)": "Get cross section data for a specific reaction",
        ".get_interpolated_xs(mt, energy, energy_grid)": "Get interpolated cross section value",
        ".plot_reaction_xs(mt, energy_grid, ...)": "Plot cross section for a specific reaction"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + properties_section + methods_section
