from mcnpy._utils import create_repr_section
import numpy as np

def angular_distribution_repr(self) -> str:
    """
    Returns a user-friendly, formatted string representation of the angular distribution.
    
    Returns
    -------
    str
        Formatted string representation
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Angular Distribution Details':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description
    description = "This object contains angular distribution data "
    
    # Add type-specific description based on the distribution_type
    if self.distribution_type.name == "ISOTROPIC":
        description += "for isotropic scattering (uniform in all directions).\n\n"
    elif self.distribution_type.name == "EQUIPROBABLE":
        description += "in equiprobable bin format (32 cosine bins with equal probability).\n\n"
    elif self.distribution_type.name == "TABULATED":
        description += "in tabulated format (explicit PDF and CDF functions).\n\n"
    elif self.distribution_type.name == "KALBACH_MANN":
        description += "using the Kalbach-Mann formalism (correlated with energy distribution).\n\n"
    else:
        description += "in an unknown format.\n\n"
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Data Information:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # MT number
    mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "MT Number", f"{mt_value}", width1=property_col_width, width2=value_col_width)
    
    # Distribution type
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Distribution Type", self.distribution_type.name,
        width1=property_col_width, width2=value_col_width)
    
    # Energy grid information
    if self.energies:
        num_energies = len(self.energies)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Points", num_energies,
            width1=property_col_width, width2=value_col_width)
        
        min_energy = self.energies[0]  # Now directly a float
        max_energy = self.energies[-1]  # Now directly a float
        energy_range = f"{min_energy:.6g} - {max_energy:.6g} MeV"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Range", energy_range,
            width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for available methods
    methods = {
        ".sample_mu(energy, random_value)": "Sample a scattering cosine for the given energy",
        ".to_dataframe(...)": "Convert to a pandas DataFrame at a specific energy",
        ".plot(...)": "Create a plot of the angular distribution at a specific energy"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Add an example section
    example = (
        "Example:\n"
        "--------\n"
        "# Sample a cosine value at 1 MeV energy with a random value of 0.5\n"
        "mu = angular_distribution.sample_mu(energy=1.0, random_value=0.5)\n\n"
        "# Create a plot of the distribution at 2 MeV\n"
        "fig, ax = angular_distribution.plot(energy=2.0)\n"
    )
    
    # Add property descriptions
    properties = {
        ".mt": "MT number of the reaction (int)",
        ".energies": "List of incident energy points as float values (List[float])"
    }
    
    properties_section = create_repr_section(
        "Property Access:", 
        properties, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + properties_section + "\n" + methods_section + "\n" + example


def isotropic_distribution_repr(self) -> str:
    """
    Returns a user-friendly, formatted string representation of the isotropic distribution.
    
    Returns
    -------
    str
        Formatted string representation
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Isotropic Angular Distribution Details':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    description = (
        "This object represents isotropic angular scattering, where the cosine of the\n"
        "scattering angle is uniformly distributed between -1 and 1. This means the\n"
        "probability density is constant at 0.5 across the entire range.\n\n"
        "Data Structure Overview:\n"
        "- The ACE file may indicate isotropic scattering in two ways:\n"
        "  * By setting the LOCB value to 0 in the locator table\n"
        "  * By storing a distribution table with NE=0 (number of energy points)\n"
        "- No actual distribution data is stored for isotropic scattering\n\n"
    )
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3
    
    info_table = "Data Information:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # MT number
    mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "MT Number", f"{mt_value}", width1=property_col_width, width2=value_col_width)
    
    # Distribution properties
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Distribution Type", "Isotropic (uniform)",
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "PDF Function", "P(μ) = 0.5 for all μ ∈ [-1, 1]",
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Sampling Method", "μ = 2*ξ - 1 where ξ ∈ [0, 1] is random",
        width1=property_col_width, width2=value_col_width)
    
    # Energy grid information
    if self.energies:
        num_energies = len(self.energies)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Points", num_energies,
            width1=property_col_width, width2=value_col_width)
        
        min_energy = self.energies[0]  # Now directly a float
        max_energy = self.energies[-1]  # Now directly a float
        energy_range = f"{min_energy:.6g} - {max_energy:.6g} MeV"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Range", energy_range,
            width1=property_col_width, width2=value_col_width)
    else:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Dependence", "None (same for all energies)",
            width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Raw data properties section
    properties = {
        ".mt": "MT number of the reaction (int)",
        ".energies": "List of incident energy points as float values (List[float])"
    }
    
    properties_section = create_repr_section(
        "Raw Data Properties (Direct from ACE file):", 
        properties, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Methods section
    methods = {
        ".sample_mu(energy, random_value)": "Sample a cosine μ = 2*random_value - 1",
        ".to_dataframe(energy, num_points)": "Convert to a pandas DataFrame with uniform probability",
        ".plot(energy)": "Create a plot of the flat distribution"
    }
    
    methods_section = create_repr_section(
        "Calculation Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Add example for directly accessing property
    example = (
        "Example:\n"
        "--------\n"
        "# Access the MT number\n"
        "mt_value = int(distribution.mt.value)\n\n"
        "# Sample a cosine for any energy (will always be uniform)\n"
        "mu = distribution.sample_mu(energy=1.0, random_value=0.5)  # Returns 0.0\n\n"
        "# Create a plot showing the uniform distribution\n"
        "fig, ax = distribution.plot(energy=1.0)\n"
    )
    
    return header + description + info_table + properties_section + "\n" + methods_section + "\n" + example


def equiprobable_distribution_repr(self) -> str:
    """
    Returns a user-friendly, formatted string representation of the equiprobable bin distribution.
    
    Returns
    -------
    str
        Formatted string representation
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Equiprobable Angular Distribution Details':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    description = (
        "This object represents an angular distribution using 32 equiprobable bins.\n"
        "The cosine range [-1, 1] is divided into 32 bins such that each bin has\n"
        "the same probability (1/32). The bin boundaries vary with incident energy.\n\n"
        "Data Structure Overview:\n"
        "- For each incident energy point, the ACE file stores 33 cosine values\n"
        "  that define the boundaries of 32 equiprobable bins\n"
        "- The first value is always -1 and the last is +1\n"
        "- Each bin has a probability of 1/32 = 0.03125\n"
        "- The density within each bin is constant (flat histogram)\n\n"
    )
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3
    
    info_table = "Data Information:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # MT number
    mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "MT Number", f"{mt_value}", width1=property_col_width, width2=value_col_width)
    
    # Distribution properties
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Distribution Type", "Equiprobable Bin",
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Bins", "32",
        width1=property_col_width, width2=value_col_width)
    
    # Energy grid information
    if self.energies:
        num_energies = len(self.energies)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Points", num_energies,
            width1=property_col_width, width2=value_col_width)
        
        min_energy = self.energies[0]  # Now directly a float
        max_energy = self.energies[-1]  # Now directly a float
        energy_range = f"{min_energy:.6g} - {max_energy:.6g} MeV"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Range", energy_range,
            width1=property_col_width, width2=value_col_width)
    
    # Bin information
    if self.cosine_bins:
        num_bin_sets = len(self.cosine_bins)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Bin Sets", num_bin_sets,
            width1=property_col_width, width2=value_col_width)
        
        # Show bin boundaries for the first energy point if available
        if num_bin_sets > 0 and len(self.cosine_bins[0]) > 0:
            first_set = self.cosine_bins[0]  # Now directly a list of floats
            first_bins = f"[{first_set[0]:.3f}, {first_set[-1]:.3f}] ({len(first_set)-1} bins)"
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "First Energy Bin Range", first_bins,
                width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Raw data properties section
    properties = {
        ".mt": "MT number of the reaction (int)",
        ".energies": "List of incident energy points as float values (List[float])",
        ".cosine_bins": "List of cosine bin boundaries for each energy as float values (List[List[float]])"
    }
    
    properties_section = create_repr_section(
        "Raw Data Properties (Direct from ACE file):", 
        properties, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Methods section
    methods = {
        ".sample_mu(energy, random_value)": "Sample a cosine from the 32 bins at the given energy",
        ".to_dataframe(energy, num_points)": "Convert to a pandas DataFrame at a specific energy",
        ".plot(energy)": "Create a plot of the distribution at a specific energy"
    }
    
    methods_section = create_repr_section(
        "Calculation Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Add example for using this specific distribution type
    example = (
        "Example:\n"
        "--------\n"
        "# Directly access raw cosine bin boundaries for the first energy point\n"
        "first_energy = distribution.energies[0]  # Returns a float, not XssEntry\n"
        "bin_boundaries = distribution.cosine_bins[0]  # Returns a list of floats\n"
        "# Note: 33 values define 32 equiprobable bins\n"
        "\n"
        "# Sample a cosine at 1 MeV with a random value of 0.5\n"
        "# This will select the bin containing the 50% quantile\n"
        "mu = distribution.sample_mu(energy=1.0, random_value=0.5)\n\n"
        "# Create a histogram-style plot of the distribution at 2 MeV\n"
        "fig, ax = distribution.plot(energy=2.0)\n"
    )
    
    return header + description + info_table + properties_section + "\n" + methods_section + "\n" + example


def tabulated_distribution_repr(self) -> str:
    """
    Returns a user-friendly, formatted string representation of the tabulated distribution.
    
    Returns
    -------
    str
        Formatted string representation
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Tabulated Angular Distribution Details':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    description = (
        "This object represents an angular distribution using tabulated probability density\n"
        "functions (PDFs) and cumulative distribution functions (CDFs). The distribution\n"
        "varies with incident energy, with each energy point having its own tabulated\n"
        "probability function.\n\n"
        "Data Structure Overview:\n"
        "- For each incident energy point, the ACE file stores a table with:\n"
        "  * Interpolation flag (1=histogram, 2=linear-linear interpolation)\n"
        "  * Set of cosine values (μ) ranging from -1 to 1\n"
        "  * PDF values (probability density) for each cosine value\n"
        "  * CDF values (cumulative distribution) for each cosine value\n\n"
    )
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3
    
    info_table = "Data Information:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # MT number
    mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "MT Number", f"{mt_value}", width1=property_col_width, width2=value_col_width)
    
    # Distribution properties
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Distribution Type", "Tabulated PDF/CDF",
        width1=property_col_width, width2=value_col_width)
    
    # Energy grid information
    if self.energies:
        num_energies = len(self.energies)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Points", num_energies,
            width1=property_col_width, width2=value_col_width)
        
        min_energy = self.energies[0]  # Now directly a float
        max_energy = self.energies[-1]  # Now directly a float
        energy_range = f"{min_energy:.6g} - {max_energy:.6g} MeV"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Range", energy_range,
            width1=property_col_width, width2=value_col_width)
    
    # Distribution table information
    if self.cosine_grid:
        num_tables = len(self.cosine_grid)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Distribution Tables", num_tables,
            width1=property_col_width, width2=value_col_width)
        
        # Information about the first table if available
        if num_tables > 0 and len(self.cosine_grid[0]) > 0:
            num_points = len(self.cosine_grid[0])
            first_cosine = self.cosine_grid[0][0]  # Now directly a float
            last_cosine = self.cosine_grid[0][-1]  # Now directly a float
            
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Points in First Table", num_points,
                width1=property_col_width, width2=value_col_width)
            
            table_range = f"[{first_cosine:.3f}, {last_cosine:.3f}]"
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "First Table Cosine Range", table_range,
                width1=property_col_width, width2=value_col_width)
    
    # Interpolation scheme
    if self.interpolation:
        interp_types = set(self.interpolation)
        interp_desc = {
            1: "Histogram",
            2: "Linear-Linear"
        }
        interp_str = ", ".join(interp_desc.get(i, f"Type {i}") for i in interp_types)
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Interpolation Type(s)", interp_str,
            width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Raw data properties section
    properties = {
        ".mt": "MT number of the reaction (int)",
        ".energies": "List of incident energy points as float values (List[float])",
        ".interpolation": "List of interpolation flags for each energy point (List[int])",
        ".cosine_grid": "List of cosine grids for each energy as float values (List[List[float]])",
        ".pdf": "List of PDF values for each energy as float values (List[List[float]])",
        ".cdf": "List of CDF values for each energy as float values (List[List[float]])"
    }
    
    properties_section = create_repr_section(
        "Raw Data Properties (Direct from ACE file):", 
        properties, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Methods section
    methods = {
        ".sample_mu(energy, random_value)": "Sample a cosine using the CDF at the given energy",
        ".to_dataframe(energy, num_points)": "Convert to a pandas DataFrame at a specific energy",
        ".plot(energy)": "Create a plot of the distribution at a specific energy"
    }
    
    methods_section = create_repr_section(
        "Calculation Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Add example for using this specific distribution type
    example = (
        "Example:\n"
        "--------\n"
        "# Directly access raw data for the first energy point\n"
        "first_energy = distribution.energies[0]  # Returns a float\n"
        "interp_flag = distribution.interpolation[0]  # 1=histogram, 2=linear-linear\n"
        "cosines = distribution.cosine_grid[0]  # List of floats for cosine values\n"
        "pdf_values = [pdf.value for pdf in distribution.pdf[0]]     # PDF values\n"
        "cdf_values = [cdf.value for cdf in distribution.cdf[0]]     # CDF values\n\n"
        "# Sample a cosine at 1 MeV using inverse CDF with a random value of 0.5\n"
        "mu = distribution.sample_mu(energy=1.0, random_value=0.5)\n\n"
        "# Create a plot showing the PDF at 2 MeV\n"
        "fig, ax = distribution.plot(energy=2.0)\n"
    )
    
    return header + description + info_table + properties_section + "\n" + methods_section + "\n" + example


def kalbach_mann_distribution_repr(self) -> str:
    """
    Returns a user-friendly, formatted string representation of the Kalbach-Mann distribution.
    
    Returns
    -------
    str
        Formatted string representation
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Kalbach-Mann Angular Distribution Details':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    description = (
        "This object represents an angular distribution using the Kalbach-Mann formalism.\n"
        "The Kalbach-Mann model correlates energy and angle distributions, with parameters\n"
        "R (precompound fraction) and A (angular slope) that vary with outgoing energy.\n\n"
        "Data Structure Overview:\n"
        "- In the ACE file, a LOCB value of -1 indicates a Kalbach-Mann distribution\n"
        "- The actual angular distribution parameters (R and A) are stored in the\n"
        "  energy distribution section as a Law=44 distribution\n"
        "- This object stores reference indices to locate the Law=44 data when needed\n\n"
        "IMPORTANT: This distribution REQUIRES Law=44 data from the energy distribution\n"
        "section (DLW/DLWH blocks). The ACE object must be provided to all methods that\n"
        "calculate or sample angular distributions. Without this data, methods will raise\n"
        "a Law44DataError exception.\n\n"
    )
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3
    
    info_table = "Data Information:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # MT number
    mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "MT Number", f"{mt_value}", width1=property_col_width, width2=value_col_width)
    
    # Distribution properties
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Distribution Type", "Kalbach-Mann (Law=44)",
        width1=property_col_width, width2=value_col_width)
    
    # Law 44 requirement
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Requires Law=44 Data", "Yes",
        width1=property_col_width, width2=value_col_width)
    
    # ACE requirement
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Requires ACE Object", "Yes",
        width1=property_col_width, width2=value_col_width)
    
    # Reaction index information
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Reaction Index", self.reaction_index,
        width1=property_col_width, width2=value_col_width)
    
    # Particle production information
    if self.is_particle_production:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Particle Production", f"Yes (particle index: {self.particle_idx})",
            width1=property_col_width, width2=value_col_width)
    else:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Particle Production", "No (incident neutron reaction)",
            width1=property_col_width, width2=value_col_width)
    
    # Kalbach-Mann formula
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Kalbach-Mann Formula", "p(μ) = (a/2)/sinh(a) * [cosh(aμ) + r*sinh(aμ)]",
        width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Raw data properties section
    properties = {
        ".mt": "MT number of the reaction (int)",
        ".reaction_index": "Index of the reaction in the energy distribution table (int)",
        ".is_particle_production": "Whether this is a particle production reaction (bool)",
        ".particle_idx": "Particle type index if particle production (int)"
    }
    
    properties_section = create_repr_section(
        "Raw Data Properties (Reference data from ACE file):", 
        properties, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Error handling section
    error_section = "Error Handling:\n"
    error_section += "-" * header_width + "\n"
    error_section += (
        "If Law=44 data is required but not available, methods will raise Law44DataError.\n"
        "This can happen when:\n"
        "  - ACE object is not provided to methods\n"
        "  - ACE object doesn't contain energy distribution data\n"
        "  - No Law=44 distribution is found for this reaction\n"
        "  - Distribution data is incomplete or invalid\n"
    )
    error_section += "-" * header_width + "\n\n"
    
    # Methods section
    methods = {
        ".sample_mu(energy, random_value, ace)": "Sample a cosine using Kalbach-Mann at the given energy",
        ".to_dataframe(energy, ace, num_points)": "Convert to a pandas DataFrame at a specific energy",
        ".plot(energy, ace)": "Create a plot of the distribution at a specific energy"
    }
    
    methods_section = create_repr_section(
        "Calculation Methods (All require ACE object):", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Add example for using this specific distribution type
    example = (
        "Example:\n"
        "--------\n"
        "# Access reference properties\n"
        "mt_value = int(distribution.mt.value)\n"
        "reaction_idx = distribution.reaction_index\n"
        "is_particle = distribution.is_particle_production\n\n"
        "# Sample a cosine at 14 MeV using the ACE object (required for Law=44 data)\n"
        "try:\n"
        "    mu = distribution.sample_mu(energy=14.0, random_value=0.5, ace=ace_object)\n"
        "except Law44DataError as e:\n"
        "    print(f\"Error: {e}\")\n\n"
        "# Create a plot showing the Kalbach-Mann distribution at 14 MeV\n"
        "try:\n"
        "    fig, ax = distribution.plot(energy=14.0, ace=ace_object)\n"
        "except Law44DataError as e:\n"
        "    print(f\"Error: {e}\")\n"
    )
    
    return header + description + info_table + properties_section + "\n" + error_section + methods_section + "\n" + example


def angular_container_repr(self) -> str:
    """
    Returns a user-friendly, formatted string representation of the container.
    
    Returns
    -------
    str
        Formatted string representation
    """
    header_width = 90
    header = "=" * header_width + "\n"
    header += f"{'Angular Distribution Container':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    description = (
        "This container holds angular distributions for different reaction types and particles.\n"
        "Angular distributions describe the probability of a particle scattering at a specific angle,\n"
        "represented by the cosine of the scattering angle (μ) ranging from -1 to +1.\n\n"
        "Note: Some distributions (Kalbach-Mann/Law=44) require additional data from the energy\n"
        "distribution section. For these distributions, the ACE object must be provided when\n"
        "calling methods to avoid Law44DataError exceptions.\n\n"
    )
    
    # Create a summary table of available data
    property_col_width = 40
    value_col_width = header_width - property_col_width - 3
    
    info_table = "Available Angular Distribution Data:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Distribution Type", "Status", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # Elastic scattering
    elastic_status = "Available"
    if not self.has_elastic_data:
        elastic_status = "Not available or isotropic"
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Elastic Scattering (MT=2)", elastic_status,
        width1=property_col_width, width2=value_col_width)
    
    # Neutron reaction distributions
    neutron_status = f"Available ({len(self.incident_neutron)} reactions)"
    if not self.has_neutron_data:
        neutron_status = "Not available"
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Neutron Reactions", neutron_status,
        width1=property_col_width, width2=value_col_width)
    
    # Photon production distributions
    photon_status = f"Available ({len(self.photon_production)} reactions)"
    if not self.has_photon_production_data:
        photon_status = "Not available"
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Photon Production", photon_status,
        width1=property_col_width, width2=value_col_width)
    
    # Particle production distributions
    if self.has_particle_production_data:
        num_particle_types = len(self.particle_production)
        particle_counts = [len(p) for p in self.particle_production]
        particle_status = f"Available ({num_particle_types} types, {sum(particle_counts)} total reactions)"
    else:
        particle_status = "Not available"
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Particle Production", particle_status,
        width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for data access - only include available data
    data_access = {}
    
    # Only add elastic if available
    if self.has_elastic_data:
        data_access[".elastic"] = "Access elastic scattering angular distribution"
    
    # Only add neutron reactions if available
    if self.has_neutron_data:
        data_access[".incident_neutron[MT]"] = "Dictionary of angular distributions for neutron reactions"
    
    # Only add photon production if available
    if self.has_photon_production_data:
        data_access[".photon_production[MT]"] = "Dictionary of angular distributions for photon production"
    
    # Only add particle production if available
    if self.has_particle_production_data:
        data_access[".particle_production[particle_idx][MT]"] = "List of dictionaries for particle production"
    
    data_access_section = create_repr_section(
        "Data Access Properties:", 
        data_access, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Add methods section - only include get methods for available data
    methods = {}
    
    # Only add get methods for data types that are available
    if self.has_neutron_data:
        methods[".get_neutron_reaction_mt_numbers()"] = "Get list of MT numbers for neutron reactions"
    
    if self.has_photon_production_data:
        methods[".get_photon_production_mt_numbers()"] = "Get list of MT numbers for photon production"
    
    if self.has_particle_production_data:
        methods[".get_particle_production_mt_numbers()"] = "Get list of MT numbers for each particle type"
    
    # Always include these general methods
    methods.update({
        ".sample_mu(...)": "Sample a scattering cosine for a specific reaction",
        ".to_dataframe(...)": "Convert distribution to DataFrame",
        ".plot(...)": "Plot an angular distribution",
        ".plot_energy_comparison(...)": "Compare distributions at different energies"
    })
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Add note about Kalbach-Mann to example section
    example = (
        "Example:\n"
        "--------\n"
        "# Get MT numbers for neutron reactions with angular distributions\n"
        "mt_numbers = container.get_neutron_reaction_mt_numbers()\n\n"
        "# Sample a scattering cosine for MT=16 at 14 MeV\n"
        "# Note: For Kalbach-Mann distributions, you need to provide the ACE object\n"
        "try:\n"
        "    mu = container.sample_mu(mt=16, energy=14.0, random_value=0.5, ace=ace_object)\n"
        "except Law44DataError as e:\n"
        "    print(f\"Error: {e} - This MT likely uses Kalbach-Mann which requires ACE data\")\n\n"
        "# Plot the angular distribution for MT=16 at 14 MeV\n"
        "fig, ax = container.plot(mt=16, energy=14.0, ace=ace_object)  # ACE needed for Kalbach-Mann\n\n"
        "# Compare angular distributions at different energies\n"
        "fig, ax = container.plot_energy_comparison(mt=16, energies=[1.0, 5.0, 14.0], ace=ace_object)\n"
    )
    
    return header + description + info_table + data_access_section + "\n" + methods_section + "\n" + example
