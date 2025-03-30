from mcnpy._utils import create_repr_section

def energy_distribution_repr(self) -> str:
    """Returns a formatted string representation of an EnergyDistribution object.
    
    This representation provides an overview of the energy distribution law and its parameters.
    
    Returns
    -------
    str
        Formatted string representation of the EnergyDistribution
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Energy Distribution Law ' + str(self.law):^{header_width}}\n"
    header += f"{self.law_name:^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the energy distribution
    description = f"This object represents energy distribution Law {self.law} ({self.law_name}).\n"
    description += "Energy distributions determine the outgoing energy of secondary particles in nuclear reactions.\n\n"
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Distribution Information:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Law Number", self.law, 
        width1=property_col_width, width2=value_col_width)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Law Name", self.law_name, 
        width1=property_col_width, width2=value_col_width)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "IDAT (Offset to data)", self.idat, 
        width1=property_col_width, width2=value_col_width)
    
    # Show applicability information if available
    has_applicability = (len(self.applicability_energies) > 0 and 
                         len(self.applicability_probabilities) > 0)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Has Applicability Data", "Yes" if has_applicability else "No", 
        width1=property_col_width, width2=value_col_width)
    
    if has_applicability:
        n_points = len(self.applicability_energies)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Applicability Data Points", n_points, 
            width1=property_col_width, width2=value_col_width)
        
        if n_points > 0:
            e_min = self.applicability_energies[0].value
            e_max = self.applicability_energies[-1].value
            energy_range = f"{e_min:.6g} - {e_max:.6g} MeV"
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Energy Range", energy_range,
                width1=property_col_width, width2=value_col_width)
    
    # Add law-specific information based on type
    # This is a generic implementation; specific subclasses might override this
    class_name = self.__class__.__name__
    if class_name != "EnergyDistribution":
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Specific Law Type", class_name, 
            width1=property_col_width, width2=value_col_width)
        
    # Add any specific fields based on the law type
    if class_name == "TabularEnergyDistribution":
        n_energies = getattr(self, 'n_incident_energies', 0)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Incident Energies", n_energies, 
            width1=property_col_width, width2=value_col_width)
    
    elif class_name == "MaxwellFissionSpectrum":
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Restriction Energy (U)", getattr(self, 'restriction_energy', 'N/A'), 
            width1=property_col_width, width2=value_col_width)
        
    elif class_name == "EnergyDependentWattSpectrum":
        n_a_energies = getattr(self, 'n_a_energies', 0)
        n_b_energies = getattr(self, 'n_b_energies', 0)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Parameter a Energy Points", n_a_energies, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Parameter b Energy Points", n_b_energies, 
            width1=property_col_width, width2=value_col_width)
            
    elif class_name == "NBodyPhaseSpaceDistribution":
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Bodies (NPSX)", getattr(self, 'npsx', 'N/A'), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Total Mass Ratio (AP)", getattr(self, 'ap', 'N/A'), 
            width1=property_col_width, width2=value_col_width)
        
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for available methods
    methods = {
        ".get_applicability_probability(energy)": "Get probability this law applies at given incident energy",
        ".sample_outgoing_energy(...)": "Sample outgoing energy using this distribution law"
    }
    
    # Add law-specific methods
    if class_name == "KalbachMannDistribution" or class_name == "TabulatedAngleEnergyDistribution":
        methods[".sample_outgoing_energy_angle(...)"] = "Sample both outgoing energy and angle"
        
    if class_name == "LevelScattering":
        methods[".get_cm_energy(energy)"] = "Calculate center-of-mass energy for level scattering"
        methods[".get_lab_energy(energy, cosine)"] = "Calculate laboratory energy given CM cosine"
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + methods_section


def energy_distribution_container_repr(self) -> str:
    """Returns a formatted string representation of the EnergyDistributionContainer object.
    
    This representation provides an overview of available energy distributions by type and MT number.
    
    Returns
    -------
    str
        Formatted string representation of the EnergyDistributionContainer
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Energy Distribution Container':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of energy distributions
    description = (
        "This container holds energy distributions for secondary particles produced in nuclear reactions.\n"
        "Distributions are organized by particle type (neutron, photon, other) and reaction (MT number).\n"
        "Each reaction can have multiple distribution laws that apply in different energy ranges.\n\n"
    )
    
    # Create a summary table of available data
    property_col_width = 40
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    data_summary = "Available Energy Distribution Data:\n"
    data_summary += "-" * header_width + "\n"
    data_summary += "{:<{width1}} {:<{width2}}\n".format(
        "Distribution Type", "Status", width1=property_col_width, width2=value_col_width)
    data_summary += "-" * header_width + "\n"
    
    # Neutron distributions
    n_neutron_mt = len(self.incident_neutron)
    data_summary += "{:<{width1}} {:<{width2}}\n".format(
        "Incident Neutron Distributions", f"{'Available' if n_neutron_mt > 0 else 'None'} ({n_neutron_mt} MT numbers)", 
        width1=property_col_width, width2=value_col_width)
    
    # Photon production
    n_photon_mt = len(self.photon_production)
    data_summary += "{:<{width1}} {:<{width2}}\n".format(
        "Photon Production Distributions", f"{'Available' if n_photon_mt > 0 else 'None'} ({n_photon_mt} MT numbers)", 
        width1=property_col_width, width2=value_col_width)
    
    # Particle production
    n_particles = len(self.particle_production)
    has_particles = n_particles > 0 and any(self.particle_production)
    data_summary += "{:<{width1}} {:<{width2}}\n".format(
        "Particle Production Distributions", f"{'Available' if has_particles else 'None'} ({n_particles} particle types)", 
        width1=property_col_width, width2=value_col_width)
    
    # Delayed neutron
    n_delayed = len(self.delayed_neutron)
    data_summary += "{:<{width1}} {:<{width2}}\n".format(
        "Delayed Neutron Distributions", f"{'Available' if n_delayed > 0 else 'None'} ({n_delayed} groups)", 
        width1=property_col_width, width2=value_col_width)
    
    # Energy-dependent yields
    n_neutron_yields = len(self.neutron_yields)
    data_summary += "{:<{width1}} {:<{width2}}\n".format(
        "Energy-Dependent Neutron Yields", f"{'Available' if n_neutron_yields > 0 else 'None'} ({n_neutron_yields} MT numbers)", 
        width1=property_col_width, width2=value_col_width)
    
    n_photon_yields = len(self.photon_yields)
    data_summary += "{:<{width1}} {:<{width2}}\n".format(
        "Energy-Dependent Photon Yields", f"{'Available' if n_photon_yields > 0 else 'None'} ({n_photon_yields} MT numbers)", 
        width1=property_col_width, width2=value_col_width)
    
    data_summary += "-" * header_width + "\n\n"
    
    # Create detail sections for each distribution type if they exist
    details = ""
    if n_neutron_mt > 0:
        details += "Incident Neutron Distribution MT Numbers:\n"
        details += "-" * header_width + "\n"
        
        # Get the MT numbers and sort them
        mt_numbers = self.get_neutron_reaction_mt_numbers()
        
        # Display MT numbers in columns
        col_width = 8
        num_cols = header_width // col_width
        for i in range(0, len(mt_numbers), num_cols):
            row_mts = mt_numbers[i:i+num_cols]
            row = "  ".join(f"{mt:4d}" for mt in row_mts)
            details += f"  {row}\n"
        details += "\n"
    
    if n_photon_mt > 0:
        details += "Photon Production Distribution MT Numbers:\n"
        details += "-" * header_width + "\n"
        
        # Get the MT numbers and sort them
        mt_numbers = self.get_photon_production_mt_numbers()
        
        # Display MT numbers in columns
        col_width = 8
        num_cols = header_width // col_width
        for i in range(0, len(mt_numbers), num_cols):
            row_mts = mt_numbers[i:i+num_cols]
            row = "  ".join(f"{mt:4d}" for mt in row_mts)
            details += f"  {row}\n"
        details += "\n"
        
    # Create a section for data access methods
    data_access = {
        ".get_neutron_reaction_mt_numbers()": "Get list of MT numbers with neutron distributions",
        ".get_photon_production_mt_numbers()": "Get list of MT numbers with photon distributions",
        ".get_particle_production_mt_numbers(particle_idx)": "Get MT numbers for a specific particle type",
        ".get_neutron_distribution(mt)": "Get neutron distribution for a specific MT number",
        ".get_photon_distribution(mt)": "Get photon distribution for a specific MT number",
        ".get_particle_distribution(particle_idx, mt)": "Get particle distribution for specific type and MT",
        ".get_delayed_neutron_distribution(group)": "Get delayed neutron distribution for a precursor group"
    }
    
    data_access_section = create_repr_section(
        "Data Access Methods:", 
        data_access, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Properties section
    properties = {
        ".has_neutron_data": "Check if neutron reaction distributions are available",
        ".has_photon_production_data": "Check if photon production distributions are available",
        ".has_particle_production_data": "Check if particle production distributions are available",
        ".has_delayed_neutron_data": "Check if delayed neutron distributions are available",
        ".has_neutron_yields": "Check if energy-dependent neutron yields are available",
        ".has_photon_yields": "Check if energy-dependent photon yields are available",
        ".has_particle_yields": "Check if energy-dependent particle yields are available"
    }
    
    properties_section = create_repr_section(
        "Available Properties:", 
        properties, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + data_summary + details + data_access_section + properties_section
