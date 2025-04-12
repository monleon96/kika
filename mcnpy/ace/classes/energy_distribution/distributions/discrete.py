# Law 2: Discrete distributions

from dataclasses import dataclass, field
from typing import List
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution
from mcnpy._utils import create_repr_section

@dataclass
class DiscreteEnergyDistribution(EnergyDistribution):
    """
    Law 2: Discrete energy distribution.
    
    This represents nuclear level excitations where the outgoing energy is discrete.
    For photons, this represents discrete photon energy lines.
    """
    law: int = 2
    n_discrete_energies: int = 0  # Number of discrete energies
    discrete_energies: List[float] = field(default_factory=list)  # Discrete energy values
    probabilities: List[float] = field(default_factory=list)  # Probabilities for each discrete energy
    
    # For photons (according to Table 33)
    lp: int = 0  # Indicator of whether photon is primary (0,1) or non-primary (2)
    eg: float = 0.0  # Photon energy or binding energy
    
    def get_photon_energy(self, incident_energy: float, awr: float) -> float:
        """
        Calculate the photon energy based on LP indicator.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        awr : float
            Atomic weight ratio
            
        Returns
        -------
        float
            The photon energy
        """
        if self.lp == 0 or self.lp == 1:
            # For LP=0 or LP=1, the photon energy is simply EG
            return self.eg
        elif self.lp == 2:
            # For LP=2, the photon energy is EG + (AWR/(AWR+1))*E_N
            return self.eg + (awr / (awr + 1.0)) * incident_energy
        else:
            # Invalid LP, return EG
            return self.eg
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the DiscreteEnergyDistribution.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Discrete Energy Distribution (Law 2)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents discrete energy lines for outgoing particles.\n"
            "For nuclear reactions, this corresponds to discrete energy levels of the residual nucleus.\n"
            "For photon production, this represents discrete gamma ray energies.\n\n"
        )
        
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
            "Number of Discrete Energies", self.n_discrete_energies, 
            width1=property_col_width, width2=value_col_width)
        
        # Add photon-specific data if applicable
        if self.lp > 0:
            lp_desc = {
                1: "Primary photon (energy = EG)",
                2: "Non-primary photon (energy = EG + (AWR/(AWR+1))*E_incident)"
            }
            lp_description = lp_desc.get(self.lp, f"Unknown LP={self.lp}")
            
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Photon Type (LP)", lp_description, 
                width1=property_col_width, width2=value_col_width)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Photon Energy (EG)", f"{self.eg:.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        # If we have discrete energies, show some statistics
        if self.discrete_energies:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Energy Range", f"{min(self.discrete_energies):.6g} - {max(self.discrete_energies):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
            
            # Show a few discrete energies if there aren't too many
            if len(self.discrete_energies) <= 5:
                energy_list = ", ".join(f"{e:.6g}" for e in self.discrete_energies)
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Discrete Energies", energy_list, 
                    width1=property_col_width, width2=value_col_width)
            
            # Show probabilities if they exist
            if self.probabilities and len(self.probabilities) == len(self.discrete_energies):
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Probability Range", f"{min(self.probabilities):.6g} - {max(self.probabilities):.6g}", 
                    width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_photon_energy(incident_energy, awr)": "Calculate photon energy for given incident energy (photon data only)"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add example section
        example = (
            "Example:\n"
            "--------\n"
            "# For photon data with LP=2, calculate the photon energy at 1 MeV\n"
            "photon_energy = distribution.get_photon_energy(incident_energy=1.0, awr=238.0)\n"
            "\n"
            "# Access discrete energy levels and their probabilities directly\n"
            "energies = distribution.discrete_energies\n"
            "probs = distribution.probabilities\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example
