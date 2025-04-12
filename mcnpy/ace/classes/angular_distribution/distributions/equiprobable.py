from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from mcnpy.ace.classes.xss import XssEntry
from mcnpy.ace.classes.angular_distribution.base import AngularDistribution
from mcnpy.ace.classes.angular_distribution.types import AngularDistributionType
from mcnpy._utils import create_repr_section


@dataclass
class EquiprobableAngularDistribution(AngularDistribution):
    """Angular distribution for 32 equiprobable bin scattering."""
    _cosine_bins: List[List[XssEntry]] = field(default_factory=list)  # List of 33 cosines for each energy
    
    def __post_init__(self):
        super().__post_init__()
        self.distribution_type = AngularDistributionType.EQUIPROBABLE
    
    @property
    def cosine_bins(self) -> List[List[float]]:
        """Get cosine bin values as lists of floats."""
        return [[c.value for c in cosine_list] for cosine_list in self._cosine_bins]

    def to_dataframe(self, energy: float, num_points: int = 100, interpolate: bool = False) -> Optional[pd.DataFrame]:
        """
        Convert equiprobable bin distribution to a pandas DataFrame.
        
        Parameters
        ----------
        energy : float
            Incident energy to evaluate the distribution at
        num_points : int, optional
            Number of angular points to generate when interpolating, defaults to 100
        interpolate : bool, optional
            Whether to interpolate onto a regular grid (True) or return original points (False)
            
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with 'energy', 'cosine', and 'pdf' columns,
            optionally with bin boundary columns if not interpolating
            Returns None if pandas is not available
        """
        # If no energies in this distribution, return isotropic for all directions
        if not self._energies:
            # For specific energy, return isotropic distribution
            if interpolate:
                cosines = np.linspace(-1, 1, num_points)
                return pd.DataFrame({
                    'energy': np.full_like(cosines, energy, dtype=float),
                    'cosine': cosines,
                    'pdf': np.ones_like(cosines) * 0.5
                })
            else:
                return pd.DataFrame({
                    'energy': [energy, energy],
                    'cosine': [-1.0, 1.0],
                    'pdf': [0.5, 0.5]
                })
        
        # If energy is outside our range, return uniform distribution
        if energy < self._energies[0].value or energy > self._energies[-1].value:
            if interpolate:
                cosines = np.linspace(-1, 1, num_points)
                return pd.DataFrame({
                    'energy': np.full_like(cosines, energy, dtype=float),
                    'cosine': cosines,
                    'pdf': np.ones_like(cosines) * 0.5
                })
            else:
                return pd.DataFrame({
                    'energy': [energy, energy],
                    'cosine': [-1.0, 1.0],
                    'pdf': [0.5, 0.5]
                })
        
        # Find bounding energy indices
        energy_values = self.energies
        idx = np.searchsorted(energy_values, energy)
        
        # Get appropriate cosine bins based on energy
        if idx == 0:
            bin_values = self.cosine_bins[0]
        elif idx >= len(energy_values):
            bin_values = self.cosine_bins[-1]
        else:
            # Interpolate between energy points
            e_low = energy_values[idx-1]
            e_high = energy_values[idx]
            frac = (energy - e_low) / (e_high - e_low)
            
            # Get the cosine values at the two bounding energies
            cosines_low = self.cosine_bins[idx-1]
            cosines_high = self.cosine_bins[idx]
            
            # Interpolate cosine bin boundaries
            bin_values = [(1-frac)*cl + frac*ch for cl, ch in zip(cosines_low, cosines_high)]
        
        if not interpolate:
            # Return the actual bin boundaries and their probabilities
            # For equiprobable bins, each bin has probability 1/32
            prob_per_bin = 1.0 / 32.0
            
            # Calculate probability density for each bin (constant within bin)
            pdf_values = []
            bin_centers = []
            bin_lows = []
            bin_highs = []
            energy_values = []
            
            for i in range(len(bin_values) - 1):
                bin_width = bin_values[i+1] - bin_values[i]
                if bin_width > 0:
                    pdf = prob_per_bin / bin_width
                else:
                    pdf = 0.0
                
                # Use bin center as the cosine value
                bin_center = (bin_values[i] + bin_values[i+1]) / 2
                
                bin_centers.append(bin_center)
                pdf_values.append(pdf)
                bin_lows.append(bin_values[i])
                bin_highs.append(bin_values[i+1])
                energy_values.append(energy)
            
            # Verify all arrays have the same length
            array_lengths = [len(bin_centers), len(pdf_values), len(bin_lows), len(bin_highs), len(energy_values)]
            if len(set(array_lengths)) > 1:
                # If lengths don't match, truncate to the shortest length
                min_length = min(array_lengths)
                bin_centers = bin_centers[:min_length]
                pdf_values = pdf_values[:min_length]
                bin_lows = bin_lows[:min_length]
                bin_highs = bin_highs[:min_length]
                energy_values = energy_values[:min_length]
            
            return pd.DataFrame({
                'energy': energy_values,
                'cosine': bin_centers,
                'pdf': pdf_values,
                'bin_low': bin_lows,
                'bin_high': bin_highs
            })
        
        # If interpolation requested, use the existing code
        # Generate a fine cosine grid
        cosines = np.linspace(-1, 1, num_points)
        
        # Calculate PDF (should be constant within each bin)
        # For a 32-bin equiprobable distribution, each bin has probability of 1/32
        prob_per_bin = 1.0 / 32.0
        
        # Initialize PDF array
        pdf_values = np.zeros_like(cosines)
        
        # Assign PDF values based on bin membership
        for i, mu in enumerate(cosines):
            # Find which bin the cosine falls into
            bin_idx = 0
            while bin_idx < 32 and bin_values[bin_idx] <= mu:
                bin_idx += 1
            
            if bin_idx > 0 and bin_idx <= 32:
                bin_width = bin_values[bin_idx] - bin_values[bin_idx-1]
                if bin_width > 0:
                    pdf_values[i] = prob_per_bin / bin_width
        
        return pd.DataFrame({
            'energy': np.full_like(cosines, energy, dtype=float),
            'cosine': cosines,
            'pdf': pdf_values
        })

    def __repr__(self) -> str:
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
            "# Create a histogram-style plot of the distribution at 2 MeV\n"
            "fig, ax = distribution.plot(energy=2.0)\n"
        )
        
        return header + description + info_table + properties_section + "\n" + methods_section + "\n" + example