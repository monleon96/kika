from typing import List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from mcnpy.ace.classes.xss import XssEntry
from mcnpy.ace.classes.angular_distribution.base import AngularDistribution
from mcnpy.ace.classes.angular_distribution.types import AngularDistributionType
from mcnpy._utils import create_repr_section


@dataclass
class TabulatedAngularDistribution(AngularDistribution):
    """Angular distribution for tabulated scattering."""
    interpolation: List[int] = field(default_factory=list)  # Interpolation flag for each energy
    _cosine_grid: List[List[XssEntry]] = field(default_factory=list)  # Cosine grid for each energy
    _pdf: List[List[XssEntry]] = field(default_factory=list)  # PDF for each energy
    _cdf: List[List[XssEntry]] = field(default_factory=list)  # CDF for each energy
    
    def __post_init__(self):
        super().__post_init__()
        self.distribution_type = AngularDistributionType.TABULATED
    
    @property
    def cosine_grid(self) -> List[List[float]]:
        """Get cosine grid values as lists of floats."""
        return [[c.value for c in cosine_list] for cosine_list in self._cosine_grid]
    
    @property
    def pdf(self) -> List[List[float]]:
        """Get PDF values as lists of floats."""
        return [[p.value for p in pdf_list] for pdf_list in self._pdf]
    
    @property
    def cdf(self) -> List[List[float]]:
        """Get CDF values as lists of floats."""
        return [[c.value for c in cdf_list] for cdf_list in self._cdf]
    
    
    def to_dataframe(self, energy: float, num_points: int = 100, interpolate: bool = False) -> Optional[pd.DataFrame]:
        """
        Convert tabulated angular distribution to a pandas DataFrame.
        
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
            DataFrame with 'energy', 'cosine', 'pdf', and potentially 'cdf' columns
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
                    'pdf': [0.5, 0.5],
                    'cdf': [0.0, 1.0]
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
                    'pdf': [0.5, 0.5],
                    'cdf': [0.0, 1.0]
                })
        
        # Find bounding energy indices
        energy_values = self.energies
        idx = np.searchsorted(energy_values, energy)
        
        # Get appropriate PDF/CDF values based on energy
        if idx == 0:
            cosine_values = self.cosine_grid[0]
            pdf_values = self.pdf[0]
            cdf_values = self.cdf[0]
        elif idx >= len(energy_values):
            cosine_values = self.cosine_grid[-1]
            pdf_values = self.pdf[-1]
            cdf_values = self.cdf[-1]
        else:
            # Interpolate between energy points
            e_low = energy_values[idx-1]
            e_high = energy_values[idx]
            frac = (energy - e_low) / (e_high - e_low)
            
            # Get values at the two bounding energies
            cosines_low = self.cosine_grid[idx-1]
            cosines_high = self.cosine_grid[idx]
            pdf_low = self.pdf[idx-1]
            pdf_high = self.pdf[idx]
            cdf_low = self.cdf[idx-1]
            cdf_high = self.cdf[idx]
            
            # If cosine grids are different between the two energy points,
            # interpolate onto a common grid to ensure consistent lengths
            if len(cosines_low) != len(cosines_high):
                # For simplicity, use a common grid based on the lower energy point
                cosine_values = cosines_low
                # Interpolate PDF and CDF from the high energy point onto the lower energy grid
                pdf_high_interp = np.interp(
                    cosine_values, cosines_high, pdf_high, 
                    left=pdf_high[0], right=pdf_high[-1]
                )
                cdf_high_interp = np.interp(
                    cosine_values, cosines_high, cdf_high, 
                    left=0.0, right=1.0
                )
                # Linear interpolation between the two energy points
                pdf_values = [(1-frac)*pl + frac*ph for pl, ph in zip(pdf_low, pdf_high_interp)]
                cdf_values = [(1-frac)*cl + frac*ch for cl, ch in zip(cdf_low, cdf_high_interp)]
            else:
                # Simple linear interpolation when grids are the same
                cosine_values = cosines_low  # Use the lower energy cosine grid
                pdf_values = [(1-frac)*pl + frac*ph for pl, ph in zip(pdf_low, pdf_high)]
                cdf_values = [(1-frac)*cl + frac*ch for cl, ch in zip(cdf_low, cdf_high)]
        
        if not interpolate:
            # Convert values to numpy arrays to ensure consistency and check lengths
            cosine_np = np.array(cosine_values)
            pdf_np = np.array(pdf_values)
            cdf_np = np.array(cdf_values)
            
            # Make sure all arrays have the same length - if not, use the shortest length
            min_length = min(len(cosine_np), len(pdf_np), len(cdf_np))
            if min_length < len(cosine_np):
                cosine_np = cosine_np[:min_length]
            if min_length < len(pdf_np):
                pdf_np = pdf_np[:min_length]
            if min_length < len(cdf_np):
                cdf_np = cdf_np[:min_length]
            
            # Create DataFrame with consistent array lengths
            df = pd.DataFrame({
                'energy': np.full_like(cosine_np, energy, dtype=float),
                'cosine': cosine_np,
                'pdf': pdf_np,
                'cdf': cdf_np
            })
            return df
        
        # If interpolation requested, create a regular grid and interpolate
        cosines = np.linspace(-1, 1, num_points)
        
        # Get interpolation type for this energy
        interp_type = None
        if idx < len(self.interpolation):
            interp_type = self.interpolation[idx]
        elif self.interpolation:
            interp_type = self.interpolation[-1]

        # Generate PDF values based on interpolation type
        if interpolate:
            pdf_values_interp = np.zeros_like(cosines)
            
            if interp_type == 1:  # Histogram
                # For histogram, each point gets the value of the left edge of its bin
                for i, mu in enumerate(cosines):
                    bin_idx = np.searchsorted(cosine_values, mu)
                    if bin_idx == 0:
                        pdf_values_interp[i] = pdf_values[0]
                    else:
                        pdf_values_interp[i] = pdf_values[bin_idx-1]
            else:  # Linear-linear (type 2) or fallback
                # Use linear interpolation (current implementation)
                pdf_values_interp = np.interp(cosines, cosine_values, pdf_values, left=0.0, right=0.0)
        
        return pd.DataFrame({
            'energy': np.full_like(cosines, energy, dtype=float),
            'cosine': cosines,
            'pdf': pdf_values_interp
        })

    def __repr__(self) -> str:
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
        
        # Methods section - Update to remove sample_mu reference
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
        
        # Add example for using this specific distribution type - Update to remove sample_mu usage
        example = (
            "Example:\n"
            "--------\n"
            "# Directly access raw data for the first energy point\n"
            "first_energy = distribution.energies[0]  # Returns a float\n"
            "interp_flag = distribution.interpolation[0]  # 1=histogram, 2=linear-linear\n"
            "cosines = distribution.cosine_grid[0]  # List of floats for cosine values\n"
            "pdf_values = distribution.pdf[0]     # PDF values\n"
            "cdf_values = distribution.cdf[0]     # CDF values\n\n"
            "# Create a plot showing the PDF at 2 MeV\n"
            "fig, ax = distribution.plot(energy=2.0)\n"
        )
        
        return header + description + info_table + properties_section + "\n" + methods_section + "\n" + example
