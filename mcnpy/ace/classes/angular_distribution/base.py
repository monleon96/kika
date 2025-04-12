from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from mcnpy.ace.classes.xss import XssEntry
from mcnpy.ace.classes.angular_distribution.types import AngularDistributionType
from mcnpy._utils import create_repr_section


@dataclass
class AngularDistribution:
    """Base class for angular distributions."""
    mt: XssEntry = None  # MT number for this reaction
    _energies: List[XssEntry] = field(default_factory=list)  # Energy grid with XssEntry objects
    distribution_type: AngularDistributionType = AngularDistributionType.ISOTROPIC
    
    def __post_init__(self):
        """Initialize after creation, ensuring values are properly stored."""
        # Convert XssEntry to value if needed
        if hasattr(self.mt, 'value'):
            self.mt = int(self.mt.value)
    
    @property
    def energies(self) -> List[float]:
        """Get energy values as floats."""
        return [entry.value for entry in self._energies]
    
    @property
    def is_isotropic(self) -> bool:
        """Check if the distribution is isotropic."""
        return self.distribution_type == AngularDistributionType.ISOTROPIC

    def to_dataframe(self, energy: float, num_points: int = 100, interpolate: bool = False) -> Optional[pd.DataFrame]:
        """
        Convert angular distribution to a pandas DataFrame.
        
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
            DataFrame with 'energy', 'cosine', and 'pdf' columns
            Returns None if pandas is not available
        """
        try:
            import pandas as pd
            
            # For isotropic distribution, return simple DataFrame
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
            
        except ImportError:
            return None

    def plot(self, energy: float, ax=None, title=None, **kwargs) -> Optional[Tuple]:
        """
        Plot the angular distribution for a specific incident energy.
        
        Parameters
        ----------
        energy : float
            Incident energy to evaluate the distribution at
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        title : str, optional
            Title for the plot, if None a default title is used
        **kwargs : dict
            Additional keyword arguments passed to the plot function
            
        Returns
        -------
        tuple or None
            Tuple of (fig, ax) or None if matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get the data to plot
            df = self.to_dataframe(energy)
            
            # Create figure and axes if not provided
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.figure
            
            # Set default parameters if not specified
            if 'linewidth' not in kwargs:
                kwargs['linewidth'] = 2
            if 'color' not in kwargs:
                kwargs['color'] = 'blue'
            
            # Plot the data
            ax.plot(df['cosine'], df['pdf'], **kwargs)
            
            # Set labels and title
            ax.set_xlabel('Cosine (μ)')
            ax.set_ylabel('Probability Density')
            
            if title is None:
                mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
                title = f'Angular Distribution for MT={mt_value} at {energy:.4g} MeV'
            ax.set_title(title)
            
            # Set axis limits
            ax.set_xlim(-1, 1)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            return fig, ax
        except ImportError:
            return None
    
    def __repr__(self) -> str:
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
