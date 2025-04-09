from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.classes.angular_distribution.types import AngularDistributionType
from mcnpy.ace.classes.angular_distribution.angular_distribution_repr import angular_distribution_repr


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
    
    def sample_mu(self, energy: float, random_value: float) -> float:
        """
        Sample a scattering cosine μ at the given energy using the provided random value.
        
        Parameters
        ----------
        energy : float
            Incident energy
        random_value : float
            Random number between 0 and 1
            
        Returns
        -------
        float
            Sampled cosine value μ
        """
        # Base implementation just returns isotropic distribution
        return 2.0 * random_value - 1.0
    
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
    
    __repr__ = angular_distribution_repr
