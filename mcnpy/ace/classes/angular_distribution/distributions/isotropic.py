from typing import Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from mcnpy.ace.classes.angular_distribution.base import AngularDistribution
from mcnpy.ace.classes.angular_distribution.types import AngularDistributionType
from mcnpy.ace.classes.angular_distribution.angular_distribution_repr import isotropic_distribution_repr

@dataclass
class IsotropicAngularDistribution(AngularDistribution):
    """Angular distribution for isotropic scattering."""
    
    def __post_init__(self):
        super().__post_init__()
        self.distribution_type = AngularDistributionType.ISOTROPIC
    
    def to_dataframe(self, energy: float, num_points: int = 100, interpolate: bool = False) -> Optional[pd.DataFrame]:
        """
        Convert isotropic angular distribution to a pandas DataFrame.
        
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
            
        # For isotropic distribution, the PDF is constant (0.5) for all cosines
        if interpolate:
            cosines = np.linspace(-1, 1, num_points)
            return pd.DataFrame({
                'energy': np.full_like(cosines, energy, dtype=float),
                'cosine': cosines,
                'pdf': np.full_like(cosines, 0.5, dtype=float)
            })
        else:
            # Just return points at the ends of the cosine range for efficiency
            return pd.DataFrame({
                'energy': [energy, energy],
                'cosine': [-1.0, 1.0],
                'pdf': [0.5, 0.5]
            })

    __repr__ = isotropic_distribution_repr