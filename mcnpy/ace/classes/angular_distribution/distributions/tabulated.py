from typing import List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.classes.angular_distribution.base import AngularDistribution
from mcnpy.ace.classes.angular_distribution.types import AngularDistributionType
from mcnpy.ace.classes.angular_distribution.angular_distribution_repr import tabulated_distribution_repr

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
        # If energy is outside our range, use isotropic scattering
        if not self._energies or energy < self._energies[0].value or energy > self._energies[-1].value:
            return 2.0 * random_value - 1.0
        
        # Find bounding energy indices
        energy_values = self.energies
        idx = np.searchsorted(energy_values, energy)
        if idx == 0:
            # Below first energy, use first distribution
            cosine_values = self.cosine_grid[0]
            cdf_values = self.cdf[0]
        elif idx >= len(energy_values):
            # Above last energy, use last distribution
            cosine_values = self.cosine_grid[-1]
            cdf_values = self.cdf[-1]
        else:
            # For now, just use the lower energy point
            # In practice, we should interpolate between distributions
            cosine_values = self.cosine_grid[idx-1]
            cdf_values = self.cdf[idx-1]
        
        # Find the bin that contains the random value
        bin_idx = np.searchsorted(cdf_values, random_value)
        if bin_idx == 0:
            return cosine_values[0]
        elif bin_idx >= len(cdf_values):
            return cosine_values[-1]
        
        # Linearly interpolate within the bin
        cdf_low = cdf_values[bin_idx-1]
        cdf_high = cdf_values[bin_idx]
        frac = (random_value - cdf_low) / (cdf_high - cdf_low)
        
        mu_low = cosine_values[bin_idx-1]
        mu_high = cosine_values[bin_idx]

        # Get the interpolation flag for this energy
        energy_idx = idx if idx < len(energy_values) else len(energy_values)-1
        interp_type = self.interpolation[energy_idx]

        # Sample based on interpolation type
        if interp_type == 1:  # Histogram
            # For histogram, return the left edge of the bin
            return cosine_values[bin_idx-1]
        else:  # Linear-linear (type 2) or fallback
            # Linear interpolation within the bin (current implementation)
            frac = (random_value - cdf_low) / (cdf_high - cdf_low)
            return (1-frac) * mu_low + frac * mu_high

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

    __repr__ = tabulated_distribution_repr