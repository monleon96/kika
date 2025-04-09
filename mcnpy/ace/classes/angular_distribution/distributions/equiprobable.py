from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.classes.angular_distribution.base import AngularDistribution
from mcnpy.ace.classes.angular_distribution.types import AngularDistributionType
from mcnpy.ace.classes.angular_distribution.angular_distribution_repr import equiprobable_distribution_repr


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
            # Below first energy, use first set of bins
            cosine_values = self.cosine_bins[0]
        elif idx >= len(energy_values):
            # Above last energy, use last set of bins
            cosine_values = self.cosine_bins[-1]
        else:
            # Interpolate between energy points
            e_low = energy_values[idx-1]
            e_high = energy_values[idx]
            frac = (energy - e_low) / (e_high - e_low)
            
            cosines_low = self.cosine_bins[idx-1]
            cosines_high = self.cosine_bins[idx]
            
            # Interpolate cosine values
            cosine_values = [(1-frac)*cl + frac*ch for cl, ch in zip(cosines_low, cosines_high)]
        
        # Select the appropriate bin
        bin_idx = min(int(32 * random_value), 31)
        mu_low = cosine_values[bin_idx]
        mu_high = cosine_values[bin_idx+1]
        
        # Linearly interpolate within the bin
        frac_in_bin = 32 * random_value - bin_idx
        return (1-frac_in_bin) * mu_low + frac_in_bin * mu_high

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

    __repr__ = equiprobable_distribution_repr