from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any, KeysView
import numpy as np
from enum import Enum
import pandas as pd
from mcnpy.ace.xss import XssEntry
from mcnpy.ace.classes.angular_distribution.angular_distribution_repr import (
    angular_distribution_repr, 
    isotropic_distribution_repr,
    equiprobable_distribution_repr,
    tabulated_distribution_repr,
    kalbach_mann_distribution_repr,
    angular_container_repr
)


class ErrorMessageDict(dict):
    """Dictionary that provides helpful error messages when accessing non-existent keys."""
    
    def __init__(self, *args, dict_name="Dictionary", **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_name = dict_name
    
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            # Create a user-friendly error message with available keys
            available_keys = sorted(self.keys_as_int())
            
            # Format the error message without newlines to avoid raw \n in output
            if available_keys:
                error_msg = f"Error: MT={key} not found in {self.dict_name}. Available MT numbers: {available_keys}"
            else:
                error_msg = f"Error: MT={key} not found in {self.dict_name}. No MT numbers available in this collection."
            
            raise KeyError(error_msg)
    
    def keys_as_int(self):
        """Get keys as integers for better display, handling XssEntry objects."""
        result = []
        for key in self.keys():
            if isinstance(key, XssEntry):
                result.append(int(key.value))
            else:
                result.append(int(key))
        return result


class ErrorMessageList(list):
    """List that provides helpful error messages when accessing non-existent indices."""
    
    def __init__(self, *args, list_name="List", **kwargs):
        super().__init__(*args, **kwargs)
        self.list_name = list_name
    
    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except IndexError:
            # Create a user-friendly error message with available indices
            if len(self) > 0:
                error_msg = f"Error: Particle index {idx} not found in {self.list_name}. Available particle indices: {list(range(len(self)))}"
                
                # Add particle counts information in a single line if available
                counts_info = []
                for i, particle_data in enumerate(self):
                    counts_info.append(f"Index {i}: {len(particle_data)} reactions")
                
                if counts_info:
                    error_msg += f" (Particle counts: {', '.join(counts_info)})"
            else:
                error_msg = f"Error: Particle index {idx} not found in {self.list_name}. No particle production data available."
            
            raise IndexError(error_msg)


class AngularDistributionType(Enum):
    """Enumeration of angular distribution types."""
    ISOTROPIC = 0
    EQUIPROBABLE = 1
    TABULATED = 2
    KALBACH_MANN = 3  # Law=44 distributions


@dataclass
class AngularDistribution:
    """Base class for angular distributions."""
    mt: XssEntry = None  # MT number for this reaction
    energies: List[XssEntry] = field(default_factory=list)  # Energy grid
    distribution_type: AngularDistributionType = AngularDistributionType.ISOTROPIC
    
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

    def to_dataframe(self, energy: float, num_points: int = 100) -> Optional[pd.DataFrame]:
        """
        Convert angular distribution to a pandas DataFrame for a specific incident energy.
        
        Parameters
        ----------
        energy : float
            Incident energy to evaluate the distribution at
        num_points : int, optional
            Number of angular points to generate, defaults to 100
            
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with 'cosine' and 'probability' columns, or None if pandas is not available
        """
        try:
            import pandas as pd
            
            # Create cosine grid from -1 to 1
            cosines = np.linspace(-1, 1, num_points)
            
            # For the base class, just return isotropic distribution
            probabilities = np.ones_like(cosines) * 0.5  # Uniform probability
            
            return pd.DataFrame({
                'cosine': cosines,
                'probability': probabilities
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
            ax.plot(df['cosine'], df['probability'], **kwargs)
            
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


@dataclass
class IsotropicAngularDistribution(AngularDistribution):
    """Angular distribution for isotropic scattering."""
    
    def __post_init__(self):
        self.distribution_type = AngularDistributionType.ISOTROPIC

    __repr__ = isotropic_distribution_repr


@dataclass
class EquiprobableAngularDistribution(AngularDistribution):
    """Angular distribution for 32 equiprobable bin scattering."""
    cosine_bins: List[List[XssEntry]] = field(default_factory=list)  # List of 33 cosines for each energy
    
    def __post_init__(self):
        self.distribution_type = AngularDistributionType.EQUIPROBABLE
    
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
        if not self.energies or energy < self.energies[0].value or energy > self.energies[-1].value:
            return 2.0 * random_value - 1.0
        
        # Find bounding energy indices
        energy_values = [e.value for e in self.energies]
        idx = np.searchsorted(energy_values, energy)
        if idx == 0:
            # Below first energy, use first set of bins
            cosines = self.cosine_bins[0]
        elif idx >= len(self.energies):
            # Above last energy, use last set of bins
            cosines = self.cosine_bins[-1]
        else:
            # Interpolate between energy points
            e_low = energy_values[idx-1]
            e_high = energy_values[idx]
            frac = (energy - e_low) / (e_high - e_low)
            
            cosines_low = self.cosine_bins[idx-1]
            cosines_high = self.cosine_bins[idx]
            
            # Get interpolated cosine values
            cosines_values_low = [c.value for c in cosines_low]
            cosines_values_high = [c.value for c in cosines_high]
            cosines_values = [(1-frac)*cl + frac*ch for cl, ch in zip(cosines_values_low, cosines_values_high)]
            
            # Convert back to XssEntry for consistency
            cosines = [XssEntry(0, val) for val in cosines_values]
        
        # Get values for computation
        cosine_values = [c.value for c in cosines]
        
        # Select the appropriate bin
        bin_idx = min(int(32 * random_value), 31)
        mu_low = cosine_values[bin_idx]
        mu_high = cosine_values[bin_idx+1]
        
        # Linearly interpolate within the bin
        frac_in_bin = 32 * random_value - bin_idx
        return (1-frac_in_bin) * mu_low + frac_in_bin * mu_high

    def to_dataframe(self, energy: float, num_points: int = 100) -> Optional[pd.DataFrame]:
        """
        Convert equiprobable bin distribution to a pandas DataFrame for a specific incident energy.
        
        Parameters
        ----------
        energy : float
            Incident energy to evaluate the distribution at
        num_points : int, optional
            Number of angular points to generate, defaults to 100
            
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with 'cosine' and 'probability' columns, or None if pandas is not available
        """
        try:
            import pandas as pd
            
            # Generate a fine cosine grid
            cosines = np.linspace(-1, 1, num_points)
            
            # If energy is outside our range, return uniform distribution
            if not self.energies or energy < self.energies[0].value or energy > self.energies[-1].value:
                return pd.DataFrame({
                    'cosine': cosines,
                    'probability': np.ones_like(cosines) * 0.5
                })
            
            # Find bounding energy indices
            energy_values = [e.value for e in self.energies]
            idx = np.searchsorted(energy_values, energy)
            
            # Get appropriate cosine bins based on energy
            if idx == 0:
                bin_values = [c.value for c in self.cosine_bins[0]]
            elif idx >= len(self.energies):
                bin_values = [c.value for c in self.cosine_bins[-1]]
            else:
                # Interpolate between energy points
                e_low = energy_values[idx-1]
                e_high = energy_values[idx]
                frac = (energy - e_low) / (e_high - e_low)
                
                # Get the cosine values at the two bounding energies
                cosines_low = [c.value for c in self.cosine_bins[idx-1]]
                cosines_high = [c.value for c in self.cosine_bins[idx]]
                
                # Interpolate cosine bin boundaries
                bin_values = [(1-frac)*cl + frac*ch for cl, ch in zip(cosines_low, cosines_high)]
            
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
                'cosine': cosines,
                'probability': pdf_values
            })
        except ImportError:
            return None
    
    __repr__ = equiprobable_distribution_repr


@dataclass
class TabulatedAngularDistribution(AngularDistribution):
    """Angular distribution for tabulated scattering."""
    interpolation: List[int] = field(default_factory=list)  # Interpolation flag for each energy
    cosine_grid: List[List[XssEntry]] = field(default_factory=list)  # Cosine grid for each energy
    pdf: List[List[XssEntry]] = field(default_factory=list)  # PDF for each energy
    cdf: List[List[XssEntry]] = field(default_factory=list)  # CDF for each energy
    
    def __post_init__(self):
        self.distribution_type = AngularDistributionType.TABULATED
    
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
        if not self.energies or energy < self.energies[0].value or energy > self.energies[-1].value:
            return 2.0 * random_value - 1.0
        
        # Find bounding energy indices
        energy_values = [e.value for e in self.energies]
        idx = np.searchsorted(energy_values, energy)
        if idx == 0:
            # Below first energy, use first distribution
            cosines = self.cosine_grid[0]
            cdfs = self.cdf[0]
        elif idx >= len(self.energies):
            # Above last energy, use last distribution
            cosines = self.cosine_grid[-1]
            cdfs = self.cdf[-1]
        else:
            # For now, just use the lower energy point
            # In practice, we should interpolate between distributions
            cosines = self.cosine_grid[idx-1]
            cdfs = self.cdf[idx-1]
        
        # Extract values for computation
        cosine_values = [c.value for c in cosines]
        cdf_values = [c.value for c in cdfs]
        
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
        
        return (1-frac) * mu_low + frac * mu_high

    def to_dataframe(self, energy: float, num_points: int = 100) -> Optional[pd.DataFrame]:
        """
        Convert tabulated angular distribution to a pandas DataFrame for a specific incident energy.
        
        Parameters
        ----------
        energy : float
            Incident energy to evaluate the distribution at
        num_points : int, optional
            Number of angular points to generate, defaults to 100
            
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with 'cosine' and 'probability' columns, or None if pandas is not available
        """
        try:
            import pandas as pd
            
            # Generate a fine cosine grid
            cosines = np.linspace(-1, 1, num_points)
            
            # If energy is outside our range, return uniform distribution
            if not self.energies or energy < self.energies[0].value or energy > self.energies[-1].value:
                return pd.DataFrame({
                    'cosine': cosines,
                    'probability': np.ones_like(cosines) * 0.5
                })
            
            # Find bounding energy indices
            energy_values = [e.value for e in self.energies]
            idx = np.searchsorted(energy_values, energy)
            
            # Get appropriate PDF values based on energy
            if idx == 0:
                cosine_values = [c.value for c in self.cosine_grid[0]]
                pdf_values = [p.value for p in self.pdf[0]]
            elif idx >= len(self.energies):
                cosine_values = [c.value for c in self.cosine_grid[-1]]
                pdf_values = [p.value for p in self.pdf[-1]]
            else:
                # For now, just use the lower energy point (same as in sample_mu)
                cosine_values = [c.value for c in self.cosine_grid[idx-1]]
                pdf_values = [p.value for p in self.pdf[idx-1]]
            
            # Interpolate PDF values onto the fine cosine grid
            interp_pdf = np.interp(cosines, cosine_values, pdf_values, left=0.0, right=0.0)
            
            return pd.DataFrame({
                'cosine': cosines,
                'probability': interp_pdf
            })
        except ImportError:
            return None
    
    __repr__ = tabulated_distribution_repr


@dataclass
class KalbachMannAngularDistribution(AngularDistribution):
    """
    Angular distribution using Law=44 (Kalbach-Mann) from the DLW/DLWH Block.
    This distribution is correlated with energy and uses the Kalbach-Mann formalism.
    The actual angular distribution data is stored in the energy-angle distribution
    section of the ACE file (DLW/DLWH blocks).
    """
    # Reference to the reaction index in the DLW/DLWH block
    reaction_index: int = -1
    # Whether this is a particle production reaction
    is_particle_production: bool = False
    # Particle index (only used if is_particle_production=True)
    particle_idx: int = -1
    
    def __post_init__(self):
        self.distribution_type = AngularDistributionType.KALBACH_MANN
    
    def _find_law44_distribution(self, ace):
        """
        Find the Law=44 (Kalbach-Mann) distribution in the energy distribution container.
        
        Parameters
        ----------
        ace : Ace
            ACE object containing the distribution data
            
        Returns
        -------
        KalbachMannDistribution or None
            The Law=44 distribution or None if not found
        """
        if ace is None or ace.energy_distributions is None:
            return None
            
        # Get MT number
        mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
        
        # Get the appropriate container based on the particle type
        if self.is_particle_production:
            # Get particle production distributions
            if (self.particle_idx < 0 or 
                self.particle_idx >= len(ace.energy_distributions.particle_production)):
                return None
                
            # Get distributions for this MT
            distributions = ace.energy_distributions.get_particle_distribution(
                self.particle_idx, mt_value)
        else:
            # Get incident neutron distributions
            distributions = ace.energy_distributions.get_neutron_distribution(mt_value)
            
        if not distributions:
            return None
            
        # Find the Law=44 distribution
        for dist in distributions:
            if dist.law == 44:  # Law=44 is Kalbach-Mann
                return dist
                
        return None
    
    def sample_mu(self, energy: float, random_value: float, ace=None) -> float:
        """
        Sample a scattering cosine μ using the Kalbach-Mann formalism.
        
        For Kalbach-Mann distributions, the angular sampling requires access 
        to the R and A parameters from the Law=44 distribution in the DLW/DLWH block.
        This method requires the ACE object to be passed to access that data.
        
        Parameters
        ----------
        energy : float
            Incident energy
        random_value : float
            Random number between 0 and 1
        ace : Ace, optional
            ACE object containing the distribution data
            
        Returns
        -------
        float
            Sampled cosine value μ (default to isotropic if ACE data not available)
        """
        # If no ACE data is available or the reaction index is invalid, use isotropic
        if ace is None or self.reaction_index < 0:
            return 2.0 * random_value - 1.0
            
        # Find the Law=44 distribution
        km_dist = self._find_law44_distribution(ace)
        if km_dist is None:
            # Fallback to isotropic if distribution not found
            return 2.0 * random_value - 1.0
            
        # First, we sample an outgoing energy from the distribution
        # For a full implementation, we'd use a separate random number for energy
        # Here we'll just use a random energy from the distribution's table 
        dist = km_dist.get_interpolated_distribution(energy)
        if not dist:
            return 2.0 * random_value - 1.0
            
        # Select a random energy point to get R and A parameters
        # In practice, this energy would come from the energy sampling step
        # which is correlated with the angular sampling
        idx = min(int(random_value * (len(dist['e_out']) - 1)), len(dist['e_out']) - 1)
            
        # Get the R and A parameters for this energy point
        r_value = dist['r'][idx]
        a_value = dist['a'][idx]
            
        # Sample from the Kalbach-Mann distribution
        # If a is very small, return isotropic
        if abs(a_value) < 1.0e-3:
            return 2.0 * random_value - 1.0
            
        # Use the sampling algorithm from the KalbachMannDistribution class
        return self._sample_kalbach_mann(a_value, r_value, random_value)
    
    def _sample_kalbach_mann(self, a: float, r: float, random_value: float) -> float:
        """
        Sample a cosine from the Kalbach-Mann angular distribution.
        
        p(μ) = (1/2)*(a/sinh(a))*[cosh(aμ) + r*sinh(aμ)]
        
        Parameters
        ----------
        a : float
            Angular distribution slope parameter
        r : float
            Precompound fraction parameter
        random_value : float
            Random number between 0 and 1
            
        Returns
        -------
        float
            Sampled cosine value in [-1, 1]
        """
        # This is a direct implementation that works well for sampling
        # a single μ value when we already have specific r and a parameters
        
        # First, calculate the conditional probability function parameters
        if abs(a) < 1.0e-3:
            # For very small a, return isotropic distribution
            return 2.0 * random_value - 1.0
            
        # Calculate the normalization factor
        sinh_a = np.sinh(a)
        
        # Direct sampling for r = 0 is simple
        if abs(r) < 1.0e-5:
            # For r ≈ 0, we can use a simpler formula:
            # μ = (1/a) * log[exp(-a) + 2*random_value*sinh(a)]
            return (1.0/a) * np.log(np.exp(-a) + 2.0 * random_value * sinh_a)
        
        # For the general case, we need to solve for μ in:
        # CDF(μ) = (cosh(aμ) - cosh(-a) + r*(sinh(aμ) - sinh(-a))) / 
        #          (2*sinh(a) + r*(cosh(a) - cosh(-a)))
        
        # Simplified form for practical use: 
        # CDF(μ) = ((1-r)*sinh(a*(μ+1))/2 + r*(cosh(a*(μ+1))-1)/2) / 
        #          ((1-r)*sinh(a) + r*(cosh(a)-1))
        
        # Solve this numerically using a simplified approach
        # For production use, a more accurate method like Newton-Raphson would be better
        
        # Define CDF(μ) function for values from -1 to +1
        denominator = (1.0 - r) * sinh_a + r * (np.cosh(a) - 1.0)
        
        # Binary search for the μ value that gives the target CDF
        left = -1.0
        right = 1.0
        target = random_value
        
        for _ in range(20):  # Usually converges in < 20 iterations
            mid = (left + right) / 2.0
            
            # Calculate CDF at midpoint
            sinh_term = np.sinh(a * (mid + 1.0)) / 2.0
            cosh_term = (np.cosh(a * (mid + 1.0)) - 1.0) / 2.0
            cdf_mid = ((1.0 - r) * sinh_term + r * cosh_term) / denominator
            
            if abs(cdf_mid - target) < 1.0e-6:
                return mid
            elif cdf_mid < target:
                left = mid
            else:
                right = mid
                
        return (left + right) / 2.0  # Return final midpoint as the answer

    def to_dataframe(self, energy: float, ace=None, num_points: int = 100) -> Optional[pd.DataFrame]:
        """
        Convert Kalbach-Mann angular distribution to a pandas DataFrame for a specific incident energy.
        
        Parameters
        ----------
        energy : float
            Incident energy to evaluate the distribution at
        ace : Ace, optional
            ACE object containing the Law=44 distribution data
        num_points : int, optional
            Number of angular points to generate, defaults to 100
            
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with 'cosine' and 'probability' columns, or None if pandas is not available
        """
        try:
            import pandas as pd
            
            # Generate a fine cosine grid
            cosines = np.linspace(-1, 1, num_points)
            
            if ace is None:
                # If no ACE data is available, return isotropic distribution
                return pd.DataFrame({
                    'cosine': cosines,
                    'probability': np.ones_like(cosines) * 0.5
                })
            
            # Find the Law=44 distribution
            km_dist = self._find_law44_distribution(ace)
            if km_dist is None:
                # If no Law=44 distribution found, return isotropic
                return pd.DataFrame({
                    'cosine': cosines,
                    'probability': np.ones_like(cosines) * 0.5
                })
            
            # Get the interpolated distribution for this energy
            dist = km_dist.get_interpolated_distribution(energy)
            if not dist:
                # If no distribution found, return isotropic
                return pd.DataFrame({
                    'cosine': cosines,
                    'probability': np.ones_like(cosines) * 0.5
                })
            
            # Verify that we have e_out, r, and a data and they're non-empty
            if ('e_out' not in dist or 'r' not in dist or 'a' not in dist or 
                not dist['e_out'] or not dist['r'] or not dist['a']):
                # Missing or empty required data, return isotropic
                return pd.DataFrame({
                    'cosine': cosines,
                    'probability': np.ones_like(cosines) * 0.5
                })
            
            # Convert all lengths to integers explicitly to avoid type issues
            e_out_len = int(len(dist['e_out'])) if isinstance(dist['e_out'], list) else 0
            r_len = int(len(dist['r'])) if isinstance(dist['r'], list) else 0
            a_len = int(len(dist['a'])) if isinstance(dist['a'], list) else 0
            
            # Make sure r and a arrays are at least as long as e_out
            if r_len < e_out_len or a_len < e_out_len:
                # Arrays have inconsistent lengths, return isotropic
                return pd.DataFrame({
                    'cosine': cosines,
                    'probability': np.ones_like(cosines) * 0.5
                })
            
            # For simplicity, use the R and A parameters from the middle of the E_out range
            middle_idx = min(e_out_len // 2, r_len - 1, a_len - 1)
            
            # Extract values with explicit conversion to float
            r_value = dist['r'][middle_idx]
            a_value = dist['a'][middle_idx]
            
            # Handle XssEntry objects if present
            r_value = float(r_value.value if hasattr(r_value, 'value') else r_value)
            a_value = float(a_value.value if hasattr(a_value, 'value') else a_value)
            
            # Calculate PDF values for the Kalbach-Mann distribution
            probabilities = np.zeros_like(cosines)
            
            # If a is very small, use isotropic distribution
            if abs(a_value) < 1.0e-3:
                probabilities.fill(0.5)
            else:
                # Calculate Kalbach-Mann PDF: p(μ) = (a/2)/sinh(a) * [cosh(aμ) + r*sinh(aμ)]
                sinh_a = np.sinh(a_value)
                normalization = (a_value / 2.0) / sinh_a
                
                for i, mu in enumerate(cosines):
                    probabilities[i] = normalization * (
                        np.cosh(a_value * mu) + r_value * np.sinh(a_value * mu)
                    )
            
            return pd.DataFrame({
                'cosine': cosines,
                'probability': probabilities
            })
        except ImportError:
            return None
        except Exception as e:
            # Log the error or handle it as appropriate
            print(f"Error in KalbachMannAngularDistribution.to_dataframe: {e}")
            # Return isotropic as fallback
            try:
                return pd.DataFrame({
                    'cosine': cosines,
                    'probability': np.ones_like(cosines) * 0.5
                })
            except:
                return None
    
    def plot(self, energy: float, ace=None, ax=None, title=None, **kwargs) -> Optional[Tuple]:
        """
        Plot the Kalbach-Mann angular distribution for a specific incident energy.
        
        Parameters
        ----------
        energy : float
            Incident energy to evaluate the distribution at
        ace : Ace, optional
            ACE object containing the Law=44 distribution data
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
            df = self.to_dataframe(energy, ace)
            
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
            ax.plot(df['cosine'], df['probability'], **kwargs)
            
            # Set labels and title
            ax.set_xlabel('Cosine (μ)')
            ax.set_ylabel('Probability Density')
            
            if title is None:
                mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
                title = f'Kalbach-Mann Angular Distribution for MT={mt_value} at {energy:.4g} MeV'
            ax.set_title(title)
            
            # Set axis limits
            ax.set_xlim(-1, 1)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            return fig, ax
        except ImportError:
            return None
    
    __repr__ = kalbach_mann_distribution_repr


@dataclass
class AngularDistributionContainer:
    """Container for all angular distributions."""
    elastic: Optional[AngularDistribution] = None  # Angular distribution for elastic scattering
    incident_neutron: Dict[int, AngularDistribution] = field(default_factory=dict)  # MT -> distribution for neutrons
    photon_production: Dict[int, AngularDistribution] = field(default_factory=dict)  # MT -> distribution for photons
    particle_production: List[Dict[int, AngularDistribution]] = field(default_factory=list)  # Particle index -> (MT -> distribution)
    
    def __post_init__(self):
        """Convert standard dictionaries to ErrorMessageDict and lists to ErrorMessageList."""
        # Convert incident_neutron dictionary to ErrorMessageDict
        if isinstance(self.incident_neutron, dict) and not isinstance(self.incident_neutron, ErrorMessageDict):
            self.incident_neutron = ErrorMessageDict(self.incident_neutron, dict_name="incident_neutron distributions")
        
        # Convert photon_production dictionary to ErrorMessageDict
        if isinstance(self.photon_production, dict) and not isinstance(self.photon_production, ErrorMessageDict):
            self.photon_production = ErrorMessageDict(self.photon_production, dict_name="photon_production distributions")
        
        # Convert particle_production list to ErrorMessageList
        if isinstance(self.particle_production, list) and not isinstance(self.particle_production, ErrorMessageList):
            # First convert the list itself
            particle_list = ErrorMessageList(self.particle_production, list_name="particle_production")
            
            # Then convert each dictionary in the list
            for i in range(len(particle_list)):
                if isinstance(particle_list[i], dict) and not isinstance(particle_list[i], ErrorMessageDict):
                    particle_list[i] = ErrorMessageDict(
                        particle_list[i], 
                        dict_name=f"particle_production[{i}] distributions"
                    )
            
            self.particle_production = particle_list
    
    @property
    def has_elastic_data(self) -> bool:
        """Check if elastic scattering angular distribution data is available."""
        return self.elastic is not None and not self.elastic.is_isotropic
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction angular distribution data is available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_photon_production_data(self) -> bool:
        """Check if photon production angular distribution data is available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production angular distribution data is available."""
        return len(self.particle_production) > 0 and any(len(p) > 0 for p in self.particle_production)
    
    def get_neutron_reaction_mt_numbers(self) -> List[int]:
        """Get the list of MT numbers for neutron reactions with angular distributions."""
        if isinstance(self.incident_neutron, ErrorMessageDict):
            return sorted(self.incident_neutron.keys_as_int())
        else:
            return sorted(list(self.incident_neutron.keys()))
    
    def get_photon_production_mt_numbers(self) -> List[int]:
        """Get the list of MT numbers for photon production with angular distributions."""
        if isinstance(self.photon_production, ErrorMessageDict):
            return sorted(self.photon_production.keys_as_int())
        else:
            return sorted(list(self.photon_production.keys()))
    
    def get_particle_production_mt_numbers(self, particle_idx: Optional[int] = None) -> Union[Dict[int, List[int]], List[int]]:
        """
        Get the list of MT numbers for particle production with angular distributions.
        
        Parameters
        ----------
        particle_idx : int, optional
            Index of the particle type. If None, returns a dictionary mapping
            particle indices to their MT numbers
            
        Returns
        -------
        Dict[int, List[int]] or List[int]
            If particle_idx is None: Dictionary mapping particle indices to lists of MT numbers
            If particle_idx is given: List of MT numbers for that particle index
        
        Raises
        ------
        IndexError
            If the specified particle index is out of bounds
        """
        # If no particle_idx specified, return dictionary for all particles
        if particle_idx is None:
            result = {}
            for idx in range(len(self.particle_production)):
                particle_data = self.particle_production[idx]
                if isinstance(particle_data, ErrorMessageDict):
                    result[idx] = sorted(particle_data.keys_as_int())
                else:
                    mt_keys = particle_data.keys()
                    if mt_keys and isinstance(next(iter(mt_keys)), XssEntry):
                        result[idx] = sorted([int(mt.value) for mt in mt_keys])
                    else:
                        result[idx] = sorted(list(mt_keys))
            return result
            
        # If particle_idx is specified, return list for that particle
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            available_indices = list(range(len(self.particle_production)))
            error_message = f"Particle index {particle_idx} is out of bounds."
            
            if len(self.particle_production) == 0:
                error_message += " No particle production data is available."
            else:
                error_message += f" Available particle indices: {available_indices}"
                
                # Add more information about particle counts for each index
                error_message += "\nParticle counts by index:"
                for idx, particle_data in enumerate(self.particle_production):
                    error_message += f"\n  Index {idx}: {len(particle_data)} reactions"
            
            raise IndexError(error_message)
        
        particle_data = self.particle_production[particle_idx]
        
        # Extract the MT values from XssEntry objects before sorting
        if isinstance(particle_data, ErrorMessageDict):
            return sorted(particle_data.keys_as_int())
        else:
            mt_keys = particle_data.keys()
            
            # Check if the keys are XssEntry objects or integers
            if mt_keys and isinstance(next(iter(mt_keys)), XssEntry):
                # If they are XssEntry objects, get their values first
                return sorted([int(mt.value) for mt in mt_keys])
            else:
                # If they are already integers, sort them directly
                return sorted(list(mt_keys))
    
    def sample_mu(self, mt: int, energy: float, random_value: float, 
                 particle_type: str = 'neutron', particle_idx: int = 0) -> float:
        """
        Sample a scattering cosine μ for a specific reaction and energy.
        
        Parameters
        ----------
        mt : int
            MT number for the reaction
        energy : float
            Incident energy
        random_value : float
            Random number between 0 and 1
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle'
        particle_idx : int, optional
            Index of the particle type (used only for particle_type='particle')
            
        Returns
        -------
        float
            Sampled cosine value μ (between -1 and 1)
        """
        # Special case for elastic scattering (MT=2)
        if particle_type == 'neutron' and mt == 2 and self.elastic:
            return self.elastic.sample_mu(energy, random_value)
        
        # Get the appropriate distribution container
        if particle_type == 'neutron':
            dist_container = self.incident_neutron
        elif particle_type == 'photon':
            dist_container = self.photon_production
        elif particle_type == 'particle':
            if particle_idx < 0 or particle_idx >= len(self.particle_production):
                # Fallback to isotropic if no data
                return 2.0 * random_value - 1.0
            dist_container = self.particle_production[particle_idx]
        else:
            raise ValueError(f"Unknown particle type: {particle_type}")
        
        # Get the angular distribution for this MT number
        if mt not in dist_container:
            # Fallback to isotropic if no data for this MT
            return 2.0 * random_value - 1.0
        
        # Sample from the distribution
        return dist_container[mt].sample_mu(energy, random_value)

    def to_dataframe(self, mt: int, energy: float, particle_type: str = 'neutron', 
                    particle_idx: int = 0, ace=None, num_points: int = 100) -> Optional[pd.DataFrame]:
        """
        Convert an angular distribution to a pandas DataFrame for a specific reaction and energy.
        
        Parameters
        ----------
        mt : int
            MT number for the reaction
        energy : float
            Incident energy to evaluate the distribution at
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle'
        particle_idx : int, optional
            Index of the particle type (used only for particle_type='particle')
        ace : Ace, optional
            ACE object containing the distribution data (needed for Kalbach-Mann)
        num_points : int, optional
            Number of angular points to generate, defaults to 100
            
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with 'cosine' and 'probability' columns, or None if pandas is not available
        """
        try:
            import pandas as pd
            
            # Special case for elastic scattering (MT=2)
            if particle_type == 'neutron' and mt == 2 and self.elastic:
                return self.elastic.to_dataframe(energy, num_points)
            
            # Get the appropriate distribution container
            if particle_type == 'neutron':
                dist_container = self.incident_neutron
            elif particle_type == 'photon':
                dist_container = self.photon_production
            elif particle_type == 'particle':
                if particle_idx < 0 or particle_idx >= len(self.particle_production):
                    # Fallback to isotropic if no data
                    return pd.DataFrame({
                        'cosine': np.linspace(-1, 1, num_points),
                        'probability': np.ones(num_points) * 0.5
                    })
                dist_container = self.particle_production[particle_idx]
            else:
                raise ValueError(f"Unknown particle type: {particle_type}")
            
            # Get the angular distribution for this MT number
            if mt not in dist_container:
                # Fallback to isotropic if no data for this MT
                return pd.DataFrame({
                    'cosine': np.linspace(-1, 1, num_points),
                    'probability': np.ones(num_points) * 0.5
                })
            
            # Special handling for Kalbach-Mann distributions
            distribution = dist_container[mt]
            if isinstance(distribution, KalbachMannAngularDistribution):
                return distribution.to_dataframe(energy, ace, num_points)
            else:
                return distribution.to_dataframe(energy, num_points)
                
        except ImportError:
            return None
    
    def plot(self, mt: int, energy: float, particle_type: str = 'neutron', 
            particle_idx: int = 0, ace=None, ax=None, title=None, **kwargs) -> Optional[Tuple]:
        """
        Plot an angular distribution for a specific reaction and energy.
        
        Parameters
        ----------
        mt : int
            MT number for the reaction
        energy : float
            Incident energy to evaluate the distribution at
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle'
        particle_idx : int, optional
            Index of the particle type (used only for particle_type='particle')
        ace : Ace, optional
            ACE object containing the distribution data (needed for Kalbach-Mann)
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
            df = self.to_dataframe(mt, energy, particle_type, particle_idx, ace)
            
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
            ax.plot(df['cosine'], df['probability'], **kwargs)
            
            # Set labels and title
            ax.set_xlabel('Cosine (μ)')
            ax.set_ylabel('Probability Density')
            
            if title is None:
                title = f'Angular Distribution for MT={mt} at {energy:.4g} MeV ({particle_type})'
            ax.set_title(title)
            
            # Set axis limits
            ax.set_xlim(-1, 1)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            return fig, ax
        except ImportError:
            return None
    
    def plot_energy_comparison(self, mt: int, energies: List[float], particle_type: str = 'neutron',
                              particle_idx: int = 0, ace=None, ax=None, title=None, 
                              colors=None, labels=None, **kwargs) -> Optional[Tuple]:
        """
        Plot angular distributions for multiple energies for comparison.
        
        Parameters
        ----------
        mt : int
            MT number for the reaction
        energies : List[float]
            List of incident energies to evaluate the distribution at
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle'
        particle_idx : int, optional
            Index of the particle type (used only for particle_type='particle')
        ace : Ace, optional
            ACE object containing the distribution data (needed for Kalbach-Mann)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        title : str, optional
            Title for the plot, if None a default title is used
        colors : List[str], optional
            List of colors for each energy, if None, default colors are used
        labels : List[str], optional
            List of labels for each energy, if None, energies are used
        **kwargs : dict
            Additional keyword arguments passed to the plot function
            
        Returns
        -------
        tuple or None
            Tuple of (fig, ax) or None if matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure and axes if not provided
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = ax.figure
            
            # Set default colors if not provided
            if colors is None:
                colors = plt.cm.viridis(np.linspace(0, 1, len(energies)))
            
            # Set default labels if not provided
            if labels is None:
                labels = [f"{e:.4g} MeV" for e in energies]
            
            # Plot for each energy
            for i, energy in enumerate(energies):
                # Get color and label for this energy
                color = colors[i] if i < len(colors) else 'blue'
                label = labels[i] if i < len(labels) else f"{energy:.4g} MeV"
                
                # Get data for this energy
                df = self.to_dataframe(mt, energy, particle_type, particle_idx, ace)
                
                # Plot data
                plot_kwargs = kwargs.copy()
                plot_kwargs['color'] = color
                plot_kwargs['label'] = label
                
                ax.plot(df['cosine'], df['probability'], **plot_kwargs)
            
            # Set labels and title
            ax.set_xlabel('Cosine (μ)')
            ax.set_ylabel('Probability Density')
            
            if title is None:
                title = f'Angular Distribution for MT={mt} ({particle_type}) at Multiple Energies'
            ax.set_title(title)
            
            # Set axis limits
            ax.set_xlim(-1, 1)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            return fig, ax
        except ImportError:
            return None
    
    __repr__ = angular_container_repr
