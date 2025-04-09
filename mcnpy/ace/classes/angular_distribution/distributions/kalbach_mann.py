from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.classes.angular_distribution.base import AngularDistribution
from mcnpy.ace.classes.angular_distribution.types import AngularDistributionType
from mcnpy.ace.classes.angular_distribution.angular_distribution_repr import kalbach_mann_distribution_repr
from mcnpy.ace.classes.angular_distribution.utils import Law44DataError


@dataclass
class KalbachMannAngularDistribution(AngularDistribution):
    """
    Angular distribution using Law=44 (Kalbach-Mann) from the DLW/DLWH Block.
    This distribution is correlated with energy and uses the Kalbach-Mann formalism.
    The actual angular distribution data is stored in the energy-angle distribution
    section of the ACE file (DLW/DLWH blocks).
    
    IMPORTANT: This distribution requires Law=44 data from the energy distribution
    section to calculate angular probabilities. The ACE object must be provided to
    methods that calculate or sample angular distributions.
    """
    # Reference to the reaction index in the DLW/DLWH block
    reaction_index: int = -1
    # Whether this is a particle production reaction
    is_particle_production: bool = False
    # Particle index (only used if is_particle_production=True)
    particle_idx: int = -1
    # Flag to indicate this distribution requires Law=44 data
    requires_law44_data: bool = True
    
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
            
        Raises
        ------
        Law44DataError
            If the ACE object is not provided or missing required data
        """
        if ace is None:
            mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
            raise Law44DataError(
                f"ACE object must be provided for Kalbach-Mann (Law=44) angular distribution (MT={mt_value})"
            )
            
        if ace.energy_distributions is None:
            mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
            raise Law44DataError(
                f"Energy distributions missing in ACE object for Kalbach-Mann distribution (MT={mt_value})"
            )
            
        # Get MT number
        mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
        
        # Get the appropriate container based on the particle type
        if self.is_particle_production:
            # Get particle production distributions
            if (self.particle_idx < 0 or 
                self.particle_idx >= len(ace.energy_distributions.particle_production)):
                raise Law44DataError(
                    f"Particle index {self.particle_idx} out of bounds for MT={mt_value}"
                )
                
            # Get distributions for this MT
            distributions = ace.energy_distributions.get_particle_distribution(
                self.particle_idx, mt_value)
        else:
            # Get incident neutron distributions
            distributions = ace.energy_distributions.get_neutron_distribution(mt_value)
            
        if not distributions:
            raise Law44DataError(
                f"No energy distributions found for MT={mt_value}"
                f"{f', particle={self.particle_idx}' if self.is_particle_production else ''}"
            )
            
        # Find the Law=44 distribution
        for dist in distributions:
            if dist.law == 44:  # Law=44 is Kalbach-Mann
                return dist
                
        raise Law44DataError(
            f"Law=44 distribution not found for MT={mt_value}"
            f"{f', particle={self.particle_idx}' if self.is_particle_production else ''}"
        )
    
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
            Sampled cosine value μ
            
        Raises
        ------
        Law44DataError
            If the ACE object is not provided or the Law=44 data is missing/invalid
        """
        mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
        
        # If no ACE data is provided, raise an error
        if ace is None:
            raise Law44DataError(
                f"ACE object must be provided to sample from Kalbach-Mann distribution (MT={mt_value})"
            )
            
        # If reaction index is invalid, raise an error
        if self.reaction_index < 0:
            raise Law44DataError(
                f"Invalid reaction index {self.reaction_index} for MT={mt_value}"
            )
            
        # Find the Law=44 distribution (this will raise Law44DataError if not found)
        km_dist = self._find_law44_distribution(ace)
            
        # Get the interpolated distribution for this energy
        dist = km_dist.get_interpolated_distribution(energy)
        if not dist:
            raise Law44DataError(
                f"No distribution data found for energy {energy} MeV in MT={mt_value}"
            )
            
        # Verify that the distribution contains required data
        if ('e_out' not in dist or 'r' not in dist or 'a' not in dist or 
            not dist['e_out'] or not dist['r'] or not dist['a']):
            raise Law44DataError(
                f"Incomplete Law=44 data for MT={mt_value} at energy {energy} MeV"
            )
            
        # Select a random energy point to get R and A parameters
        # In practice, this energy would come from the energy sampling step
        # which is correlated with the angular sampling
        try:
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
        except (IndexError, KeyError, TypeError) as e:
            raise Law44DataError(
                f"Error accessing Law=44 data for MT={mt_value}: {str(e)}"
            ) from e
    
    def to_dataframe(self, energy: Optional[float] = None, ace=None, num_points: int = 100, interpolate: bool = True) -> Optional[pd.DataFrame]:
        """
        Convert Kalbach-Mann angular distribution to a pandas DataFrame.
        
        Parameters
        ----------
        energy : float, optional
            Incident energy to evaluate the distribution at. If None and ace is provided, 
            returns data for a range of energies in the Law=44 distribution.
        ace : Ace, optional
            ACE object containing the Law=44 distribution data
        num_points : int, optional
            Number of angular points to generate, defaults to 100
        interpolate : bool, optional
            Whether to interpolate onto a regular grid - always True for Kalbach-Mann
            as there are no raw data points to return
            
        Returns
        -------
        pandas.DataFrame or None
            When energy is specified: DataFrame with 'cosine' and 'pdf' columns
            When energy is None: DataFrame with 'energy', 'cosine', and 'pdf' columns for a range of energies
            Returns None if pandas is not available
            
        Raises
        ------
        Law44DataError
            If the ACE object is not provided or the Law=44 data is missing/invalid
        """
        # For Kalbach-Mann, we always interpolate since there's no "raw data"
        # to return in the same sense as the other distribution types
        try:
            
            mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
            
            # If no ACE data is provided, raise an error
            if ace is None:
                raise Law44DataError(
                    f"ACE object must be provided for Kalbach-Mann (Law=44) angular distribution (MT={mt_value})"
                )
            
            # Find the Law=44 distribution (this will raise Law44DataError if not found)
            km_dist = self._find_law44_distribution(ace)
            
            # If no specific energy is requested, return data for a range of energies
            if energy is None:
                # Get incident energies from the Law=44 distribution
                km_energies = km_dist.get_incident_energies()
                if not km_energies:
                    raise Law44DataError(f"No incident energies found in Law=44 data for MT={mt_value}")
                
                # Sample a few energies (or use all if there are fewer than 5)
                sample_energies = km_energies
                if len(sample_energies) > 5:
                    # Sample 5 evenly spaced energies
                    indices = np.linspace(0, len(sample_energies)-1, 5, dtype=int)
                    sample_energies = [sample_energies[i] for i in indices]
                
                # Create rows for each energy/cosine combination
                rows = []
                cosines = np.linspace(-1, 1, num_points)
                
                for e in sample_energies:
                    try:
                        # Get the interpolated distribution for this energy
                        dist = km_dist.get_interpolated_distribution(e)
                        if not dist or 'r' not in dist or 'a' not in dist:
                            continue
                        
                        # Use middle outgoing energy point for R and A parameters
                        if 'e_out' in dist and len(dist['e_out']) > 0:
                            middle_idx = len(dist['e_out']) // 2
                            r_value = dist['r'][middle_idx]
                            a_value = dist['a'][middle_idx]
                            
                            # Handle XssEntry objects if present
                            r_value = float(r_value.value if hasattr(r_value, 'value') else r_value)
                            a_value = float(a_value.value if hasattr(a_value, 'value') else a_value)
                            
                            # Calculate PDF values
                            for cosine in cosines:
                                # If a is very small, use isotropic
                                if abs(a_value) < 1.0e-3:
                                    pdf = 0.5
                                else:
                                    # Calculate Kalbach-Mann PDF
                                    sinh_a = np.sinh(a_value)
                                    normalization = (a_value / 2.0) / sinh_a
                                    pdf = normalization * (
                                        np.cosh(a_value * cosine) + r_value * np.sinh(a_value * cosine)
                                    )
                                
                                rows.append({
                                    'energy': e,
                                    'cosine': cosine,
                                    'pdf': pdf,
                                    'r': r_value,
                                    'a': a_value
                                })
                    except Exception as e:
                        # Skip this energy if there's an error
                        continue
                
                # Return the dataframe if we have data
                if rows:
                    return pd.DataFrame(rows)
                else:
                    raise Law44DataError(f"Could not generate data for any energies in Law=44 distribution for MT={mt_value}")
            
            # For specific energy, use the existing implementation
            # Get the interpolated distribution for this energy
            dist = km_dist.get_interpolated_distribution(energy)
            if not dist:
                raise Law44DataError(
                    f"No distribution data found for energy {energy} MeV in MT={mt_value}"
                )
            
            # Verify that we have e_out, r, and a data and they're non-empty
            if ('e_out' not in dist or 'r' not in dist or 'a' not in dist or 
                not dist['e_out'] or not dist['r'] or not dist['a']):
                raise Law44DataError(
                    f"Incomplete Law=44 data for MT={mt_value} at energy {energy} MeV"
                )
            
            # Convert all lengths to integers explicitly to avoid type issues
            e_out_len = int(len(dist['e_out'])) if isinstance(dist['e_out'], list) else 0
            r_len = int(len(dist['r'])) if isinstance(dist['r'], list) else 0
            a_len = int(len(dist['a'])) if isinstance(dist['a'], list) else 0
            
            # Make sure r and a arrays are at least as long as e_out
            if r_len < e_out_len or a_len < e_out_len:
                raise Law44DataError(
                    f"Inconsistent Law=44 data lengths for MT={mt_value}: "
                    f"e_out={e_out_len}, r={r_len}, a={a_len}"
                )
            
            # For simplicity, use the R and A parameters from the middle of the E_out range
            middle_idx = min(e_out_len // 2, r_len - 1, a_len - 1)
            
            # Extract values with explicit conversion to float
            r_value = dist['r'][middle_idx]
            a_value = dist['a'][middle_idx]
            
            # Handle XssEntry objects if present
            r_value = float(r_value.value if hasattr(r_value, 'value') else r_value)
            a_value = float(a_value.value if hasattr(a_value, 'value') else a_value)
            
            # Generate a fine cosine grid
            cosines = np.linspace(-1, 1, num_points)
            
            # Calculate PDF values for the Kalbach-Mann distribution
            pdf_values = np.zeros_like(cosines)
            
            # If a is very small, use isotropic distribution
            if abs(a_value) < 1.0e-3:
                pdf_values.fill(0.5)
            else:
                # Calculate Kalbach-Mann PDF: p(μ) = (a/2)/sinh(a) * [cosh(aμ) + r*sinh(aμ)]
                sinh_a = np.sinh(a_value)
                normalization = (a_value / 2.0) / sinh_a
                
                for i, mu in enumerate(cosines):
                    pdf_values[i] = normalization * (
                        np.cosh(a_value * mu) + r_value * np.sinh(a_value * mu)
                    )
            
            # Create arrays for the scalar values to ensure consistent length
            r_values = np.full_like(cosines, r_value, dtype=float)
            a_values = np.full_like(cosines, a_value, dtype=float)
            energy_values = np.full_like(cosines, energy, dtype=float)
            
            return pd.DataFrame({
                'energy': energy_values,
                'cosine': cosines,
                'pdf': pdf_values,
                'r': r_values,
                'a': a_values
            })

        except Law44DataError:
            # Re-raise Law44DataError exceptions
            raise
        except Exception as e:
            # Convert other exceptions to Law44DataError with a clear message
            raise Law44DataError(
                f"Error calculating Kalbach-Mann distribution for MT={mt_value}: {str(e)}"
            ) from e
    
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
            
        Raises
        ------
        Law44DataError
            If the ACE object is not provided or the Law=44 data is missing/invalid
        """
            
        # Get the data to plot - this will raise Law44DataError if ACE is missing
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
        ax.plot(df['cosine'], df['pdf'], **kwargs)
        
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
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        mt_value = int(self.mt.value) if hasattr(self.mt, 'value') else int(self.mt)
        particle_info = f", particle={self.particle_idx}" if self.is_particle_production else ""
        return (f"Kalbach-Mann Angular Distribution (MT={mt_value}{particle_info})\n"
                f"REQUIRES: Law=44 data from energy distribution section\n"
                f"NOTE: Must provide ACE object when sampling or plotting this distribution")
    
    __repr__ = kalbach_mann_distribution_repr