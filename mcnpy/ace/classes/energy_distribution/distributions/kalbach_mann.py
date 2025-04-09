# Law 44: Kalbach-Mann distribution

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution

@dataclass
class KalbachMannDistribution(EnergyDistribution):
    """
    Law 44: Kalbach-Mann Correlated Energy-Angle Distribution.
    
    From ENDF-6 MF=6 LAW=1, LANG=2, this represents a correlated energy-angle
    distribution using the Kalbach-87 formalism.
    
    Data format (Table 43 and 44):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E(l): Incident energy grid
    - L(l): Location of distributions
    
    For each incident energy:
    - INTT': Combined interpolation parameter (10*N_D + INTT)
    - N_p: Number of points in distribution
    - E_out(l): Outgoing energy grid
    - PDF(l): Probability density function
    - CDF(l): Cumulative density function
    - R(l): Precompound fraction r
    - A(l): Angular distribution slope value a
    
    Angular distribution is given by:
    p(μ, E_in, E_out) = (1/2)*(a/sinh(a))*[cosh(aμ) + r*sinh(aμ)]
    """
    law: int = 44
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each energy-angle distribution
    # Each entry is a dictionary containing the distribution data
    distributions: List[Dict] = field(default_factory=list)
    
    def get_distribution(self, energy_idx: int) -> Dict:
        """
        Get the distribution for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the distribution data with keys:
            - 'intt': Interpolation type
            - 'n_discrete': Number of discrete lines
            - 'n_points': Number of points
            - 'e_out': Outgoing energy grid
            - 'pdf': Probability density function
            - 'cdf': Cumulative density function
            - 'r': Precompound fraction
            - 'a': Angular distribution slope
            or None if index is invalid
        """
        if 0 <= energy_idx < len(self.distributions):
            return self.distributions[energy_idx]
        return None
    
    def get_interpolated_distribution(self, incident_energy: float) -> Dict:
        """
        Get an interpolated distribution for a specific incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the interpolated distribution data
        """
        # Convert incident energies from XssEntry to float values for comparison
        incident_energy_values = [e.value if hasattr(e, 'value') else float(e) for e in self.incident_energies]
        
        # Find the bracketing incident energies
        if not incident_energy_values or incident_energy <= incident_energy_values[0]:
            # Below the minimum incident energy, return the first distribution
            return self.get_distribution(0) if self.distributions else None
        
        if incident_energy >= incident_energy_values[-1]:
            # Above the maximum incident energy, return the last distribution
            return self.get_distribution(len(self.distributions) - 1) if self.distributions else None
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(incident_energy_values, incident_energy, side='right') - 1
        
        # Get the distributions for the bracketing energies
        dist_low = self.get_distribution(idx)
        dist_high = self.get_distribution(idx + 1)
        
        if not dist_low or not dist_high:
            return dist_low if dist_low else dist_high
            
        # Interpolate the distributions based on the incident energy
        energy_low = self.incident_energies[idx]
        energy_high = self.incident_energies[idx + 1]
        
        # Determine the interpolation scheme from the regions
        # For simplicity, assume linear-linear interpolation
        # In a full implementation, we would determine this from the NBT and INT arrays
        
        # Interpolation fraction
        frac = (incident_energy - energy_low) / (energy_high - energy_low)
        
        # Create common grid for interpolation (use outgoing energy points from low distribution)
        e_out = dist_low['e_out']
        
        # Interpolate PDF values
        pdf_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['pdf'])
        pdf_interp = (1.0 - frac) * np.array(dist_low['pdf']) + frac * pdf_high_interp
        
        # Interpolate CDF values
        cdf_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['cdf'])
        cdf_interp = (1.0 - frac) * np.array(dist_low['cdf']) + frac * cdf_high_interp
        
        # Interpolate r values (precompound fraction)
        r_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['r'])
        r_interp = (1.0 - frac) * np.array(dist_low['r']) + frac * r_high_interp
        
        # Interpolate a values (angular distribution slope)
        a_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['a'])
        a_interp = (1.0 - frac) * np.array(dist_low['a']) + frac * a_high_interp
        
        # Create interpolated distribution
        interp_dist = {
            'intt': dist_low['intt'],  # Use the INTT from the lower energy
            'n_discrete': dist_low['n_discrete'],  # Use discrete count from lower energy
            'n_points': len(e_out),
            'e_out': e_out,
            'pdf': pdf_interp.tolist(),
            'cdf': cdf_interp.tolist(),
            'r': r_interp.tolist(),
            'a': a_interp.tolist()
        }
        
        return interp_dist
    
    def sample_outgoing_energy_angle(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """
        Sample an outgoing energy and angle from the distribution for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        Tuple[float, float]
            Tuple of (outgoing_energy, cosine)
        """
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
            
        # Get interpolated distribution
        dist = self.get_interpolated_distribution(incident_energy)
        if not dist:
            return 0.0, 0.0
            
        # 1. Sample outgoing energy using the CDF
        xi = rng.random()
        e_out = np.interp(xi, dist['cdf'], dist['e_out'])
        
        # 2. Find the index in the e_out array that corresponds to this energy
        idx = np.searchsorted(dist['e_out'], e_out, side='right') - 1
        idx = max(0, min(idx, len(dist['e_out']) - 1))
        
        # 3. Get the corresponding a and r values
        a_value = dist['a'][idx]
        r_value = dist['r'][idx]
        
        # 4. Sample the cosine using the Kalbach-Mann formula
        # p(μ) = (1/2)*(a/sinh(a))*[cosh(aμ) + r*sinh(aμ)]
        
        # Special case for small a (nearly isotropic)
        if abs(a_value) < 1.0e-3:
            return e_out, 2.0 * rng.random() - 1.0
        
        # Sample from the Kalbach-Mann distribution
        cosine = self.sample_kalbach_mann(a_value, r_value, rng)
        
        return e_out, cosine
    
    def sample_kalbach_mann(self, a: float, r: float, rng: np.random.Generator) -> float:
        """
        Sample a cosine from the Kalbach-Mann angular distribution.
        
        p(μ) = (1/2)*(a/sinh(a))*[cosh(aμ) + r*sinh(aμ)]
        
        Parameters
        ----------
        a : float
            Angular distribution slope parameter
        r : float
            Precompound fraction parameter
        rng : np.random.Generator
            Random number generator
            
        Returns
        -------
        float
            Sampled cosine value in [-1, 1]
        """
        # Use rejection sampling for simplicity
        # A more efficient algorithm could be used for production code
        
        # Maximum value of the distribution occurs at μ = 1 for r > 0
        # and at μ = -1 for r < 0
        if r >= 0:
            p_max = (a / (2 * np.sinh(a))) * (np.cosh(a) + r * np.sinh(a))
            mu_max = 1.0
        else:
            p_max = (a / (2 * np.sinh(a))) * (np.cosh(-a) - r * np.sinh(-a))
            mu_max = -1.0
        
        while True:
            # Sample μ uniformly in [-1, 1]
            mu = 2.0 * rng.random() - 1.0
            
            # Calculate probability
            p_mu = (a / (2 * np.sinh(a))) * (np.cosh(a * mu) + r * np.sinh(a * mu))
            
            # Accept with probability p_mu / p_max
            if rng.random() <= p_mu / p_max:
                return mu
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample just an outgoing energy from the distribution for a given incident energy.
        
        This is a convenience method that only returns the energy part of the energy-angle pair.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        float
            Sampled outgoing energy
        """
        e_out, _ = self.sample_outgoing_energy_angle(incident_energy, rng)
        return e_out