# Law 66: N-body phase space

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution

@dataclass
class NBodyPhaseSpaceDistribution(EnergyDistribution):
    """
    Law 66: N-Body Phase Space Distribution.
    
    From ENDF-6 MF=6 LAW=6, this represents a phase space energy distribution
    for N bodies in the final state.
    
    Data format (Table 48):
    - LDAT(1): NPSX - Number of bodies in the phase space
    - LDAT(2): A_P - Total mass ratio for the NPSX particles
    - LDAT(3): INTT - Interpolation parameter
    - LDAT(4): N_P - Number of points in the distribution
    - LDAT(5) to LDAT(4+N_P): ξ_out(j) - ξ grid (between 0 and 1)
    - LDAT(5+N_P) to LDAT(4+2*N_P): PDF(j) - Probability density function
    - LDAT(5+2*N_P) to LDAT(4+3*N_P): CDF(j) - Cumulative density function
    """
    law: int = 66
    npsx: int = 0  # Number of bodies in the phase space
    ap: float = 0.0  # Total mass ratio for the NPSX particles
    intt: int = 0  # Interpolation parameter (1=histogram, 2=linear-linear)
    n_points: int = 0  # Number of points in the distribution
    xi_grid: List[float] = field(default_factory=list)  # ξ grid (between 0 and 1)
    pdf: List[float] = field(default_factory=list)  # Probability density function
    cdf: List[float] = field(default_factory=list)  # Cumulative density function
    
    def get_max_energy(self, incident_energy: float, awr: float, q_value: float = 0.0) -> float:
        """
        Calculate the maximum energy available to the outgoing particles.
        
        E_i^max = (A_P - 1)/A_P * (A / (A + 1) * E_in + Q)
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        awr : float
            Atomic weight ratio
        q_value : float, optional
            Q-value for the reaction
            
        Returns
        -------
        float
            Maximum available energy
        """
        if self.ap <= 1.0:
            return 0.0
            
        term1 = (self.ap - 1.0) / self.ap
        term2 = (awr / (awr + 1.0)) * incident_energy + q_value
        return term1 * term2
    
    def sample_t_fraction(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample the T(ξ) fraction from the tabulated distribution.
        
        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        float
            Sampled T(ξ) value between 0 and 1
        """
        if not self.xi_grid or not self.cdf or len(self.xi_grid) != len(self.cdf):
            # If no distribution data is available, use a simple
            # approximation based on number of bodies (NPSX)
            if rng is None:
                rng = np.random.default_rng()
                
            # For large NPSX values, the distribution peaks near 0.5
            # For NPSX=2, it's fairly flat
            if self.npsx <= 2:
                return rng.random()
            else:
                # Simple approximation - use a beta distribution
                # that gets more peaked as NPSX increases
                alpha = 2.0
                beta = 2.0
                return rng.beta(alpha, beta)
                
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
            
        # Sample using the CDF
        xi = rng.random()
        
        # Determine the T value using the tabulated CDF
        t_xi = np.interp(xi, self.cdf, self.xi_grid)
        
        return t_xi
    
    def sample_outgoing_energy(self, incident_energy: float, awr: float, 
                              q_value: float = 0.0, 
                              rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the N-body phase space distribution.
        
        E_out = T(ξ) * E_i^max
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        awr : float
            Atomic weight ratio
        q_value : float, optional
            Q-value for the reaction
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        float
            Sampled outgoing energy
        """
        # Calculate the maximum available energy
        e_max = self.get_max_energy(incident_energy, awr, q_value)
        
        # Sample the T(ξ) fraction
        t_xi = self.sample_t_fraction(rng)
        
        # Calculate outgoing energy: E_out = T(ξ) * E_i^max
        e_out = t_xi * e_max
        
        return e_out