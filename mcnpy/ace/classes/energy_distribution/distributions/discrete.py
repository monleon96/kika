# Law 2: Discrete distributions

from dataclasses import dataclass, field
from typing import List
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution

@dataclass
class DiscreteEnergyDistribution(EnergyDistribution):
    """
    Law 2: Discrete energy distribution.
    
    This represents nuclear level excitations where the outgoing energy is discrete.
    For photons, this represents discrete photon energy lines.
    """
    law: int = 2
    n_discrete_energies: int = 0  # Number of discrete energies
    discrete_energies: List[float] = field(default_factory=list)  # Discrete energy values
    probabilities: List[float] = field(default_factory=list)  # Probabilities for each discrete energy
    
    # For photons (according to Table 33)
    lp: int = 0  # Indicator of whether photon is primary (0,1) or non-primary (2)
    eg: float = 0.0  # Photon energy or binding energy
    
    def get_photon_energy(self, incident_energy: float, awr: float) -> float:
        """
        Calculate the photon energy based on LP indicator.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        awr : float
            Atomic weight ratio
            
        Returns
        -------
        float
            The photon energy
        """
        if self.lp == 0 or self.lp == 1:
            # For LP=0 or LP=1, the photon energy is simply EG
            return self.eg
        elif self.lp == 2:
            # For LP=2, the photon energy is EG + (AWR/(AWR+1))*E_N
            return self.eg + (awr / (awr + 1.0)) * incident_energy
        else:
            # Invalid LP, return EG
            return self.eg
