# EnergyDependentYield class

from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class EnergyDependentYield:
    """
    Energy-dependent neutron yield data for reactions with |TY| > 100.
    
    Data format (Table 52):
    - N_R: Number of interpolation regions
    - NBT(l), l = 1,...,N_R: ENDF interpolation parameters
    - INT(l), l = 1,...,N_R: ENDF interpolation scheme
    - N_E: Number of energies
    - E(l), l = 1,...,N_E: Tabular energy points
    - Y(l), l = 1,...,N_E: Corresponding energy-dependent yields
    """
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # ENDF interpolation parameters
    interp: List[int] = field(default_factory=list)  # ENDF interpolation scheme
    n_energies: int = 0  # Number of energy points
    energies: List[float] = field(default_factory=list)  # Tabular energy points
    yields: List[float] = field(default_factory=list)  # Corresponding yields
    
    def get_yield(self, energy: float) -> float:
        """
        Get the yield value for a given incident energy.
        
        Parameters
        ----------
        energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The interpolated yield value
        """
        if not self.energies or not self.yields:
            return 0.0
            
        # If energy is outside the tabulated range, use the closest value
        if energy <= self.energies[0]:
            return self.yields[0]
        if energy >= self.energies[-1]:
            return self.yields[-1]
            
        # Use linear interpolation to get yield value
        # In a full implementation, we would use the interpolation scheme from nbt and interp
        return np.interp(energy, self.energies, self.yields)
