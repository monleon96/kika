# Law 3: Level scattering

from dataclasses import dataclass
import numpy as np
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution

@dataclass
class LevelScattering(EnergyDistribution):
    """
    Law 3: Level scattering.
    
    This represents discrete two-body scattering.
    
    Data format (Table 34):
    - LDAT(1): (A + 1)/A|Q|
    - LDAT(2): (A / (A + 1))^2
    """
    law: int = 3
    aplusoaabsq: float = 0.0  # (A + 1)/A|Q| parameter
    asquare: float = 0.0      # (A / (A + 1))^2 parameter
    
    def get_cm_energy(self, incident_energy: float) -> float:
        """
        Calculate the outgoing center-of-mass energy.
        
        E_out^CM = LDAT(2) * (E - LDAT(1))
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            Outgoing center-of-mass energy
        """
        return self.asquare * (incident_energy - self.aplusoaabsq)
    
    def get_lab_energy(self, incident_energy: float, cm_cosine: float) -> float:
        """
        Calculate the outgoing energy in the laboratory system.
        
        E_out^LAB = E_out^CM + {E + 2μ_CM(A+1)(E*E_out^CM)^0.5} / (A+1)^2
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        cm_cosine : float
            Cosine of the center-of-mass scattering angle
            
        Returns
        -------
        float
            Outgoing energy in the laboratory system
        """
        # First get the CM energy
        e_cm = self.get_cm_energy(incident_energy)
        
        # If negative or zero, return 0
        if e_cm <= 0.0:
            return 0.0
        
        # Calculate the A value from LDAT(2)
        # LDAT(2) = (A/(A+1))^2, so A+1 = A/sqrt(LDAT(2))
        a_plus_1 = 1.0 / np.sqrt(self.asquare)
        
        # Calculate the second term
        term1 = incident_energy
        term2 = 2.0 * cm_cosine * np.sqrt(incident_energy * e_cm)
        second_term = (term1 + term2) / (a_plus_1 * a_plus_1)
        
        # Calculate the lab energy
        e_lab = e_cm + second_term
        
        return max(0.0, e_lab)