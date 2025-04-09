# Law 7: Maxwell fission spectrum

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution


@dataclass
class MaxwellFissionSpectrum(EnergyDistribution):
    """
    Law 7: Maxwell fission spectrum.
    
    From ENDF-6, MF=5, LF=7, this represents a Maxwellian fission spectrum:
    f(E→E') = (sqrt(E_out) / I) * exp(-E_out / θ(E))
    
    where I is the normalization constant and θ(E) is the temperature parameter.
    
    Data format (Table 38):
    - N_R: Interpolation scheme between temperatures
    - NBT, INT: Interpolation parameters for temperatures
    - N_E: Number of incident energies
    - E(l): Incident energy table
    - θ(l): Effective temperature tabulated on incident energies
    - U: Restriction energy (upper limit constraint)
    """
    law: int = 7
    n_temp_interp_regions: int = 0  # Number of interpolation regions for temperature
    temp_nbt: List[int] = field(default_factory=list)  # Temperature interpolation region boundaries
    temp_interp: List[int] = field(default_factory=list)  # Temperature interpolation schemes
    n_incident_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy values
    temperatures: List[float] = field(default_factory=list)  # θ parameter values for each incident energy
    restriction_energy: float = 0.0  # Restriction energy U (upper limit constraint)
    
    def get_temperature(self, incident_energy: float) -> float:
        """
        Get the θ parameter for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The θ parameter value (temperature)
        """
        if not self.incident_energies or len(self.incident_energies) == 0:
            return 0.0
            
        # If energy is outside the tabulated range, use the closest value
        if incident_energy <= self.incident_energies[0]:
            return self.temperatures[0]
        if incident_energy >= self.incident_energies[-1]:
            return self.temperatures[-1]
            
        # Use linear interpolation to get temperature
        # In a full implementation, we would use the interpolation scheme from temp_nbt and temp_interp
        idx = np.searchsorted(self.incident_energies, incident_energy) - 1
        e_low = self.incident_energies[idx]
        e_high = self.incident_energies[idx + 1]
        t_low = self.temperatures[idx]
        t_high = self.temperatures[idx + 1]
        
        # Linear interpolation
        t = t_low + (t_high - t_low) * (incident_energy - e_low) / (e_high - e_low)
        return t
    
    def calculate_normalization_constant(self, incident_energy: float, temperature: float) -> float:
        """
        Calculate the normalization constant I.
        
        I = (θ^3 * sqrt(π) / 2) * [ erf(√((E - U)/θ)) - √((E - U)/θ) * exp(−(E - U)/θ) ]
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        temperature : float
            The effective temperature
            
        Returns
        -------
        float
            The normalization constant I
        """
        from scipy import special
        
        # Calculate (E - U)/θ
        arg = (incident_energy - self.restriction_energy) / temperature
        if arg <= 0:
            return 1.0  # Default value if restriction exceeds incident energy
        
        # Calculate I
        sqrt_arg = np.sqrt(arg)
        erf_term = special.erf(sqrt_arg)
        exp_term = sqrt_arg * np.exp(-arg)
        
        # I = (θ^3 * sqrt(π) / 2) * [ erf(√((E - U)/θ)) - √((E - U)/θ) * exp(−(E - U)/θ) ]
        normalization = (temperature**3 * np.sqrt(np.pi) / 2.0) * (erf_term - exp_term)
        
        return max(normalization, 1.0e-30)  # Prevent division by zero
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the Maxwell fission spectrum.
        
        This uses the Maxwellian spectrum equation:
        f(E → E_out) = (sqrt(E_out) / I) * exp(-E_out / θ(E))
        
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
        if rng is None:
            rng = np.random.default_rng()
        
        # Get the temperature parameter for this incident energy
        temperature = self.get_temperature(incident_energy)
        
        # Check if we have valid temperature
        if temperature <= 0.0:
            return 0.0
        
        # Calculate the restriction on outgoing energy: 0 ≤ E_out ≤ (E − U)
        max_e_out = max(0.0, incident_energy - self.restriction_energy)
        
        # Use rejection sampling to sample from the Maxwell distribution
        while True:
            # Sample from exponential distribution with mean = temperature
            e_out = -temperature * np.log(rng.random())
            
            # Check if within the allowed range
            if e_out > max_e_out:
                continue
                
            # Acceptance probability proportional to sqrt(E_out)
            if rng.random() <= np.sqrt(e_out / temperature):
                return e_out
