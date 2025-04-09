# Law 11: Watt spectrum

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution

@dataclass
class EnergyDependentWattSpectrum(EnergyDistribution):
    """
    Law 11: Energy Dependent Watt Spectrum.
    
    From ENDF-6, MF=5, LF=11, this represents a Watt spectrum:
    f(E→E') = (1/I) * exp(-E_out / a) * sinh(sqrt(b * E_out))
    
    where I is the normalization constant and a(E), b(E) are energy-dependent parameters.
    
    Data format (Table 40):
    - N_Ra, NBT_a, INT_a: Interpolation scheme for parameter a
    - N_Ea, E_a, a: Tabulated values of parameter a(E)
    - N_Rb, NBT_b, INT_b: Interpolation scheme for parameter b
    - N_Eb, E_b, b: Tabulated values of parameter b(E)
    - U: Restriction energy (upper limit constraint)
    """
    law: int = 11
    
    # Parameter a interpolation data
    n_a_interp_regions: int = 0  # Number of interpolation regions for parameter a
    a_nbt: List[int] = field(default_factory=list)  # Parameter a interpolation region boundaries
    a_interp: List[int] = field(default_factory=list)  # Parameter a interpolation schemes
    n_a_energies: int = 0  # Number of incident energies for parameter a
    a_incident_energies: List[float] = field(default_factory=list)  # Incident energy table for a
    a_values: List[float] = field(default_factory=list)  # Parameter a values
    
    # Parameter b interpolation data
    n_b_interp_regions: int = 0  # Number of interpolation regions for parameter b
    b_nbt: List[int] = field(default_factory=list)  # Parameter b interpolation region boundaries
    b_interp: List[int] = field(default_factory=list)  # Parameter b interpolation schemes
    n_b_energies: int = 0  # Number of incident energies for parameter b
    b_incident_energies: List[float] = field(default_factory=list)  # Incident energy table for b
    b_values: List[float] = field(default_factory=list)  # Parameter b values
    
    restriction_energy: float = 0.0  # Restriction energy U (upper limit constraint)
    
    def get_a_parameter(self, incident_energy: float) -> float:
        """
        Get the a parameter for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The a parameter value
        """
        if not self.a_incident_energies or len(self.a_incident_energies) == 0:
            return 0.0
            
        # If energy is outside the tabulated range, use the closest value
        if incident_energy <= self.a_incident_energies[0]:
            return self.a_values[0]
        if incident_energy >= self.a_incident_energies[-1]:
            return self.a_values[-1]
            
        # Use linear interpolation to get parameter a
        # In a full implementation, we would use the interpolation scheme from a_nbt and a_interp
        idx = np.searchsorted(self.a_incident_energies, incident_energy) - 1
        e_low = self.a_incident_energies[idx]
        e_high = self.a_incident_energies[idx + 1]
        a_low = self.a_values[idx]
        a_high = self.a_values[idx + 1]
        
        # Linear interpolation
        a = a_low + (a_high - a_low) * (incident_energy - e_low) / (e_high - e_low)
        return a
    
    def get_b_parameter(self, incident_energy: float) -> float:
        """
        Get the b parameter for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The b parameter value
        """
        if not self.b_incident_energies or len(self.b_incident_energies) == 0:
            return 0.0
            
        # If energy is outside the tabulated range, use the closest value
        if incident_energy <= self.b_incident_energies[0]:
            return self.b_values[0]
        if incident_energy >= self.b_incident_energies[-1]:
            return self.b_values[-1]
            
        # Use linear interpolation to get parameter b
        # In a full implementation, we would use the interpolation scheme from b_nbt and b_interp
        idx = np.searchsorted(self.b_incident_energies, incident_energy) - 1
        e_low = self.b_incident_energies[idx]
        e_high = self.b_incident_energies[idx + 1]
        b_low = self.b_values[idx]
        b_high = self.b_values[idx + 1]
        
        # Linear interpolation
        b = b_low + (b_high - b_low) * (incident_energy - e_low) / (e_high - e_low)
        return b
    
    def calculate_normalization_constant(self, incident_energy: float, a: float, b: float) -> float:
        """
        Calculate the normalization constant I for the Watt spectrum.
        
        I = (1/2) * sqrt(π * a^3 * b / 4) * exp(b * a / 4) *
            [erf((E − U)/a + sqrt(ab)/2) + erf(sqrt((E − U)/a − sqrt(ab)/2))]
            − a * exp(−(E − U)/a) * sinh(sqrt(b * (E − U)))
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        a : float
            The a parameter value
        b : float
            The b parameter value
            
        Returns
        -------
        float
            The normalization constant I
        """
        from scipy import special
        
        # Check for valid parameters
        if a <= 0.0 or b <= 0.0:
            return 1.0
        
        # Calculate (E − U)
        e_minus_u = incident_energy - self.restriction_energy
        if e_minus_u <= 0:
            return 1.0  # Default value if restriction exceeds incident energy
            
        # Calculate common terms
        sqrt_ab = np.sqrt(a * b)
        sqrt_e_minus_u = np.sqrt(e_minus_u)
        e_minus_u_over_a = e_minus_u / a
        
        # First term: (1/2) * sqrt(π * a^3 * b / 4) * exp(b*a/4)
        term1 = 0.5 * np.sqrt(np.pi * a**3 * b / 4.0) * np.exp(b * a / 4.0)
        
        # Error function terms
        arg1 = e_minus_u_over_a + sqrt_ab / 2.0
        arg2 = np.sqrt(e_minus_u_over_a) - sqrt_ab / 2.0
        
        if arg2 < 0:
            # Handle negative argument in second erf term
            erf_term = special.erf(arg1) - special.erf(-arg2)
        else:
            erf_term = special.erf(arg1) + special.erf(arg2)
            
        # Sinh term
        sinh_term = a * np.exp(-e_minus_u_over_a) * np.sinh(sqrt_ab * sqrt_e_minus_u / a)
        
        # Calculate I using equation 11
        normalization = term1 * erf_term - sinh_term
        
        return max(normalization, 1.0e-30)  # Prevent division by zero
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the Watt spectrum.
        
        This uses the Watt spectrum equation:
        f(E → E_out) = (1/I) * exp(-E_out / a) * sinh(sqrt(b * E_out))
        
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
        
        # Get parameters a and b for this incident energy
        a = self.get_a_parameter(incident_energy)
        b = self.get_b_parameter(incident_energy)
        
        # Check if we have valid parameters
        if a <= 0.0 or b <= 0.0:
            return 0.0
        
        # Calculate the restriction on outgoing energy: 0 ≤ E_out ≤ (E − U)
        max_e_out = max(0.0, incident_energy - self.restriction_energy)
        
        # Use simplified sampling algorithm for the Watt spectrum
        # Note: A more sophisticated approach would use the exact distribution
        while True:
            # Sample from exponential distribution with mean = a
            e1 = -a * np.log(rng.random())
            
            # Sample from exponential distribution with mean = 1/b
            e2 = -np.log(rng.random()) / np.sqrt(b)
            
            # Combine samples to approximate Watt spectrum
            e_out = e1 + e2**2
            
            # Check if within the allowed range
            if e_out <= max_e_out:
                return e_out
