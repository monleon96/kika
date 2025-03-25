from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from mcnpy.ace.xss import XssEntry

@dataclass
class EnergyDistribution:
    """Base class for energy distributions of secondary particles."""
    law: int = 0  # Energy distribution law number
    idat: int = 0  # Offset to distribution data in XSS array (relative to JED)
    
    # Law applicability parameters
    applicability_energies: List[XssEntry] = field(default_factory=list)  # Energies at which to check if law applies
    applicability_probabilities: List[XssEntry] = field(default_factory=list)  # Probability of law validity at each energy
    nbt: List[int] = field(default_factory=list)  # NBT interpolation parameters
    interp: List[int] = field(default_factory=list)  # INT interpolation scheme
    
    @property
    def law_name(self) -> str:
        """Get a descriptive name for the energy distribution law."""
        law_names = {
            1: "Tabular Energy Distribution",
            2: "Discrete Energy Distribution",
            3: "Level Scattering",
            4: "Continuous Energy-Angle Distribution",
            5: "General Evaporation Spectrum",
            7: "Maxwell Fission Spectrum",
            9: "Evaporation Spectrum",
            11: "Energy-dependent Watt Spectrum",
            22: "Tabular Linear Functions of Incident Energy Out",
            24: "Tabular Energy Multipliers",
            44: "Kalbach-Mann Correlated Energy-Angle Distribution",
            61: "Correlated Energy-Angle Distribution",
            66: "N-body Phase Space Distribution"
        }
        return law_names.get(self.law, f"Unknown Law {self.law}")
    
    def get_applicability_probability(self, energy: float) -> float:
        """
        Get the probability that this law applies at the given incident energy.
        
        If the law is the only law for a reaction, this returns 1.0.
        Otherwise, it interpolates between probability values.
        
        Parameters
        ----------
        energy : float
            The incident particle energy
            
        Returns
        -------
        float
            Probability between 0 and 1 that this law applies
        """
        # If there's no applicability data, this is the only law
        if not self.applicability_energies or not self.applicability_probabilities:
            return 1.0
        
        # Convert applicability_energies and applicability_probabilities to lists of float values
        energy_values = [entry.value for entry in self.applicability_energies]
        prob_values = [entry.value for entry in self.applicability_probabilities]
        
        # If energy is outside the tabulated range, use the closest value
        if energy <= energy_values[0]:
            return prob_values[0]
        if energy >= energy_values[-1]:
            return prob_values[-1]
        
        # Otherwise, interpolate
        # Note: For a complete implementation, we would use the NBT and INT arrays
        # to determine the interpolation scheme for each region
        # For simplicity, we use linear interpolation here
        return np.interp(energy, energy_values, prob_values)


@dataclass
class TabularEnergyDistribution(EnergyDistribution):
    """
    Law 1: Tabular energy distribution.
    
    This is a tabular function of outgoing energy E' and incident energy E.
    """
    law: int = 1
    interpolation: int = 0  # Interpolation scheme (1=histogram, 2=lin-lin)
    n_incident_energies: int = 0  # Number of incident energies
    incident_energies: List[XssEntry] = field(default_factory=list)  # Incident energy values
    
    # For each incident energy, there's a tabular distribution of outgoing energies
    # Each distribution has a number of points, an interpolation scheme, and energy-pdf pairs
    distribution_data: List[Dict] = field(default_factory=list)
    
    def get_outgoing_energy_distribution(self, incident_energy: float) -> Tuple[List[float], List[float]]:
        """
        Get the outgoing energy distribution for a given incident energy.
        
        For Law 1, this involves interpolating between the tabular distributions
        at the closest incident energies.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        Tuple[List[float], List[float]]
            Tuple of (energies, probabilities)
        """
        # Extract incident energy values
        incident_energy_values = [e.value for e in self.incident_energies]
        
        # If energy is outside the tabulated range or we don't have enough data, return empty lists
        if not incident_energy_values or not self.distribution_data:
            return [], []
            
        if incident_energy <= incident_energy_values[0]:
            dist = self.distribution_data[0]
            return [e.value for e in dist['e_out']], [p.value for p in dist['pdf']]
            
        if incident_energy >= incident_energy_values[-1]:
            dist = self.distribution_data[-1]
            return [e.value for e in dist['e_out']], [p.value for p in dist['pdf']]
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(incident_energy_values, incident_energy) - 1
        
        # Get the distributions for the bracketing energies
        dist_low = self.distribution_data[idx]
        dist_high = self.distribution_data[idx + 1]
        
        # Calculate interpolation factor
        energy_low = incident_energy_values[idx]
        energy_high = incident_energy_values[idx + 1]
        factor = (incident_energy - energy_low) / (energy_high - energy_low)
        
        # Extract values from XssEntry objects
        e_out_low = [e.value for e in dist_low['e_out']]
        pdf_low = [p.value for p in dist_low['pdf']]
        e_out_high = [e.value for e in dist_high['e_out']]
        pdf_high = [p.value for p in dist_high['pdf']]
        
        # For simplicity, we'll use the energy grid from the lower distribution
        # and interpolate the PDF values 
        pdf_high_interp = np.interp(e_out_low, e_out_high, pdf_high)
        pdf_interp = [(1.0 - factor) * p_low + factor * p_high 
                      for p_low, p_high in zip(pdf_low, pdf_high_interp)]
        
        return e_out_low, pdf_interp


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


@dataclass
class ContinuousTabularDistribution(EnergyDistribution):
    """
    Law 4: Continuous tabular energy distribution.
    
    From ENDF-6 Law 1, this represents a fully tabulated energy distribution.
    The distribution may be discrete, continuous, or a combination.
    
    Data format (Table 35 and 36):
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
    """
    law: int = 4
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each energy distribution
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
            Dictionary containing the distribution data
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
        # Find the bracketing incident energies
        if not self.incident_energies or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first distribution
            return self.get_distribution(0) if self.distributions else None
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last distribution
            return self.get_distribution(len(self.incident_energies) - 1) if self.distributions else None
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Get the distributions for the bracketing energies
        dist_low = self.get_distribution(idx)
        dist_high = self.get_distribution(idx + 1)
        
        if not dist_low or not dist_high:
            return None
            
        # Interpolate the distributions based on the incident energy
        energy_low = self.incident_energies[idx]
        energy_high = self.incident_energies[idx + 1]
        
        # Determine the interpolation scheme from the regions
        # For simplicity, assume linear-linear interpolation
        # In a full implementation, we would determine this from the NBT and INT arrays
        
        # Interpolation fraction
        frac = (incident_energy - energy_low) / (energy_high - energy_low)
        
        # Interpolate the distribution arrays
        e_out_interp = np.interp(
            dist_low['e_out'],  # x-coordinates from first distribution
            dist_high['e_out'],  # x-coordinates from second distribution
            frac  # interpolation fraction
        )
        
        pdf_interp = np.interp(
            dist_low['pdf'],  # y-coordinates from first distribution
            dist_high['pdf'],  # y-coordinates from second distribution
            frac  # interpolation fraction
        )
        
        cdf_interp = np.interp(
            dist_low['cdf'],  # y-coordinates from first distribution
            dist_high['cdf'],  # y-coordinates from second distribution
            frac  # interpolation fraction
        )
        
        # Create interpolated distribution
        interp_dist = {
            'intt': dist_low['intt'],  # Use the INTT from the lower energy
            'n_discrete': dist_low['n_discrete'],  # Use discrete count from lower energy
            'n_points': len(e_out_interp),
            'e_out': e_out_interp,
            'pdf': pdf_interp,
            'cdf': cdf_interp
        }
        
        return interp_dist
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the distribution for a given incident energy.
        
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
        # Get interpolated distribution
        dist = self.get_interpolated_distribution(incident_energy)
        if not dist:
            return 0.0
            
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
            
        # Generate random number
        xi = rng.random()
        
        # Sample from the distribution using the CDF
        e_out = np.interp(xi, dist['cdf'], dist['e_out'])
        
        return e_out

# Additional law classes as needed

@dataclass
class GeneralEvaporationSpectrum(EnergyDistribution):
    """
    Law 5: General Evaporation Spectrum.
    
    From ENDF-6, MF=5, LF=5, this represents an evaporation spectrum where:
    E_out = X(ξ) * θ(E)
    
    where X(ξ) is a randomly sampled value from equiprobable bins and
    θ(E) is the effective temperature tabulated on incident energy.
    
    Data format (Table 37):
    - N_R: Interpolation scheme between temperatures
    - NBT, INT: Interpolation parameters for temperatures
    - N_E: Number of incident energies
    - E(l): Incident energy table
    - θ(l): Effective temperature tabulated on incident energies
    - NET: Number of X's tabulated
    - X(l): Equiprobable bins
    """
    law: int = 5
    n_temp_interp_regions: int = 0  # Number of interpolation regions for temperature
    temp_nbt: List[int] = field(default_factory=list)  # Temperature interpolation region boundaries
    temp_interp: List[int] = field(default_factory=list)  # Temperature interpolation schemes
    n_incident_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy table
    temperatures: List[float] = field(default_factory=list)  # Effective temperature table
    n_equiprob_bins: int = 0  # Number of equiprobable bin boundaries
    equiprob_values: List[float] = field(default_factory=list)  # Equiprobable bin boundaries (X values)
    
    def get_temperature(self, incident_energy: float) -> float:
        """
        Get the effective temperature for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The effective temperature value
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
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the evaporation spectrum.
        
        E_out = X(ξ) * θ(E)
        
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
        
        # Get the effective temperature
        temperature = self.get_temperature(incident_energy)
        
        # Sample X value from equiprobable bins
        if not self.equiprob_values or len(self.equiprob_values) <= 1:
            return 0.0
            
        # Generate random bin index
        bin_idx = rng.integers(0, len(self.equiprob_values) - 1)
        x_value = self.equiprob_values[bin_idx]
        
        # Calculate outgoing energy
        e_out = x_value * temperature
        return e_out


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


@dataclass
class EvaporationSpectrum(EnergyDistribution):
    """
    Law 9: Evaporation Spectrum.
    
    From ENDF-6, MF=5, LF=9, this represents an evaporation spectrum where:
    f(E→E') = (sqrt(E_out) / I) * exp(-E_out / θ(E))
    
    where I is the normalization constant:
    I = θ^2 * [1 - exp(-(E - U) / θ) * (1 + (E - U) / θ)]
    
    Data format (Table 39):
    - N_R: Interpolation scheme between temperatures
    - NBT, INT: Interpolation parameters for temperatures
    - N_E: Number of incident energies
    - E(l): Incident energy table
    - θ(l): Effective temperature table
    - U: Restriction energy (upper limit constraint)
    """
    law: int = 9
    n_temp_interp_regions: int = 0  # Number of interpolation regions for temperature
    temp_nbt: List[int] = field(default_factory=list)  # Temperature interpolation region boundaries
    temp_interp: List[int] = field(default_factory=list)  # Temperature interpolation schemes
    n_incident_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy table
    temperatures: List[float] = field(default_factory=list)  # Effective temperature table
    restriction_energy: float = 0.0  # Restriction energy U (upper limit constraint)
    
    def get_temperature(self, incident_energy: float) -> float:
        """
        Get the effective temperature for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The effective temperature value
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
        
        I = θ^2 * [1 - exp(-(E - U) / θ) * (1 + (E - U) / θ)]
        
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
        # Calculate (E - U)/θ
        arg = (incident_energy - self.restriction_energy) / temperature
        if arg <= 0:
            return temperature * temperature  # Limiting case
        
        # Calculate I using equation 9
        exp_term = np.exp(-arg) * (1.0 + arg)
        normalization = temperature * temperature * (1.0 - exp_term)
        
        return max(normalization, 1.0e-30)  # Prevent division by zero
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the evaporation spectrum.
        
        This uses the evaporation spectrum equation:
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
        
        # Use rejection sampling to sample from the evaporation distribution
        while True:
            # Sample from exponential distribution with mean = temperature
            e_out = -temperature * np.log(rng.random())
            
            # Check if within the allowed range
            if e_out > max_e_out:
                continue
                
            # Acceptance probability proportional to sqrt(E_out)
            if rng.random() <= np.sqrt(e_out / temperature):
                return e_out


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
        
        # Calculate (E - U)
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


@dataclass
class TabularLinearFunctions(EnergyDistribution):
    """
    Law 22: Tabular Linear Functions of Incident Energy Out.
    
    From UK Law 2, this represents a tabular function form where the outgoing 
    energy is a linear function of the incident energy: E_out = C_ik * (E - T_ik).
    
    Data format (Table 41):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E_in(l): Tabulated incident energies
    - LOCE(l): Locators of E_out tables
    
    For each incident energy E_in(i):
    - NF_i: Number of functions for this energy
    - P_ik: Probability for each function
    - T_ik: Origin parameter for each function
    - C_ik: Slope parameter for each function
    """
    law: int = 22
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    table_locators: List[int] = field(default_factory=list)  # Locators of E_out tables
    
    # Store the function data for each incident energy
    # Each entry is a dictionary with 'nf', 'p', 't', 'c' keys
    function_data: List[Dict] = field(default_factory=list)
    
    def get_function_data(self, energy_idx: int) -> Dict:
        """
        Get the function data for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the function data with keys:
            - 'nf': Number of functions
            - 'p': Probability for each function
            - 't': Origin parameter for each function
            - 'c': Slope parameter for each function
            or None if index is invalid
        """
        if 0 <= energy_idx < len(self.function_data):
            return self.function_data[energy_idx]
        return None
    
    def get_interpolated_function_data(self, incident_energy: float) -> Dict:
        """
        Get interpolated function data for a specific incident energy.
        
        For Law 22, we don't interpolate between function data sets.
        Instead, we find the bracket incident energies and use the lower one's function data.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the function data or None if not available
        """
        # Find the bracketing incident energies
        if not self.incident_energies or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first function data
            return self.get_function_data(0) if self.function_data else None
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last function data
            return self.get_function_data(len(self.incident_energies) - 1) if self.function_data else None
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Return the function data for the lower incident energy
        return self.get_function_data(idx)
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the distribution for a given incident energy.
        
        For Law 22, we use equations:
        1. Find function index k such that: ∑(P_ij, j=1...k-1) < ξ ≤ ∑(P_ij, j=1...k)
        2. Calculate outgoing energy: E_out = C_ik * (E − T_ik)
        
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
        # Get function data for this incident energy
        function_data = self.get_interpolated_function_data(incident_energy)
        if not function_data:
            return 0.0
            
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
            
        # Generate random number
        xi = rng.random()
        
        # Get the probability, origin, and slope arrays
        nf = function_data['nf']
        p_values = function_data['p']
        t_values = function_data['t']
        c_values = function_data['c']
        
        # Find the function index k using cumulative probability
        cum_prob = 0.0
        k = 0
        for i in range(nf):
            cum_prob += p_values[i]
            if xi <= cum_prob:
                k = i
                break
        
        # Calculate outgoing energy using selected function
        # E_out = C_ik * (E − T_ik)
        e_out = c_values[k] * (incident_energy - t_values[k])
        
        # Ensure non-negative energy
        return max(0.0, e_out)


@dataclass
class TabularEnergyMultipliers(EnergyDistribution):
    """
    Law 24: Tabular Energy Multipliers.
    
    From UK Law 6, this represents a tabular function where the outgoing 
    energy is a multiplier of the incident energy: E_out = T_k(l) * E.
    
    Data format (Table 42):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E_in(l): Tabulated incident energies
    - NET: Number of outgoing values in each table
    - T_i(l): Tables of energy multipliers for each incident energy
    """
    law: int = 24
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    n_mult_values: int = 0  # Number of multiplier values in each table (NET)
    
    # Store the multiplier tables for each incident energy
    # Each row corresponds to one incident energy
    multiplier_tables: List[List[float]] = field(default_factory=list)
    
    def get_multiplier_table(self, energy_idx: int) -> List[float]:
        """
        Get the multiplier table for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        List[float]
            List of multiplier values or None if index is invalid
        """
        if 0 <= energy_idx < len(self.multiplier_tables):
            return self.multiplier_tables[energy_idx]
        return None
    
    def get_interpolated_multiplier_table(self, incident_energy: float) -> List[float]:
        """
        Get interpolated multiplier table for a specific incident energy.
        
        For tabular energy multipliers, we interpolate between the
        multiplier tables of adjacent incident energies.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        List[float]
            Interpolated multiplier table or None if not available
        """
        # Find the bracketing incident energies
        if not self.incident_energies or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first table
            return self.get_multiplier_table(0)
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last table
            return self.get_multiplier_table(len(self.incident_energies) - 1)
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Get the multiplier tables for the bracketing energies
        table_low = self.get_multiplier_table(idx)
        table_high = self.get_multiplier_table(idx + 1)
        
        if not table_low or not table_high or len(table_low) != len(table_high):
            return table_low if table_low else table_high
            
        # Calculate interpolation factor
        energy_low = self.incident_energies[idx]
        energy_high = self.incident_energies[idx + 1]
        factor = (incident_energy - energy_low) / (energy_high - energy_low)
        
        # Linearly interpolate between tables
        interp_table = [table_low[i] + factor * (table_high[i] - table_low[i]) for i in range(len(table_low))]
        
        return interp_table
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the distribution for a given incident energy.
        
        For Law 24, the outgoing energy is: E_out = T * E
        where T is a multiplier sampled from the tables.
        
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
        # Get interpolated multiplier table for this incident energy
        multiplier_table = self.get_interpolated_multiplier_table(incident_energy)
        if not multiplier_table or len(multiplier_table) <= 1:
            return 0.0
            
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
            
        # Generate random number for equiprobable bins
        bin_idx = rng.integers(0, len(multiplier_table) - 1)
        
        # Sample multiplier value from the bin
        multiplier = multiplier_table[bin_idx]
        
        # Calculate outgoing energy: E_out = T * E
        e_out = multiplier * incident_energy
        
        # Ensure non-negative energy
        return max(0.0, e_out)


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


@dataclass
class TabulatedAngleEnergyDistribution(EnergyDistribution):
    """
    Law 61: Tabulated Angle-Energy Distribution.
    
    Similar to LAW=44 but with tabular angular distributions instead of the Kalbach-Mann formalism.
    
    Data format (Table 45 and 46):
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
    - LC(l): Location of angular distribution tables
    
    For each angular distribution (Table 47):
    - JJ: Interpolation flag
    - N_p: Number of points
    - CosOut(j): Cosine scattering angular grid
    - PDF(j): Probability density function
    - CDF(j): Cumulative density function
    """
    law: int = 61
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each energy-angle distribution
    # Each entry is a dictionary containing the distribution data
    distributions: List[Dict] = field(default_factory=list)
    
    # Store the angular distribution tables
    # Each table is a dictionary with 'jj', 'n_points', 'cosines', 'pdf', and 'cdf' keys
    angular_tables: List[Dict] = field(default_factory=list)
    
    def get_distribution(self, energy_idx: int) -> Dict:
        """
        Get the energy distribution for a specific incident energy index.
        
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
            - 'lc': Location of angular distribution tables
            or None if index is invalid
        """
        if 0 <= energy_idx < len(self.distributions):
            return self.distributions[energy_idx]
        return None
    
    def get_angular_table(self, table_idx: int) -> Dict:
        """
        Get the angular distribution table with the specified index.
        
        Parameters
        ----------
        table_idx : int
            Index of the angular table
            
        Returns
        -------
        Dict
            Dictionary containing the angular distribution with keys:
            - 'jj': Interpolation flag
            - 'n_points': Number of points
            - 'cosines': Cosine scattering angular grid
            - 'pdf': Probability density function
            - 'cdf': Cumulative density function
            or None if index is invalid
        """
        if 0 <= table_idx < len(self.angular_tables):
            return self.angular_tables[table_idx]
        return None
    
    def get_interpolated_distribution(self, incident_energy: float) -> Dict:
        """
        Get an interpolated energy distribution for a specific incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the interpolated distribution data
        """
        # Find the bracketing incident energies
        if not self.incident_energies or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first distribution
            return self.get_distribution(0) if self.distributions else None
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last distribution
            return self.get_distribution(len(self.incident_energies) - 1) if self.distributions else None
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
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
        
        # LC values can't be interpolated since they're indices
        # We'll use the nearest LC value based on interpolation fraction
        lc_values = dist_low['lc'] if frac < 0.5 else dist_high['lc']
        
        # Create interpolated distribution
        interp_dist = {
            'intt': dist_low['intt'],  # Use the INTT from the lower energy
            'n_discrete': dist_low['n_discrete'],  # Use discrete count from lower energy
            'n_points': len(e_out),
            'e_out': e_out,
            'pdf': pdf_interp.tolist(),
            'cdf': cdf_interp.tolist(),
            'lc': lc_values
        }
        
        return interp_dist
    
    def sample_angular_distribution(self, table_idx: int, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample a cosine from a tabular angular distribution.
        
        Parameters
        ----------
        table_idx : int
            Index of the angular distribution table
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        float
            Sampled cosine of scattering angle
        """
        # Get the angular table
        table = self.get_angular_table(table_idx)
        if not table:
            # If table not found, return isotropic scattering
            if rng is None:
                rng = np.random.default_rng()
            return 2.0 * rng.random() - 1.0
        
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
        
        # Sample using CDF
        xi = rng.random()
        cosine = np.interp(xi, table['cdf'], table['cosines'])
        
        return cosine
    
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
        
        # 3. Get the corresponding LC value (index to angular distribution table)
        lc_idx = abs(dist['lc'][idx]) - 1  # Convert to 0-indexed and handle negative values
        
        # 4. Sample cosine from the angular distribution
        cosine = self.sample_angular_distribution(lc_idx, rng)
        
        return e_out, cosine
    
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


@dataclass
class LaboratoryAngleEnergyDistribution(EnergyDistribution):
    """
    Law 67: Laboratory Angle-Energy Distribution.
    
    From ENDF-6 MF=6 LAW=7, this represents a distribution that directly 
    specifies the angular and energy distributions in the laboratory frame.
    
    Data format (Table 49, 50, 51):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E(l): Incident energy grid
    - L(l): Locations of distributions
    
    For each incident energy:
    - INTMU: Interpolation scheme for angles
    - NMU: Number of secondary cosines
    - XMU(l): Secondary cosines
    - LMU(l): Locations of secondary energy distributions
    
    For each cosine:
    - INTEP: Interpolation parameter for secondary energies
    - NPEP: Number of secondary energies
    - E_p(l): Secondary energy grid
    - PDF(l): Probability density function
    - CDF(l): Cumulative density function
    """
    law: int = 67
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each angle-energy distribution
    # Each entry corresponds to one incident energy
    # Format: List of dictionaries with keys:
    # - 'intmu': Interpolation scheme for angles
    # - 'n_cosines': Number of secondary cosines
    # - 'cosines': Array of secondary cosines
    # - 'energy_dist_locations': Locations of energy distributions for each cosine
    # - 'energy_distributions': List of energy distributions for each cosine
    angle_energy_distributions: List[Dict] = field(default_factory=list)
    
    def get_distribution(self, energy_idx: int) -> Dict:
        """
        Get the angle-energy distribution for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the distribution data or None if index is invalid
        """
        if 0 <= energy_idx < len(self.angle_energy_distributions):
            return self.angle_energy_distributions[energy_idx]
        return None
    
    def get_interpolated_distribution(self, incident_energy: float) -> Dict:
        """
        Get an interpolated distribution for a specific incident energy.
        
        For Law 67, we find the bracketing incident energies and use the
        distribution for the lower energy. In a full implementation,
        we would interpolate between the distributions.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the distribution data or None if not available
        """
        # Find the bracketing incident energies
        if not self.incident_energies or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first distribution
            return self.get_distribution(0)
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last distribution
            return self.get_distribution(len(self.incident_energies) - 1)
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Return the distribution for the lower incident energy
        # In a full implementation, we would interpolate between distributions
        return self.get_distribution(idx)
    
    def sample_outgoing_angle_energy(self, incident_energy: float, 
                                    rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """
        Sample an outgoing cosine and energy from the distribution.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        Tuple[float, float]
            Tuple of (cosine, outgoing_energy)
        """
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
        
        # Get the distribution for this incident energy
        dist = self.get_interpolated_distribution(incident_energy)
        if not dist:
            return 0.0, 0.0
        
        # Sample cosine
        intmu = dist['intmu']
        cosines = dist['cosines']
        
        if not cosines or len(cosines) < 2:
            # If no valid cosines, return isotropic scattering
            return 2.0 * rng.random() - 1.0, 0.0
        
        # Choose a random cosine from the tabulated values
        # In a full implementation, we would use proper sampling according to intmu
        xi = rng.random()
        cosine_idx = int(xi * (len(cosines) - 1))
        cosine = cosines[cosine_idx]
        
        # Get the energy distribution for this cosine
        energy_dists = dist['energy_distributions']
        if not energy_dists or cosine_idx >= len(energy_dists) or not energy_dists[cosine_idx]:
            return cosine, 0.0
        
        energy_dist = energy_dists[cosine_idx]
        
        # Sample outgoing energy
        intep = energy_dist['intep']
        e_out = energy_dist['e_out']
        pdf = energy_dist['pdf']
        cdf = energy_dist['cdf']
        
        if not e_out or len(e_out) < 2:
            return cosine, 0.0
        
        # Sample from CDF
        xi = rng.random()
        outgoing_energy = np.interp(xi, cdf, e_out)
        
        return cosine, outgoing_energy
    
    def sample_outgoing_energy(self, incident_energy: float, 
                              rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample just an outgoing energy from the distribution.
        
        This is a convenience method that first samples an angle, then 
        samples an energy for that angle.
        
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
        _, energy = self.sample_outgoing_angle_energy(incident_energy, rng)
        return energy


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


@dataclass
class EnergyDistributionContainer:
    """Container for energy distributions of secondary particles."""
    incident_neutron: Dict[int, List[EnergyDistribution]] = field(default_factory=dict)
    photon_production: Dict[int, List[EnergyDistribution]] = field(default_factory=dict)
    particle_production: List[Dict[int, List[EnergyDistribution]]] = field(default_factory=list)
    delayed_neutron: List[EnergyDistribution] = field(default_factory=list)
    
    # Add containers for energy-dependent yields
    neutron_yields: Dict[int, EnergyDependentYield] = field(default_factory=dict)
    photon_yields: Dict[int, EnergyDependentYield] = field(default_factory=dict)
    particle_yields: List[Dict[int, EnergyDependentYield]] = field(default_factory=list)
