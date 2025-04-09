from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution
from mcnpy.ace.classes.energy_distribution.distributions.evaporation import GeneralEvaporationSpectrum

def parse_general_evaporation_spectrum(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> GeneralEvaporationSpectrum:
    """
    Parse a general evaporation spectrum (Law 5).
    
    According to Table 37, LAW=5 contains:
    - LDAT(1): N_R - Number of interpolation regions between temperatures
    - LDAT(2) to LDAT(1+N_R): NBT - Interpolation parameters for temperatures
    - LDAT(2+N_R) to LDAT(1+2*N_R): INT - Interpolation scheme for temperatures
    - LDAT(2+2*N_R): N_E - Number of incident energies tabulated
    - LDAT(3+2*N_R) to LDAT(2+2*N_R+N_E): E(l) - Incident energy table
    - LDAT(3+2*N_R+N_E) to LDAT(2+2*N_R+2*N_E): θ(l) - Effective temperature table
    - LDAT(3+2*N_R+2*N_E): NET - Number of X's tabulated
    - LDAT(4+2*N_R+2*N_E) to end: X(l) - Equiprobable bins
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    base_dist : EnergyDistribution
        Base distribution with common properties
    idat_idx : int
        Starting index for the law data in the XSS array
        
    Returns
    -------
    GeneralEvaporationSpectrum
        General evaporation spectrum object
    """
    # Create a new distribution object using the base properties
    distribution = GeneralEvaporationSpectrum(
        law=base_dist.law,
        idat=base_dist.idat
    )
    
    # Copy applicability data from base_dist
    distribution.applicability_energies = base_dist.applicability_energies
    distribution.applicability_probabilities = base_dist.applicability_probabilities
    distribution.nbt = base_dist.nbt
    distribution.interp = base_dist.interp
    
    # Check if we have data to parse
    if idat_idx >= len(ace.xss_data):
        return distribution
    
    # Read the number of interpolation regions for temperature (N_R)
    n_temp_interp_regions_entry = ace.xss_data[idat_idx]
    distribution.n_temp_interp_regions = int(n_temp_interp_regions_entry.value)
    n_r = distribution.n_temp_interp_regions
    idx = idat_idx + 1
    
    # Read interpolation parameters for temperature if present
    if n_r > 0:
        # Read NBT values for temperature interpolation
        if idx + n_r <= len(ace.xss_data):
            distribution.temp_nbt = [int(ace.xss_data[idx + i].value) for i in range(n_r)]
            idx += n_r
        
        # Read INT values for temperature interpolation
        if idx + n_r <= len(ace.xss_data):
            distribution.temp_interp = [int(ace.xss_data[idx + i].value) for i in range(n_r)]
            idx += n_r
    
    # Read the number of incident energies (N_E)
    if idx < len(ace.xss_data):
        n_incident_energies_entry = ace.xss_data[idx]
        distribution.n_incident_energies = int(n_incident_energies_entry.value)
        n_e = distribution.n_incident_energies
        idx += 1
    else:
        return distribution
    
    # Read the incident energy table - store the XssEntry objects
    if idx + n_e <= len(ace.xss_data):
        distribution.incident_energies = [ace.xss_data[idx + i] for i in range(n_e)]
        idx += n_e
    else:
        return distribution
    
    # Read the temperature table - store the XssEntry objects
    if idx + n_e <= len(ace.xss_data):
        distribution.temperatures = [ace.xss_data[idx + i] for i in range(n_e)]
        idx += n_e
    else:
        return distribution
    
    # Read the number of equiprobable bin values (NET)
    if idx < len(ace.xss_data):
        n_equiprob_bins_entry = ace.xss_data[idx]
        distribution.n_equiprob_bins = int(n_equiprob_bins_entry.value)
        net = distribution.n_equiprob_bins
        idx += 1
    else:
        return distribution
    
    # Read the equiprobable bin values - store the XssEntry objects
    if idx + net <= len(ace.xss_data):
        distribution.equiprob_values = [ace.xss_data[idx + i] for i in range(net)]
    
    return distribution