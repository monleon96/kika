from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.classes.energy_distribution.distributions.watt import EnergyDependentWattSpectrum


def parse_energy_dependent_watt_spectrum(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> EnergyDependentWattSpectrum:
    """
    Parse an energy-dependent Watt spectrum (Law 11).
    
    According to Table 40, LAW=11 contains:
    - First section: parameters for a(E)
      - LDAT(1): N_Ra - Number of interpolation regions for parameter a
      - LDAT(2) to LDAT(1+N_Ra): NBT_a - Interpolation parameters for a
      - LDAT(2+N_Ra) to LDAT(1+2*N_Ra): INT_a - Interpolation scheme for a
      - LDAT(2+2*N_Ra): N_Ea - Number of incident energies for a
      - LDAT(3+2*N_Ra) to LDAT(2+2*N_Ra+N_Ea): E_a - Incident energy table for a
      - LDAT(3+2*N_Ra+N_Ea) to LDAT(2+2*N_Ra+2*N_Ea): a - Parameter a values
    
    - Second section: parameters for b(E)
      - LDAT(L): N_Rb - Number of interpolation regions for parameter b
      - LDAT(L+1) to LDAT(L+N_Rb): NBT_b - Interpolation parameters for b
      - LDAT(L+1+N_Rb) to LDAT(L+1+2*N_Rb): INT_b - Interpolation scheme for b
      - LDAT(L+1+2*N_Rb): N_Eb - Number of incident energies for b
      - LDAT(L+2+2*N_Rb) to LDAT(L+1+2*N_Rb+N_Eb): E_b - Incident energy table for b
      - LDAT(L+2+2*N_Rb+N_Eb) to LDAT(L+1+2*N_Rb+2*N_Eb): b - Parameter b values
    
    - Final value:
      - LDAT(L+2+2*N_Rb+2*N_Eb): U - Restriction energy
    
    where L = 3 + 2 * (N_Ra + N_Ea)
    
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
    EnergyDependentWattSpectrum
        Energy-dependent Watt spectrum object
    """
    # Create a new distribution object using the base properties
    distribution = EnergyDependentWattSpectrum(
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
    
    # ---- First section: parameters for a(E) ----
    
    # Read the number of interpolation regions for parameter a (N_Ra)
    n_a_interp_regions_entry = ace.xss_data[idat_idx]
    distribution.n_a_interp_regions = int(n_a_interp_regions_entry.value)
    n_ra = distribution.n_a_interp_regions
    idx = idat_idx + 1
    
    # Read interpolation parameters for a if present
    if n_ra > 0:
        # Read NBT_a values
        if idx + n_ra <= len(ace.xss_data):
            distribution.a_nbt = [int(ace.xss_data[idx + i].value) for i in range(n_ra)]
            idx += n_ra
        
        # Read INT_a values
        if idx + n_ra <= len(ace.xss_data):
            distribution.a_interp = [int(ace.xss_data[idx + i].value) for i in range(n_ra)]
            idx += n_ra
    
    # Read the number of incident energies for parameter a (N_Ea)
    if idx < len(ace.xss_data):
        n_a_energies_entry = ace.xss_data[idx]
        distribution.n_a_energies = int(n_a_energies_entry.value)
        n_ea = distribution.n_a_energies
        idx += 1
    else:
        return distribution
    
    # Read the incident energy table for parameter a - store the XssEntry objects
    if idx + n_ea <= len(ace.xss_data):
        distribution.a_incident_energies = [ace.xss_data[idx + i] for i in range(n_ea)]
        idx += n_ea
    else:
        return distribution
    
    # Read the a parameter values - store the XssEntry objects
    if idx + n_ea <= len(ace.xss_data):
        distribution.a_values = [ace.xss_data[idx + i] for i in range(n_ea)]
        idx += n_ea
    else:
        return distribution
    
    # Calculate L = 3 + 2 * (N_Ra + N_Ea)
    l_idx = 3 + 2 * (n_ra + n_ea)
    
    # ---- Second section: parameters for b(E) ----
    
    # Read the number of interpolation regions for parameter b (N_Rb)
    if idx < len(ace.xss_data):
        n_b_interp_regions_entry = ace.xss_data[idx]
        distribution.n_b_interp_regions = int(n_b_interp_regions_entry.value)
        n_rb = distribution.n_b_interp_regions
        idx += 1
    else:
        return distribution
    
    # Read interpolation parameters for b if present
    if n_rb > 0:
        # Read NBT_b values
        if idx + n_rb <= len(ace.xss_data):
            distribution.b_nbt = [int(ace.xss_data[idx + i].value) for i in range(n_rb)]
            idx += n_rb
        
        # Read INT_b values
        if idx + n_rb <= len(ace.xss_data):
            distribution.b_interp = [int(ace.xss_data[idx + i].value) for i in range(n_rb)]
            idx += n_rb
    
    # Read the number of incident energies for parameter b (N_Eb)
    if idx < len(ace.xss_data):
        n_b_energies_entry = ace.xss_data[idx]
        distribution.n_b_energies = int(n_b_energies_entry.value)
        n_eb = distribution.n_b_energies
        idx += 1
    else:
        return distribution
    
    # Read the incident energy table for parameter b - store the XssEntry objects
    if idx + n_eb <= len(ace.xss_data):
        distribution.b_incident_energies = [ace.xss_data[idx + i] for i in range(n_eb)]
        idx += n_eb
    else:
        return distribution
    
    # Read the b parameter values - store the XssEntry objects
    if idx + n_eb <= len(ace.xss_data):
        distribution.b_values = [ace.xss_data[idx + i] for i in range(n_eb)]
        idx += n_eb
    else:
        return distribution
    
    # ---- Final value: restriction energy ----
    
    # Read the restriction energy (U)
    if idx < len(ace.xss_data):
        restriction_energy_entry = XssEntry(idx, ace.xss_data[idx])
        distribution.restriction_energy = restriction_energy_entry.value
    
    return distribution