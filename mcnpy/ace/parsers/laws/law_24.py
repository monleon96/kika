from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution.energy_distribution import EnergyDistribution, TabularEnergyMultipliers

def parse_tabular_energy_multipliers(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> TabularEnergyMultipliers:
    """
    Parse a tabular energy multipliers distribution (Law 24).
    
    According to Table 42, LAW=24 contains:
    - LDAT(1): N_R - Number of interpolation regions
    - LDAT(2) to LDAT(1+N_R): NBT - Interpolation parameters
    - LDAT(2+N_R) to LDAT(1+2*N_R): INT - Interpolation schemes
    - LDAT(2+2*N_R): N_E - Number of incident energies
    - LDAT(3+2*N_R) to LDAT(2+2*N_R+N_E): E_in(l) - Incident energy table
    - LDAT(3+2*N_R+N_E): NET - Number of outgoing values in each table
    - LDAT(4+2*N_R+N_E) to end: Tables of multipliers for each incident energy
    
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
    TabularEnergyMultipliers
        Tabular energy multipliers distribution object
    """
    # Create a new distribution object using the base properties
    distribution = TabularEnergyMultipliers(
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
    
    # Read the number of interpolation regions (N_R)
    distribution.n_interp_regions = int(ace.xss_data[idat_idx].value)
    idx = idat_idx + 1
    n_r = distribution.n_interp_regions
    
    # Read the interpolation parameters if present
    if n_r > 0 and idx + 2*n_r - 1 < len(ace.xss_data):
        # Read NBT values
        distribution.nbt = [int(ace.xss_data[idx + i].value) for i in range(n_r)]
        idx += n_r
        
        # Read INT values
        distribution.interp = [int(ace.xss_data[idx + i].value) for i in range(n_r)]
        idx += n_r
    
    # Read the number of incident energies (N_E)
    if idx >= len(ace.xss_data):
        return distribution
    
    distribution.n_energies = int(ace.xss_data[idx].value)
    idx += 1
    n_e = distribution.n_energies
    
    # Read the incident energies - store the XssEntry objects
    if idx + n_e - 1 >= len(ace.xss_data):
        return distribution
    
    distribution.incident_energies = [ace.xss_data[idx + i] for i in range(n_e)]
    idx += n_e
    
    # Read the number of multiplier values (NET)
    if idx >= len(ace.xss_data):
        return distribution
    
    distribution.n_mult_values = int(ace.xss_data[idx].value)
    idx += 1
    net = distribution.n_mult_values
    
    # Read the multiplier tables - store the XssEntry objects
    distribution.multiplier_tables = []
    
    # Each table has NET values, and there are N_E tables
    for i in range(n_e):
        if idx + net - 1 >= len(ace.xss_data):
            break
        
        table = [ace.xss_data[idx + j] for j in range(net)]
        distribution.multiplier_tables.append(table)
        idx += net
    
    return distribution