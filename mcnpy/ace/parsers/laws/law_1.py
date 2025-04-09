from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution
from mcnpy.ace.classes.energy_distribution.distributions.tabular import TabularEnergyDistribution

def parse_tabular_energy_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> TabularEnergyDistribution:
    """
    Parse a tabular energy distribution (Law 1).
    
    According to Table 32, LAW=1 represents Tabular Equiprobable Energy Bins (From ENDF Law 1):
    - LDAT(1): Number of interpolation regions between tables of E_out (N_R)
    - LDAT(2) to LDAT(1+N_R): NBT interpolation parameters
    - LDAT(2+N_R) to LDAT(1+2*N_R): INT interpolation schemes
    - LDAT(2+2*N_R): Number of incident energies (N_E)
    - LDAT(3+2*N_R) to LDAT(2+2*N_R+N_E): List of incident energies
    - LDAT(3+2*N_R+N_E): Number of outgoing energies in each table (NET)
    - LDAT(4+2*N_R+N_E) to end: E_out tables, each with NET entries
    
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
    TabularEnergyDistribution
        Tabular energy distribution object
    """
    # Create a new distribution object using the base properties
    distribution = TabularEnergyDistribution(
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
    n_r_entry = ace.xss_data[idat_idx]
    n_r = int(n_r_entry.value)
    idx = idat_idx + 1
    
    # Read the interpolation parameters if present
    e_out_nbt = []
    e_out_int = []
    if n_r > 0:
        # Read NBT values - store references to the original XssEntry objects
        e_out_nbt = [ace.xss_data[idx + i] for i in range(n_r)]
        idx += n_r
        
        # Read INT values - store references to the original XssEntry objects
        e_out_int = [ace.xss_data[idx + i] for i in range(n_r)]
        idx += n_r
    
    # Read the number of incident energies (N_E)
    if idx >= len(ace.xss_data):
        return distribution
    
    n_e_entry = ace.xss_data[idx]
    n_e = int(n_e_entry.value)
    idx += 1
    distribution.n_incident_energies = n_e
    
    # Check if we have enough data
    if idx + n_e >= len(ace.xss_data):
        return distribution
    
    # Read the list of incident energies - store references to the original XssEntry objects
    incident_energies = [ace.xss_data[idx + i] for i in range(n_e)]
    idx += n_e
    distribution.incident_energies = incident_energies
    
    # Read the number of outgoing energies (NET) in each table
    if idx >= len(ace.xss_data):
        return distribution
    
    net_entry = ace.xss_data[idx]
    net = int(net_entry.value)
    idx += 1
    
    # Read the E_out tables for each incident energy
    # Each table has NET entries
    distribution.distribution_data = []
    
    for i in range(n_e):
        if idx + net > len(ace.xss_data):
            break
        
        # Get the outgoing energy boundaries for this incident energy
        e_out = [ace.xss_data[idx + j] for j in range(net)]
        idx += net
        
        # Calculate the probability (uniform across bins)
        p_out = [1.0 / (net - 1)] * (net - 1) + [0.0]  # NET-1 equal probability bins
        
        # Store with interpolation scheme (always linear-linear for equiprobable bins)
        distribution.distribution_data.append((net, 2, e_out, p_out))
    
    return distribution