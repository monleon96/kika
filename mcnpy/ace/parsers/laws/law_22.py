from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution import EnergyDistribution, TabularLinearFunctions

def parse_tabular_linear_functions(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> TabularLinearFunctions:
    """
    Parse a tabular linear functions distribution (Law 22).
    
    According to Table 41, LAW=22 contains:
    - LDAT(1): N_R - Number of interpolation regions
    - LDAT(2) to LDAT(1+N_R): NBT - Interpolation parameters
    - LDAT(2+N_R) to LDAT(1+2*N_R): INT - Interpolation schemes
    - LDAT(2+2*N_R): N_E - Number of incident energies
    - LDAT(3+2*N_R) to LDAT(2+2*N_R+N_E): E_in(l) - Incident energy table
    - LDAT(3+2*N_R+N_E) to LDAT(2+2*N_R+2*N_E): LOCE(l) - Locators of E_out tables
    
    For each incident energy E_in(i), the data contains:
    - NF_i: Number of functions for this energy
    - P_ik: Probability for each function
    - T_ik: Origin parameter for each function
    - C_ik: Slope parameter for each function
    
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
    TabularLinearFunctions
        Tabular linear functions distribution object
    """
    # Create a new distribution object using the base properties
    distribution = TabularLinearFunctions(
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
    
    # Read the table locators
    if idx + n_e - 1 >= len(ace.xss_data):
        return distribution
    
    distribution.table_locators = [int(ace.xss_data[idx + i].value) for i in range(n_e)]
    idx += n_e
    
    # Initialize function_data list
    distribution.function_data = []
    
    # Read the function data for each incident energy
    base_idx = idat_idx  # Base address for locators
    
    for i in range(n_e):
        locator = distribution.table_locators[i]
        if locator <= 0:
            # Skip if locator is invalid
            distribution.function_data.append(None)
            continue
        
        # Calculate absolute index
        func_idx = base_idx + locator - 1  # -1 for 0-indexing
        
        # Check if we're within bounds
        if func_idx >= len(ace.xss_data):
            distribution.function_data.append(None)
            continue
        
        # Read number of functions (NF_i)
        nf = int(ace.xss_data[func_idx].value)
        
        # Check if we have enough data
        if func_idx + 1 + 3*nf - 1 >= len(ace.xss_data):
            distribution.function_data.append(None)
            continue
        
        # Read probability values (P_ik)
        p_values = [ace.xss_data[func_idx + 1 + j] for j in range(nf)]
        
        # Read origin parameter values (T_ik)
        t_values = [ace.xss_data[func_idx + 1 + nf + j] for j in range(nf)]
        
        # Read slope parameter values (C_ik)
        c_values = [ace.xss_data[func_idx + 1 + 2*nf + j] for j in range(nf)]
        
        # Store function data
        func_data = {
            'nf': nf,
            'p': p_values,
            't': t_values,
            'c': c_values
        }
        
        distribution.function_data.append(func_data)
    
    return distribution