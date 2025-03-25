from mcnpy.ace.ace import Ace
from mcnpy.ace.classes.energy_distribution import EnergyDependentYield

def parse_energy_dependent_yield(ace: Ace, ky_idx: int) -> EnergyDependentYield:
    """
    Parse energy-dependent neutron yield data.
    
    According to Table 52, the data format is:
    - KY: N_R - Number of interpolation regions
    - KY+1: NBT(l), l = 1,...,N_R - ENDF interpolation parameters
    - KY+1+N_R: INT(l), l = 1,...,N_R - ENDF interpolation scheme
    - KY+1+2N_R: N_E - Number of energies
    - KY+2+2N_R: E(l), l = 1,...,N_E - Tabular energy points
    - KY+2+N_R+N_E: Y(l), l = 1,...,N_E - Corresponding energy-dependent yields
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    ky_idx : int
        Starting index in the XSS array (KY)
        
    Returns
    -------
    EnergyDependentYield
        Energy-dependent yield object
    """
    # Create a new yield object
    yield_obj = EnergyDependentYield()
    
    # Ensure ky_idx is an integer
    ky_idx = int(ky_idx)
    
    # Check if we have data to parse
    if ky_idx >= len(ace.xss_data):
        return yield_obj
    
    # Read the number of interpolation regions (N_R)
    yield_obj.n_interp_regions = int(ace.xss_data[ky_idx].value)
    n_r = yield_obj.n_interp_regions
    idx = ky_idx + 1
    
    # Read interpolation parameters for regions if present
    if n_r > 0:
        # Read NBT values - store the XssEntry objects
        if idx + n_r <= len(ace.xss_data):
            yield_obj.nbt = [ace.xss_data[idx + i] for i in range(n_r)]
            idx += n_r
        
        # Read INT values - store the XssEntry objects
        if idx + n_r <= len(ace.xss_data):
            yield_obj.interp = [ace.xss_data[idx + i] for i in range(n_r)]
            idx += n_r
    
    # Read the number of energies (N_E)
    if idx < len(ace.xss_data):
        yield_obj.n_energies = int(ace.xss_data[idx].value)
        n_e = yield_obj.n_energies
        idx += 1
    else:
        return yield_obj
    
    # Read the tabular energy points - store the XssEntry objects
    if idx + n_e <= len(ace.xss_data):
        yield_obj.energies = [ace.xss_data[idx + i] for i in range(n_e)]
        idx += n_e
    else:
        return yield_obj
    
    # Read the corresponding yields - store the XssEntry objects
    if idx + n_e <= len(ace.xss_data):
        yield_obj.yields = [ace.xss_data[idx + i] for i in range(n_e)]
    
    return yield_obj
