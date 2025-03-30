from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution.energy_distribution import EnergyDistribution, NBodyPhaseSpaceDistribution

def parse_nbody_phase_space_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> NBodyPhaseSpaceDistribution:
    """
    Parse an N-body phase space distribution (Law 66).
    
    According to Table 48, LAW=66 contains:
    - LDAT(1): NPSX - Number of bodies in the phase space
    - LDAT(2): A_P - Total mass ratio for the NPSX particles
    - LDAT(3): INTT - Interpolation parameter
    - LDAT(4): N_P - Number of points in the distribution
    - LDAT(5) to LDAT(4+N_P): ξ_out(j) - ξ grid (between 0 and 1)
    - LDAT(5+N_P) to LDAT(4+2*N_P): PDF(j) - Probability density function
    - LDAT(5+2*N_P) to LDAT(4+3*N_P): CDF(j) - Cumulative density function
    
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
    NBodyPhaseSpaceDistribution
        N-body phase space distribution object
    """
    # Create a new distribution object using the base properties
    distribution = NBodyPhaseSpaceDistribution(
        law=base_dist.law,
        idat=base_dist.idat
    )
    
    # Copy applicability data from base_dist
    distribution.applicability_energies = base_dist.applicability_energies
    distribution.applicability_probabilities = base_dist.applicability_probabilities
    distribution.nbt = base_dist.nbt
    distribution.interp = base_dist.interp
    
    # Check if we have data to parse
    if idat_idx + 1 >= len(ace.xss_data):
        return distribution
    
    # Read NPSX (number of bodies in the phase space)
    distribution.npsx = int(ace.xss_data[idat_idx].value)
    
    # Read A_P (total mass ratio for the NPSX particles)
    if idat_idx + 2 <= len(ace.xss_data):
        distribution.ap = ace.xss_data[idat_idx + 1]
    
    # Read INTT (interpolation parameter)
    if idat_idx + 3 <= len(ace.xss_data):
        distribution.intt = int(ace.xss_data[idat_idx + 2].value)
    
    # Read N_P (number of points in the distribution)
    if idat_idx + 4 <= len(ace.xss_data):
        distribution.n_points = int(ace.xss_data[idat_idx + 3].value)
    
    # Get number of points
    n_p = distribution.n_points
    
    # Read ξ grid - store the XssEntry objects
    if idat_idx + 4 + n_p <= len(ace.xss_data):
        distribution.xi_grid = [ace.xss_data[idat_idx + 4 + i] for i in range(n_p)]
    
    # Read PDF - store the XssEntry objects
    if idat_idx + 4 + n_p + n_p <= len(ace.xss_data):
        distribution.pdf = [ace.xss_data[idat_idx + 4 + n_p + i] for i in range(n_p)]
    
    # Read CDF - store the XssEntry objects
    if idat_idx + 4 + n_p + n_p + n_p <= len(ace.xss_data):
        distribution.cdf = [ace.xss_data[idat_idx + 4 + 2*n_p + i] for i in range(n_p)]
    
    return distribution