from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution import EnergyDistribution, KalbachMannDistribution

def parse_kalbach_mann_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> KalbachMannDistribution:
    """
    Parse a Kalbach-Mann correlated energy-angle distribution (Law 44).
    
    According to Table 43 and 44, LAW=44 contains:
    - LDAT(1): N_R (number of interpolation regions)
    - LDAT(2..1+N_R): NBT interpolation parameters
    - LDAT(2+N_R..1+2*N_R): INT interpolation schemes
    - LDAT(2+2*N_R): N_E (number of incident energies)
    - LDAT(3+2*N_R..2+2*N_R+N_E): E(l) (incident energies)
    - LDAT(3+2*N_R+N_E..2+2*N_R+2*N_E): L(l) (distribution locations)
    
    For each incident energy E(i), the distribution contains:
    - INTT': Combined interpolation parameter (10*N_D + INTT)
    - N_p: Number of points in the distribution
    - E_out(l): Outgoing energy grid
    - PDF(l): Probability density function
    - CDF(l): Cumulative density function
    - R(l): Precompound fraction
    - A(l): Angular distribution slope
    
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
    KalbachMannDistribution
        Kalbach-Mann distribution object
    """
    # Create a new distribution object using the base properties
    distribution = KalbachMannDistribution(
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
    
    # Check if we have enough data
    if idx + n_e - 1 >= len(ace.xss_data):
        return distribution
    
    # Read the incident energies - store the XssEntry objects
    distribution.incident_energies = [ace.xss_data[idx + i] for i in range(n_e)]
    idx += n_e
    
    # Check if we have enough data for the locations
    if idx + n_e - 1 >= len(ace.xss_data):
        return distribution
    
    # Read the distribution locations (L values)
    distribution.distribution_locations = [int(ace.xss_data[idx + i].value) for i in range(n_e)]
    
    # Now read each distribution and store it
    distribution.distributions = []
    jed = idat_idx  # Base address for JED
    
    for i in range(n_e):
        # Get the location of this distribution
        loc = distribution.distribution_locations[i]
        if loc <= 0:
            # Skip if location is invalid
            distribution.distributions.append(None)
            continue
        
        # Convert to absolute index
        dist_idx = jed + loc - 1  # -1 for 0-indexing
        
        # Check if we're within bounds
        if dist_idx >= len(ace.xss_data):
            distribution.distributions.append(None)
            continue
        
        # Read the combined interpolation parameter (INTT')
        intt_prime = int(ace.xss_data[dist_idx].value)
        
        # Separate into N_D (number of discrete lines) and INTT (interpolation scheme)
        n_discrete = intt_prime // 10
        intt = intt_prime % 10
        
        # Read the number of points (N_p)
        n_points = int(ace.xss_data[dist_idx + 1].value)
        
        # Check if we have enough data for the full distribution
        # We need 5 arrays of length n_points (E_out, PDF, CDF, R, A)
        if dist_idx + 2 + 5*n_points - 1 >= len(ace.xss_data):
            distribution.distributions.append(None)
            continue
        
        # Read the outgoing energy grid (E_out)
        e_out = [ace.xss_data[dist_idx + 2 + j] for j in range(n_points)]
        
        # Read the probability density function (PDF)
        pdf = [ace.xss_data[dist_idx + 2 + n_points + j] for j in range(n_points)]
        
        # Read the cumulative density function (CDF)
        cdf = [ace.xss_data[dist_idx + 2 + 2*n_points + j] for j in range(n_points)]
        
        # Read the precompound fraction (R)
        r_values = [ace.xss_data[dist_idx + 2 + 3*n_points + j] for j in range(n_points)]
        
        # Read the angular distribution slope (A)
        a_values = [ace.xss_data[dist_idx + 2 + 4*n_points + j] for j in range(n_points)]
        
        # Store the distribution
        dist_data = {
            'intt': intt,
            'n_discrete': n_discrete,
            'n_points': n_points,
            'e_out': e_out,
            'pdf': pdf,
            'cdf': cdf,
            'r': r_values,
            'a': a_values
        }
        
        distribution.distributions.append(dist_data)
    
    return distribution