from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution.energy_distribution import EnergyDistribution, ContinuousTabularDistribution
from mcnpy.ace.parsers.xss import XssEntry
from typing import List

def parse_continuous_energy_angle_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> ContinuousTabularDistribution:
    """
    Parse a continuous energy-angle distribution (Law 4).
    
    According to Table 35 and 36, LAW=4 contains:
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
    ContinuousTabularDistribution
        Continuous tabular distribution object
    """
    # Create a new distribution object using the base properties
    distribution = ContinuousTabularDistribution(
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
    n_interp_regions_entry = ace.xss_data[idat_idx]
    distribution.n_interp_regions = int(n_interp_regions_entry.value)
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
    
    n_energies_entry = ace.xss_data[idx]
    distribution.n_energies = int(n_energies_entry.value)
    idx += 1
    n_e = distribution.n_energies
    
    # Check if we have enough data
    if idx + n_e - 1 >= len(ace.xss_data):
        return distribution
    
    # Read the incident energies - store the original XssEntry objects
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
        intt_prime_entry = ace.xss_data[dist_idx]
        intt_prime = int(intt_prime_entry.value)
        
        # Separate into N_D (number of discrete lines) and INTT (interpolation scheme)
        n_discrete = intt_prime // 10
        intt = intt_prime % 10
        
        # Read the number of points (N_p)
        n_points_entry = ace.xss_data[dist_idx + 1]
        n_points = int(n_points_entry.value)
        
        # Check if we have enough data for the full distribution
        if dist_idx + 2 + 3*n_points - 1 >= len(ace.xss_data):
            distribution.distributions.append(None)
            continue
        
        # Read the outgoing energy grid (E_out)
        e_out = [ace.xss_data[dist_idx + 2 + j].value for j in range(n_points)]
        
        # Read the probability density function (PDF)
        pdf = [ace.xss_data[dist_idx + 2 + n_points + j].value for j in range(n_points)]
        
        # Read the cumulative density function (CDF)
        cdf = [ace.xss_data[dist_idx + 2 + 2*n_points + j].value for j in range(n_points)]
        
        # Store the distribution
        dist_data = {
            'intt': intt,
            'n_discrete': n_discrete,
            'n_points': n_points,
            'e_out': e_out,
            'pdf': pdf,
            'cdf': cdf
        }
        
        distribution.distributions.append(dist_data)
    
    return distribution