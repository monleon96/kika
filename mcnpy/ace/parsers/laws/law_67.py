from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution
from mcnpy.ace.classes.energy_distribution.distributions.angle_energy import LaboratoryAngleEnergyDistribution


def parse_laboratory_angle_energy_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> LaboratoryAngleEnergyDistribution:
    """
    Parse a laboratory angle-energy distribution (Law 67).
    
    According to Table 49, 50, 51, LAW=67 contains:
    - LDAT(1): N_R - Number of interpolation regions
    - LDAT(2) to LDAT(1+N_R): NBT - Interpolation parameters
    - LDAT(2+N_R) to LDAT(1+2*N_R): INT - Interpolation schemes
    - LDAT(2+2*N_R): N_E - Number of incident energies
    - LDAT(3+2*N_R) to LDAT(2+2*N_R+N_E): E(l) - Incident energy grid
    - LDAT(3+2*N_R+N_E) to LDAT(2+2*N_R+2*N_E): L(l) - Locations of distributions
    
    For each incident energy E(i), the angle distribution contains:
    - INTMU: Interpolation scheme for angles
    - NMU: Number of secondary cosines
    - XMU(l): Secondary cosines
    - LMU(l): Locations of energy distributions for each cosine
    
    For each secondary cosine XMU(j), the energy distribution contains:
    - INTEP: Interpolation parameter for secondary energies
    - NPEP: Number of secondary energies
    - E_p(l): Secondary energy grid
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
    LaboratoryAngleEnergyDistribution
        Laboratory angle-energy distribution object
    """
    # Create a new distribution object using the base properties
    distribution = LaboratoryAngleEnergyDistribution(
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
    
    # Initialize the angle-energy distributions list
    distribution.angle_energy_distributions = []
    
    # Get the JXS values for the relevant blocks
    jxs_dlw = ace.header.jxs_array[10] - 1  # JXS(11), convert to 0-indexed
    jxs_dlwp = ace.header.jxs_array[18] - 1  # JXS(19), convert to 0-indexed
    jxs_dned = ace.header.jxs_array[26] - 1  # JXS(27), convert to 0-indexed
    
    # Now read each angle-energy distribution
    for i in range(n_e):
        # Get the location of this distribution
        loc = distribution.distribution_locations[i]
        if loc <= 0:
            # Skip if location is invalid
            distribution.angle_energy_distributions.append(None)
            continue
        
        # Calculate base address depending on data type
        # For simplicity, we'll use JXS(11) for now, but in a full implementation
        # we'd need to determine which JXS to use based on the data type
        base_idx = jxs_dlw
        
        # Convert to absolute index
        dist_idx = base_idx + loc - 1  # -1 for 0-indexing
        
        # Check if we're within bounds
        if dist_idx >= len(ace.xss_data):
            distribution.angle_energy_distributions.append(None)
            continue
        
        # Read INTMU (interpolation scheme for angles)
        intmu = int(ace.xss_data[dist_idx].value)
        
        # Read NMU (number of secondary cosines)
        nmu = int(ace.xss_data[dist_idx + 1].value)
        
        # Read the secondary cosines (XMU) - store the XssEntry objects
        if dist_idx + 2 + nmu - 1 >= len(ace.xss_data):
            distribution.angle_energy_distributions.append(None)
            continue
            
        cosines = [ace.xss_data[dist_idx + 2 + j] for j in range(nmu)]
        
        # Read the energy distribution locations (LMU)
        if dist_idx + 2 + nmu + nmu - 1 >= len(ace.xss_data):
            distribution.angle_energy_distributions.append(None)
            continue
            
        lmu_values = [int(ace.xss_data[dist_idx + 2 + nmu + j].value) for j in range(nmu)]
        
        # For each cosine, read the energy distribution
        energy_distributions = []
        
        for j in range(nmu):
            lmu = lmu_values[j]
            if lmu <= 0:
                # Skip if location is invalid
                energy_distributions.append(None)
                continue
                
            # Calculate energy distribution index
            # Try with both neutron reactions and photon production
            energy_dist_idx = jxs_dlw + lmu - 1  # First try with neutron reactions
            
            # If out of bounds, try with photon production
            if energy_dist_idx >= len(ace.xss_data):
                energy_dist_idx = jxs_dlwp + lmu - 1
                
            # Skip if still out of bounds
            if energy_dist_idx >= len(ace.xss_data):
                energy_distributions.append(None)
                continue
                
            # Read INTEP (interpolation parameter for secondary energies)
            intep = int(ace.xss_data[energy_dist_idx].value)
            
            # Read NPEP (number of secondary energies)
            npep = int(ace.xss_data[energy_dist_idx + 1].value)
            
            # Read the secondary energy grid (E_p) - store the XssEntry objects
            if energy_dist_idx + 2 + npep - 1 >= len(ace.xss_data):
                energy_distributions.append(None)
                continue
                
            e_p = [ace.xss_data[energy_dist_idx + 2 + k] for k in range(npep)]
            
            # Read the probability density function (PDF) - store the XssEntry objects
            if energy_dist_idx + 2 + npep + npep - 1 >= len(ace.xss_data):
                energy_distributions.append(None)
                continue
                
            pdf = [ace.xss_data[energy_dist_idx + 2 + npep + k] for k in range(npep)]
            
            # Read the cumulative density function (CDF) - store the XssEntry objects
            if energy_dist_idx + 2 + 2*npep + npep - 1 >= len(ace.xss_data):
                energy_distributions.append(None)
                continue
                
            cdf = [ace.xss_data[energy_dist_idx + 2 + 2*npep + k] for k in range(npep)]
            
            # Store the energy distribution
            energy_dist = {
                'intep': intep,
                'npep': npep,
                'e_out': e_p,
                'pdf': pdf,
                'cdf': cdf
            }
            
            energy_distributions.append(energy_dist)
        
        # Store the angle-energy distribution
        angle_energy_dist = {
            'intmu': intmu,
            'n_cosines': nmu,
            'cosines': cosines,
            'energy_dist_locations': lmu_values,
            'energy_distributions': energy_distributions
        }
        
        distribution.angle_energy_distributions.append(angle_energy_dist)
    
    return distribution