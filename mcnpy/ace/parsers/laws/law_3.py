from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution
from mcnpy.ace.classes.energy_distribution.distributions.level_scattering import LevelScattering

def parse_level_scattering(ace: Ace, base_dist: EnergyDistribution, idat_idx: int) -> LevelScattering:
    """
    Parse a level scattering distribution (Law 3).
    
    According to Table 34, LAW=3 contains:
    - LDAT(1): (A + 1)/A|Q|
    - LDAT(2): (A / (A + 1))^2
    
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
    LevelScattering
        Level scattering distribution object
    """
    # Create a new distribution object using the base properties
    distribution = LevelScattering(
        law=base_dist.law,
        idat=base_dist.idat
    )
    
    # Copy applicability data from base_dist
    distribution.applicability_energies = base_dist.applicability_energies
    distribution.applicability_probabilities = base_dist.applicability_probabilities
    distribution.nbt = base_dist.nbt
    distribution.interp = base_dist.interp
    
    # Read the two parameters for Law 3
    if idat_idx + 1 < len(ace.xss_data):
        # LDAT(1): (A + 1)/A|Q|
        aplusoaabsq_entry = ace.xss_data[idat_idx]
        distribution.aplusoaabsq = aplusoaabsq_entry.value
        
        # LDAT(2): (A / (A + 1))^2
        asquare_entry = ace.xss_data[idat_idx + 1]
        distribution.asquare = asquare_entry.value
    
    return distribution