import logging
from typing import List, Optional
from mcnpy.ace.ace import Ace
from mcnpy.ace.classes.particle_production_locators import ParticleProductionLocators, ParticleLocatorSet

# Setup logger
logger = logging.getLogger(__name__)

def read_particle_production_locators_block(ace: Ace, debug=False) -> None:
    """
    Read the IXS block containing particle production locators.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with particle production locator data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug:
        logger.debug("\n===== PARTICLE PRODUCTION LOCATORS BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize container if not already present
    if not hasattr(ace, "particle_production_locators"):
        ace.particle_production_locators = ParticleProductionLocators()
    
    # Check if we have the necessary data
    if (not ace.header or not ace.header.jxs_array or not ace.header.nxs_array or 
        not ace.xss_data or not ace.secondary_particles or not ace.secondary_particles.has_data):
        if debug:
            logger.debug("Skipping IXS block: required data missing")
        return
    
    # Get the number of particle types
    ntype = ace.secondary_particles.num_secondary_particles
    
    if debug:
        logger.debug(f"Number of secondary particle types: {ntype}")
    
    if ntype <= 0:
        if debug:
            logger.debug("No secondary particle types defined, nothing to process")
        return  # No particle types defined
    
    # Get the starting index for the IXS block (JXS(32))
    next_idx = ace.header.jxs_array[31]  # JXS(32) - convert to 0-indexed array
    
    if debug:
        logger.debug(f"JXS(32) = {next_idx} → Locator for IXS block (FORTRAN 1-indexed)")
    
    if next_idx <= 0 or next_idx > len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid index or no IXS block: JXS(32)={next_idx}")
        return  # Invalid index
    
    # Convert to 0-indexed
    next_idx -= 1
    
    if debug:
        logger.debug(f"IXS block starts at index {next_idx} (0-indexed)")
    
    # Read the locators for each particle type
    for j in range(1, ntype + 1):
        # Calculate the starting index for this particle type
        ltype = next_idx + 10 * (j - 1)
        
        if debug:
            logger.debug(f"\nProcessing particle type {j}:")
            logger.debug(f"  Index calculation: next_idx + 10*(j-1) = {next_idx} + 10*({j}-1) = {ltype}")
        
        # Check if we have enough data
        if ltype + 9 >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: Not enough data for particle type {j}: need 10 entries, but only {len(ace.xss_data) - ltype} available")
            break  # Not enough data for this particle type
        
        # Extract all locator values for debugging
        if debug:
            locator_names = ["HPD", "MTRH", "TYRH", "LSIGH", "SIGH", "LANDH", "ANDH", "LDLWH", "DLWH", "YH"]
            for k in range(10):
                logger.debug(f"  {locator_names[k]} = {int(ace.xss_data[ltype + k].value)}")
        
        # Create a locator set for this particle type
        locator_set = ParticleLocatorSet(
            hpd=int(ace.xss_data[ltype].value),
            mtrh=int(ace.xss_data[ltype + 1].value),
            tyrh=int(ace.xss_data[ltype + 2].value),
            lsigh=int(ace.xss_data[ltype + 3].value),
            sigh=int(ace.xss_data[ltype + 4].value),
            landh=int(ace.xss_data[ltype + 5].value),
            andh=int(ace.xss_data[ltype + 6].value),
            ldlwh=int(ace.xss_data[ltype + 7].value),
            dlwh=int(ace.xss_data[ltype + 8].value),
            yh=int(ace.xss_data[ltype + 9].value)
        )
        
        ace.particle_production_locators.locator_sets.append(locator_set)
        
        if debug:
            logger.debug(f"  Successfully read locators for particle type {j}")
    
    if ace.particle_production_locators.locator_sets:
        ace.particle_production_locators.has_data = True
        if debug:
            logger.debug(f"Successfully read locators for {len(ace.particle_production_locators.locator_sets)} particle types")
