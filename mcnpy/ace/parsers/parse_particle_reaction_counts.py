import logging
from typing import List, Optional
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.particle_reaction_counts import ParticleReactionCounts

# Setup logger
logger = logging.getLogger(__name__)

def read_particle_reaction_counts_block(ace: Ace, debug=False) -> None:
    """
    Read the NTRO block containing the number of reactions per particle type.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with particle reaction count data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug:
        logger.debug("\n===== PARTICLE REACTION COUNTS BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize container if not already present
    if not hasattr(ace, "particle_reaction_counts"):
        ace.particle_reaction_counts = ParticleReactionCounts()
    
    # Check if we have the necessary data
    if (not ace.header or not ace.header.jxs_array or not ace.header.nxs_array or 
        not ace.xss_data or not ace.secondary_particles or not ace.secondary_particles.has_data):
        if debug:
            logger.debug("Skipping NTRO block: required data missing")
        return
    
    # Get the number of particle types
    ntype = ace.secondary_particles.num_secondary_particles
    
    if debug:
        logger.debug(f"Number of secondary particle types: {ntype}")
    
    if ntype <= 0:
        if debug:
            logger.debug("No particle types defined, nothing to process")
        return  # No particle types defined
    
    # Get the starting index for the NTRO block (JXS(31))
    ltype_idx = ace.header.jxs_array[31]  # JXS(31)
    
    if debug:
        logger.debug(f"JXS(31) = {ltype_idx} → Locator for NTRO block")
    
    if ltype_idx <= 0 or ltype_idx > len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid index or no NTRO block: JXS(31)={ltype_idx}")
        return  # Invalid index
    
    if debug:
        logger.debug(f"NTRO block starts at index {ltype_idx}")
    
    # Check if we have enough data
    if ltype_idx + ntype - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data for NTRO block: need {ntype} entries, but only {len(ace.xss_data) - ltype_idx} available")
        return  # Not enough data
    
    # Read the number of reactions for each particle type
    for i in range(ntype):
        reaction_count = int(ace.xss_data[ltype_idx + i].value)
        ace.particle_reaction_counts.reaction_counts.append(reaction_count)
        
        if debug:
            logger.debug(f"  Particle type {i+1}: {reaction_count} reactions")
    
    if ace.particle_reaction_counts.reaction_counts:
        ace.particle_reaction_counts.has_data = True
        if debug:
            logger.debug(f"Successfully read reaction counts for {len(ace.particle_reaction_counts.reaction_counts)} particle types")
