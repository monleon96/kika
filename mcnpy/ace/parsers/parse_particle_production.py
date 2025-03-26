import logging
from typing import List, Optional
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.particle_production import ParticleProductionTypes

# Setup logger
logger = logging.getLogger(__name__)

def read_particle_types_block(ace: Ace, debug=False) -> None:
    """
    Read the PTYPE block containing particle type identifiers.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with particle type data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug:
        logger.debug("\n===== PTYPE BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize container if not already present
    if not hasattr(ace, "particle_types"):
        ace.particle_types = ParticleProductionTypes()
    
    # Check if we have the necessary data and if PTYPE block exists
    if not ace.header or not ace.header.jxs_array or not ace.header.nxs_array or not ace.xss_data:
        if debug:
            logger.debug("Skipping PTYPE block: required data missing")
        return
    
    # Get the number of particle types (NTYPE = NXS(7))
    ntype = ace.header.nxs_array[7]  # NXS(7)
    
    if debug:
        logger.debug(f"NXS(7) = {ntype} → Number of particle types")
    
    if ntype <= 0:
        if debug:
            logger.debug("No particle types defined, nothing to process")
        return  # No particle types defined
    
    # Get the starting index for the PTYPE block (LTYPE = JXS(30))
    ltype_idx = ace.header.jxs_array[30]  # JXS(30)
    
    if debug:
        logger.debug(f"JXS(30) = {ltype_idx} → Locator for PTYPE block")
    
    if ltype_idx <= 0 or ltype_idx > len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid index or no PTYPE block: JXS(30)={ltype_idx}")
        return  # Invalid index
    
    if debug:
        logger.debug(f"PTYPE block starts at index {ltype_idx}")
    
    # Check if we have enough data
    if ltype_idx + ntype - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data for PTYPE block: need {ntype} entries, but only {len(ace.xss_data) - ltype_idx} available")
        return  # Not enough data
    
    # Read the particle type identifiers
    for i in range(ntype):
        particle_id_entry = ace.xss_data[ltype_idx + i]
        particle_id = int(particle_id_entry.value)
        ace.particle_types.particle_ids.append(particle_id)
        
        if debug:
            logger.debug(f"  Particle type {i+1}: ID={particle_id}")
    
    if ace.particle_types.particle_ids:
        ace.particle_types.has_data = True
        if debug:
            logger.debug(f"Successfully read {len(ace.particle_types.particle_ids)} particle type identifiers")

def read_secondary_particle_types_block(ace: Ace, debug=False) -> None:
    """
    Read the PTYPE block containing secondary particle production types.
    
    This block identifies which types of secondary particles (neutrons, protons, etc.)
    can be produced in reactions, and for which the ACE file contains production data.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with secondary particle production data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug:
        logger.debug("\n===== SECONDARY PARTICLE TYPES BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize container if not already present
    if not hasattr(ace, "secondary_particles"):
        ace.secondary_particles = ParticleProductionTypes()
    
    # Check if we have the necessary data and if PTYPE block exists
    if not ace.header or not ace.header.jxs_array or not ace.header.nxs_array or not ace.xss_data:
        if debug:
            logger.debug("Skipping secondary particles block: required data missing")
        return
    
    # Get the number of particle types (NTYPE = NXS(7))
    ntype = ace.header.nxs_array[7]  # NXS(7)
    
    if debug:
        logger.debug(f"NXS(7) = {ntype} → Number of secondary particle types")
    
    if ntype <= 0:
        if debug:
            logger.debug("No secondary particle types defined, nothing to process")
        return  # No particle types defined
    
    # Get the starting index for the PTYPE block (LTYPE = JXS(30))
    ltype_idx = ace.header.jxs_array[30]  # JXS(30)
    
    if debug:
        logger.debug(f"JXS(30) = {ltype_idx} → Locator for secondary particles block")
    
    if ltype_idx <= 0 or ltype_idx > len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid index or no secondary particles block: JXS(30)={ltype_idx}")
        return  # Invalid index
    
    if debug:
        logger.debug(f"Secondary particles block starts at index {ltype_idx}")
    
    # Check if we have enough data
    if ltype_idx + ntype - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data for secondary particles block: need {ntype} entries, but only {len(ace.xss_data) - ltype_idx} available")
        return  # Not enough data
    
    # Read the particle type identifiers
    for i in range(ntype):
        particle_id_entry = ace.xss_data[ltype_idx + i]
        particle_id = int(particle_id_entry.value)
        ace.secondary_particles.particle_ids.append(particle_id)
        
        if debug:
            logger.debug(f"  Secondary particle type {i+1}: ID={particle_id}")
    
    if ace.secondary_particles.particle_ids:
        ace.secondary_particles.has_data = True
        if debug:
            logger.debug(f"Successfully read {len(ace.secondary_particles.particle_ids)} secondary particle type identifiers")
