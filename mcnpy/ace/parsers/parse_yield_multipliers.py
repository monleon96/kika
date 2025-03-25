import logging
from typing import List, Optional
from mcnpy.ace.ace import Ace
from mcnpy.ace.classes.yield_multipliers import PhotonYieldMultipliers, ParticleYieldMultipliers

# Setup logger
logger = logging.getLogger(__name__)

def read_yield_multiplier_blocks(ace: Ace, debug=False) -> None:
    """
    Read the YP and YH blocks containing yield multiplier MT numbers.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with yield multiplier data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug:
        logger.debug("\n===== YIELD MULTIPLIER BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize containers if not already present
    if not hasattr(ace, "photon_yield_multipliers"):
        ace.photon_yield_multipliers = PhotonYieldMultipliers()
    
    if not hasattr(ace, "particle_yield_multipliers"):
        ace.particle_yield_multipliers = ParticleYieldMultipliers()
    
    # Read photon yield multipliers (YP block)
    read_photon_yield_multipliers(ace, debug)
    
    # Read particle yield multipliers (YH block)
    read_particle_yield_multipliers(ace, debug)

def read_photon_yield_multipliers(ace: Ace, debug=False) -> None:
    """
    Read photon yield multiplier MT numbers (YP block).
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    # Check if we have the necessary data
    if not ace.header or not ace.header.nxs_array or not ace.xss_data:
        if debug:
            logger.debug("Skipping YP block: required data missing")
        return
    
    if debug:
        logger.debug("\n----- YP Block -----")
    
    # For YP, LY = NXS(6)
    ly = ace.header.nxs_array[5]  # NXS(6) - Number of MT numbers in photon production
    
    if debug:
        logger.debug(f"NXS(6) = {ly} → Number of photon production MT numbers")
    
    if ly <= 0 or ly >= len(ace.xss_data):
        if debug:
            logger.debug(f"No YP block present: NXS(6)={ly}")
        return
    
    # Convert to 0-indexed
    ly_0 = ly - 1
    
    if debug:
        logger.debug(f"LY 0-indexed = {ly_0}")
    
    # Read number of MTs
    nyp_entry = ace.xss_data[ly_0]
    nyp = int(nyp_entry.value)
    
    if debug:
        logger.debug(f"NYP = {nyp} → Number of yield multiplier MT numbers")
    
    if nyp <= 0 or ly_0 + nyp >= len(ace.xss_data):
        if debug:
            logger.debug(f"No yield multiplier MT numbers: NYP={nyp}")
        return
    
    # Read the MT numbers
    if debug:
        logger.debug(f"Reading MT numbers from XSS[{ly_0}:{ly_0+nyp}]")
    
    for i in range(nyp):
        mt_entry = ace.xss_data[ly_0 + i]
        mt = int(mt_entry.value)
        ace.photon_yield_multipliers.multiplier_mts.append(mt)
        
        if debug:
            logger.debug(f"  MT[{i}] = {mt}")
    
    if ace.photon_yield_multipliers.multiplier_mts:
        ace.photon_yield_multipliers.has_data = True
        if debug:
            logger.debug(f"Successfully read {len(ace.photon_yield_multipliers.multiplier_mts)} photon yield multiplier MT numbers")

def read_particle_yield_multipliers(ace: Ace, debug=False) -> None:
    """
    Read particle yield multiplier MT numbers (YH block).
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    # Check if we have the necessary data
    if not ace.header or not ace.header.jxs_array or not ace.xss_data:
        if debug:
            logger.debug("Skipping YH block: required data missing")
        return
    
    if debug:
        logger.debug("\n----- YH Block -----")
    
    # Check if we have particle types
    if ace.header.nxs_array[12] <= 0:  # NXS(13) - Number of particle types
        if debug:
            logger.debug(f"No particle types: NXS(13)={ace.header.nxs_array[12]}")
        return
    
    n_types = ace.header.nxs_array[12]
    
    if debug:
        logger.debug(f"Number of particle types: {n_types}")
    
    # Iterate through each particle type
    for i in range(1, n_types + 1):
        if debug:
            logger.debug(f"\nProcessing particle type {i}:")
        
        # Get JXS(32) index
        jxs32_idx = ace.header.jxs_array[31] - 1  # JXS(32) - convert to 0-indexed
        
        if debug:
            logger.debug(f"  JXS(32) 0-indexed = {jxs32_idx}")
        
        if jxs32_idx < 0 or jxs32_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: JXS(32) index {jxs32_idx} is out of bounds ({len(ace.xss_data)})")
            continue
        
        # Calculate JED index: JED = XSS(JXS(32) + 10 * (i - 1) + 8)
        jed_idx = jxs32_idx + 10 * (i - 1) + 8
        
        if debug:
            logger.debug(f"  JED index: jxs32_idx + 10*(i-1) + 8 = {jxs32_idx} + 10*({i}-1) + 8 = {jed_idx}")
        
        if jed_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: JED index {jed_idx} is out of bounds ({len(ace.xss_data)})")
            continue
        
        # Get LY value from JED
        ly_entry = ace.xss_data[jed_idx]
        ly = int(ly_entry.value)
        
        if debug:
            logger.debug(f"  LY = {ly} → Location of yield multiplier data (FORTRAN 1-indexed)")
        
        if ly <= 0 or ly >= len(ace.xss_data):
            if debug:
                logger.debug(f"  No yield multiplier data: LY={ly}")
            continue
        
        # Convert to 0-indexed
        ly_0 = ly - 1
        
        if debug:
            logger.debug(f"  LY 0-indexed = {ly_0}")
        
        # Read number of MTs
        nyh_entry = ace.xss_data[ly_0]
        nyh = int(nyh_entry.value)
        
        if debug:
            logger.debug(f"  NYH = {nyh} → Number of yield multiplier MT numbers")
        
        if nyh <= 0 or ly_0 + nyh >= len(ace.xss_data):
            if debug:
                logger.debug(f"  No yield multiplier MT numbers: NYH={nyh}")
            continue
        
        # Read the MT numbers for this particle type
        if debug:
            logger.debug(f"  Reading MT numbers from XSS[{ly_0+1}:{ly_0+nyh+1}]")
        
        particle_mts = []
        for j in range(nyh):
            mt_entry = ace.xss_data[ly_0 + 1 + j]  # Skip the first value (which is NYH)
            mt = int(mt_entry.value)
            particle_mts.append(mt)
            
            if debug:
                logger.debug(f"    MT[{j}] = {mt}")
        
        if particle_mts:
            ace.particle_yield_multipliers.particle_multipliers[i] = particle_mts
            ace.particle_yield_multipliers.multiplier_mts.extend(particle_mts)
            ace.particle_yield_multipliers.has_data = True
            
            if debug:
                logger.debug(f"  Successfully read {len(particle_mts)} yield multiplier MT numbers for particle type {i}")
