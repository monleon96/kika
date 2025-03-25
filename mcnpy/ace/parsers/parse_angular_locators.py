import logging
from mcnpy.ace.ace import Ace
from mcnpy.ace.classes.angular_distribution.angular_locators import AngularDistributionLocators

# Setup logger
logger = logging.getLogger(__name__)

def read_angular_locator_blocks(ace: Ace, debug=False) -> None:
    """
    Read the LAND, LANDP, and LANDH blocks from the ACE file.
    
    These blocks contain locators (indices) to the angular distribution data:
    - LAND: For incident neutron reactions (JXS[7])
    - LANDP: For photon production reactions (JXS[15])
    - LANDH: For other particle production reactions (accessed through JXS[31])
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If required data is missing or indices are invalid
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        if debug:
            logger.debug("Skipping angular locator blocks: required data missing")
        raise ValueError("Cannot read angular locator blocks: header or XSS data missing")
    
    if debug:
        logger.debug("\n===== ANGULAR DISTRIBUTION LOCATOR BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Create a new container if not present
    if ace.angular_locators is None:
        ace.angular_locators = AngularDistributionLocators()
    
    # Get the NXS values for number of MT values
    num_reactions = ace.header.num_reactions  # NXS(4)
    num_secondary_neutron_reactions = ace.header.num_secondary_neutron_reactions  # NXS(5)
    num_photon_production_reactions = ace.header.num_photon_production_reactions  # NXS(6)
    num_particle_types = ace.header.num_particle_types  # NXS(7)
    
    if debug:
        logger.debug(f"NXS(4) = {num_reactions} → Total number of reactions")
        logger.debug(f"NXS(5) = {num_secondary_neutron_reactions} → Number of secondary neutron reactions")
        logger.debug(f"NXS(6) = {num_photon_production_reactions} → Number of photon production reactions")
        logger.debug(f"NXS(7) = {num_particle_types} → Number of particle types")
    
    # Store the number of MT values in the angular_locators container
    ace.angular_locators.num_neutron_reactions = num_reactions
    ace.angular_locators.num_secondary_neutron_reactions = num_secondary_neutron_reactions
    ace.angular_locators.num_photon_production_reactions = num_photon_production_reactions
    ace.angular_locators.num_particle_types = num_particle_types
    
    # Get the JXS pointers for each block
    land_idx = ace.header.jxs_array[7] - 1  # JXS(8), convert to 0-indexed
    landp_idx = ace.header.jxs_array[15] - 1  # JXS(16), convert to 0-indexed
    particle_pointer_idx = ace.header.jxs_array[30] - 1  # JXS(31), convert to 0-indexed
    
    if debug:
        logger.debug(f"JXS(8) = {land_idx+1} → Locator for LAND block (FORTRAN 1-indexed)")
        logger.debug(f"JXS(16) = {landp_idx+1} → Locator for LANDP block (FORTRAN 1-indexed)")
        logger.debug(f"JXS(31) = {particle_pointer_idx+1} → Locator for particle information (FORTRAN 1-indexed)")
    
    # Read LAND block (incident neutron reactions with secondary neutrons)
    if num_secondary_neutron_reactions > 0:
        if debug:
            logger.debug("\n----- LAND Block -----")
        read_land_block(ace, land_idx, num_secondary_neutron_reactions, debug)
    elif debug:
        logger.debug("No LAND block to process (no secondary neutron reactions)")
    
    # Read LANDP block (photon production)
    if num_photon_production_reactions > 0:
        if debug:
            logger.debug("\n----- LANDP Block -----")
        read_landp_block(ace, landp_idx, num_photon_production_reactions, debug)
    elif debug:
        logger.debug("No LANDP block to process (no photon production reactions)")
    
    # Read LANDH block (other particle production)
    if num_particle_types > 0:
        if debug:
            logger.debug("\n----- LANDH Block -----")
        read_landh_block(ace, particle_pointer_idx, num_particle_types, debug)
    elif debug:
        logger.debug("No LANDH block to process (no particle types)")


def read_land_block(ace: Ace, land_idx: int, num_reactions: int, debug=False) -> None:
    """
    Read the LAND block for incident neutron reaction angular distribution locators.
    
    This block contains locators (LOCB values) that point to angular distributions:
    - LOCB1 = LAND[0]: Angular distribution for elastic scattering
    - LOCB2 = LAND[1]: Angular distribution for the first reaction
    - ...
    - LOCB_NMT = LAND[NMT-1]: Angular distribution for the last reaction
    
    These locators are relative offsets from AND=JXS(9). To get the actual
    XSS index, use: AND + LOCB - 1
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    land_idx : int
        Starting index of the LAND block in the XSS array
    num_reactions : int
        Number of reactions (NXS(5) for secondary neutron reactions)
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the LAND block data is missing or invalid
    """
    if land_idx < 0:
        if debug:
            logger.debug(f"Invalid LAND block index: {land_idx+1}")
        raise ValueError(f"Invalid LAND block index: {land_idx+1}")
    
    if debug:
        logger.debug(f"LAND block starts at index {land_idx} (0-indexed)")
        logger.debug(f"Number of reactions (including elastic): {num_reactions+1}")
    
    if land_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"LAND block index out of bounds: {land_idx} >= {len(ace.xss_data)}")
        raise ValueError(f"LAND block index out of bounds: {land_idx} >= {len(ace.xss_data)}")
    
    # Make sure we have enough data (including the elastic scattering locator)
    total_entries = num_reactions + 1  # +1 for elastic scattering
    if land_idx + total_entries > len(ace.xss_data):
        if debug:
            logger.debug(f"LAND block truncated: need {total_entries} entries, but only {len(ace.xss_data) - land_idx} available")
        raise ValueError(f"LAND block truncated: need {total_entries} entries, but only {len(ace.xss_data) - land_idx} available")
    
    # LOCB1 - Angular distribution for elastic scattering (first element)
    # This is a special case stored at the beginning of the LAND block
    ace.angular_locators.elastic_scattering = ace.xss_data[land_idx]
    
    if debug:
        logger.debug(f"Elastic scattering locator: {ace.angular_locators.elastic_scattering.value}")
    
    # LOCB2 through LOCB_NMT - Angular distributions for other neutron reactions
    # These are stored sequentially after the elastic scattering locator
    ace.angular_locators.incident_neutron = ace.xss_data[land_idx + 1:land_idx + 1 + num_reactions]
    
    if debug:
        logger.debug(f"Successfully read {len(ace.angular_locators.incident_neutron)} neutron reaction angular distribution locators")


def read_landp_block(ace: Ace, landp_idx: int, num_photon_reactions: int, debug=False) -> None:
    """
    Read the LANDP block for photon production angular distribution locators.
    
    This block contains locators (LOCB values) that point to photon production angular distributions:
    - LOCB1 = LANDP[0]: Angular distribution for first photon production reaction
    - LOCB2 = LANDP[1]: Angular distribution for second photon production reaction
    - ...
    - LOCB_NMT = LANDP[NMT-1]: Angular distribution for the last photon production reaction
    
    These locators are relative offsets from ANDP=JXS(17). To get the actual
    XSS index, use: ANDP + LOCB - 1
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    landp_idx : int
        Starting index of the LANDP block in the XSS array
    num_photon_reactions : int
        Number of photon production reactions (NXS(6))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the LANDP block data is missing or invalid
    """
    if landp_idx < 0:
        if debug:
            logger.debug(f"Invalid LANDP block index: {landp_idx+1}")
        raise ValueError(f"Invalid LANDP block index: {landp_idx+1}")
    
    if debug:
        logger.debug(f"LANDP block starts at index {landp_idx} (0-indexed)")
        logger.debug(f"Number of photon production reactions: {num_photon_reactions}")
    
    if landp_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"LANDP block index out of bounds: {landp_idx} >= {len(ace.xss_data)}")
        raise ValueError(f"LANDP block index out of bounds: {landp_idx} >= {len(ace.xss_data)}")
    
    # Make sure we have enough data
    if landp_idx + num_photon_reactions > len(ace.xss_data):
        if debug:
            logger.debug(f"LANDP block truncated: need {num_photon_reactions} entries, but only {len(ace.xss_data) - landp_idx} available")
        raise ValueError(f"LANDP block truncated: need {num_photon_reactions} entries, but only {len(ace.xss_data) - landp_idx} available")
    
    # Read the photon production locators (LOCB1 through LOCB_NMT)
    # Each entry points to the angular distribution for a photon production reaction
    ace.angular_locators.photon_production = ace.xss_data[landp_idx:landp_idx + num_photon_reactions]
    
    if debug:
        logger.debug(f"Successfully read {len(ace.angular_locators.photon_production)} photon production angular distribution locators")


def read_landh_block(ace: Ace, particle_pointer_idx: int, num_particle_types: int, debug=False) -> None:
    """
    Read the LANDH block for other particle production angular distribution locators.
    
    This block contains locators for angular distributions of particle production reactions:
    - For each particle type i (1 to NXS(7)):
      - The number of reactions is stored at XSS(JXS(31)+i-1)
      - The LANDH pointer is at XSS(JXS(32)+10*(i-1)+5)
      - This pointer points to a list of locators, one for each MT number
    
    These locators are relative offsets from ANDH=XSS(JXS(32)+10*(i-1)+6). To get the actual
    XSS index, use: ANDH + LOCB - 1
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    particle_pointer_idx : int
        Starting index of the particle pointer block (JXS(31)) in the XSS array
    num_particle_types : int
        Number of particle types (NXS(7))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the LANDH block data is missing or invalid
    """
    if particle_pointer_idx < 0:
        if debug:
            logger.debug(f"Invalid particle pointer block index: {particle_pointer_idx+1}")
        raise ValueError(f"Invalid particle pointer block index: {particle_pointer_idx+1}")
    
    if debug:
        logger.debug(f"Particle pointer index: {particle_pointer_idx} (0-indexed)")
        logger.debug(f"Number of particle types: {num_particle_types}")
    
    if particle_pointer_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Particle pointer block index out of bounds: {particle_pointer_idx} >= {len(ace.xss_data)}")
        raise ValueError(f"Particle pointer block index out of bounds: {particle_pointer_idx} >= {len(ace.xss_data)}")
    
    # Get the base index for JXS(32)
    if len(ace.header.jxs_array) <= 31:
        if debug:
            logger.debug("JXS array too short: missing JXS(32) index")
        raise ValueError("JXS array too short: missing JXS(32) index")
    
    jxs32_idx = ace.header.jxs_array[31] - 1  # JXS(32), convert to 0-indexed
    if jxs32_idx < 0:
        if debug:
            logger.debug(f"Invalid JXS(32) value: {jxs32_idx + 1}")
        raise ValueError(f"Invalid JXS(32) value: {jxs32_idx + 1}")
    
    if debug:
        logger.debug(f"JXS(32) index: {jxs32_idx} (0-indexed)")
    
    # Initialize the particle production locators list
    ace.angular_locators.particle_production = []
    
    # For each particle type i, process its angular distribution locators
    for i in range(1, num_particle_types + 1):
        if debug:
            logger.debug(f"\nProcessing particle type {i}:")
        
        # 1. Get the number of MT values for this particle type
        mt_count_idx = particle_pointer_idx + (i - 1)
        
        if debug:
            logger.debug(f"  MT count index: particle_pointer_idx + (i-1) = {particle_pointer_idx} + ({i-1}) = {mt_count_idx}")
        
        if mt_count_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: MT count index out of bounds: {mt_count_idx} >= {len(ace.xss_data)}")
            raise ValueError(f"MT count index out of bounds for particle type {i}: {mt_count_idx} >= {len(ace.xss_data)}")
        
        num_mt_values = int(ace.xss_data[mt_count_idx].value)
        
        if debug:
            logger.debug(f"  Number of MT values: {num_mt_values}")
        
        # 2. Get the LANDH pointer from the particle's data block
        # LANDH pointer is at XSS(JXS(32)+10*(i-1)+5)
        landh_pointer_idx = jxs32_idx + 10 * (i - 1) + 5 - 1  # -1 for 0-indexing
        
        if debug:
            logger.debug(f"  LANDH pointer index: jxs32_idx + 10*(i-1) + 4 = {jxs32_idx} + 10*({i-1}) + 4 = {landh_pointer_idx}")
        
        if landh_pointer_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: LANDH pointer index out of bounds: {landh_pointer_idx} >= {len(ace.xss_data)}")
            raise ValueError(f"LANDH pointer index out of bounds for particle type {i}: {landh_pointer_idx} >= {len(ace.xss_data)}")
        
        # Get the actual pointer value
        landh_pointer = int(ace.xss_data[landh_pointer_idx].value)
        
        if debug:
            logger.debug(f"  LANDH pointer value: {landh_pointer}")
        
        # 3. Special case: If LANDH pointer is 0, there's no angular distribution data for this particle
        if landh_pointer <= 0:
            if debug:
                logger.debug("  No angular distribution data for this particle (LANDH pointer ≤ 0)")
            ace.angular_locators.particle_production.append([])
            continue
        
        # 4. Convert to 0-indexed for accessing the XSS array
        landh_pointer = landh_pointer - 1
        
        if debug:
            logger.debug(f"  LANDH pointer (0-indexed): {landh_pointer}")
        
        # 5. Make sure we have enough data for all MT values
        if landh_pointer + num_mt_values > len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: LANDH block truncated: need {num_mt_values} entries, but only {len(ace.xss_data) - landh_pointer} available")
            raise ValueError(f"LANDH block truncated for particle type {i}: need {num_mt_values} entries, but only {len(ace.xss_data) - landh_pointer} available")
        
        # 6. Read the locators for this particle type
        # Each locator points to the angular distribution for the corresponding MT number
        locators = ace.xss_data[landh_pointer:landh_pointer + num_mt_values]
        ace.angular_locators.particle_production.append(locators)
        
        if debug:
            logger.debug(f"  Successfully read {len(locators)} angular distribution locators")
    
    if debug:
        logger.debug(f"\nSuccessfully read angular distribution locators for {len(ace.angular_locators.particle_production)} particle types")
